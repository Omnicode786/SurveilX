"""Compute elimination, not just hidden labels. No model training."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sqlalchemy import select

from surveilx.detection_preferences import DetectionSelection, infer_selected, selected_class_ids
from surveilx.model_router import CategoryRouter
from surveilx.vision import Detection, YOLODetector
from surveilx.detector_expert import AdaptedYOLODetector
from test_runtime import make_capture
from test_model_selection import candidate


@pytest.mark.parametrize("adapter", [YOLODetector, AdaptedYOLODetector])
def test_yolo_filter_before_nms_and_clear_between_cameras(adapter):
    calls = []
    boxes = SimpleNamespace(xyxyn=torch.tensor([[.1, .1, .4, .4]]),
                            cls=torch.tensor([1.]), conf=torch.tensor([.8]))
    result = SimpleNamespace(names={0: 'person', 1: 'fire'}, boxes=boxes)
    model = adapter.__new__(adapter)
    model.model = SimpleNamespace(names=result.names, predict=lambda *args, **kwargs: calls.append(kwargs) or [result])
    model.device = 'cpu'
    model.classes = ['person', 'fire']
    model.calibration = {'status': 'fitted', 'temperature': 1., 'threshold': .1}
    model.manifest = {'classes': model.classes, 'config': {'image_size': 64}, 'calibration': model.calibration}
    frame = np.zeros((40, 64, 3), dtype=np.uint8)
    assert infer_selected(model, frame, 64, DetectionSelection(disabled_labels=['person']))[0].label == 'fire'
    assert calls[-1]['classes'] == [1]
    infer_selected(model, frame, 64, DetectionSelection())
    assert calls[-1]['classes'] is None
    assert infer_selected(model, frame, 64, DetectionSelection(disabled_labels=['person', 'fire'])) == []
    assert len(calls) == 2
    assert selected_class_ids({4: 'traffic light', 7: 'person'}, DetectionSelection(disabled_labels=['traffic_light'])) == [7]


def test_router_skips_disabled_expert_frees_budget_and_recovers_enabled_routes():
    calls = []
    a, b = candidate('a'), candidate('b')
    for model in (a, b):
        version = model['version']
        def run(frame, resolution, selection, version=version):
            calls.append((version, selection.disabled_labels))
            return [Detection([0, 0, 1, 1], label, .9) for label in ('person', 'helmet') if selection.enabled(label)]
        model['expert'] = SimpleNamespace(manifest=model['manifest'], infer_selected=run)
    router = CategoryRouter([a, b], maximum_models=1)
    router.routes = [{'category': 'person', 'candidates': ['a']}, {'category': 'helmet', 'candidates': ['b']}]
    assert [d.label for d in router.infer_selected(None, 320, DetectionSelection(disabled_labels=['person']))] == ['helmet']
    assert calls == [('b', ['person'])]
    assert router.infer_selected(None, 320, DetectionSelection(disabled_labels=['person', 'helmet'])) == []
    assert len(calls) == 1 and router.last_routes == []
    # Camera preferences stay local; an all-on camera can use the same router.
    router.maximum_models = 2
    assert sorted(d.label for d in router.infer(None)) == ['helmet', 'person']
    # Shared expert runs once and receives the union of its enabled categories.
    router.routes[1]['candidates'] = ['a']
    calls.clear()
    assert len(router.infer(None)) == 2 and calls == [('a', [])]
    def fail(*args):
        raise RuntimeError('device unavailable')
    a['expert'].infer_selected = fail
    router.routes = [{'category': 'helmet', 'candidates': ['a', 'b']}]
    assert [d.label for d in router.infer_selected(None, 320, DetectionSelection(disabled_labels=['person']))] == ['helmet']
    assert router.last_routes[0]['version'] == 'b'


def test_all_off_skips_forward_history_tracking_and_decision_writes(client, monkeypatch):
    from surveilx.database import Record, transaction
    import surveilx.runtime as runtime_module
    runtime, capture = make_capture(client)
    runtime.set_detection_selection(capture.id, DetectionSelection(disabled_labels=['synthetic_entity']))
    assert not runtime.has_detection_work(capture)
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled work executed')
    monkeypatch.setattr(runtime.synthetic_detector, 'infer', forbidden)
    monkeypatch.setattr(capture.tracker, 'update', forbidden)
    monkeypatch.setattr(runtime_module.cv2, 'imencode', forbidden)
    runtime.infer(capture)
    assert capture.consumed == 1 and capture.inference_count == 0
    assert not capture.entity_history and not capture.scene_history
    assert capture.selection_work['primary_skipped']
    assert capture.selection_work['history_encoded'] is False
    assert capture.frame is not None  # Preview source survives.
    with transaction() as session:
        assert not session.scalars(select(Record).where(Record.kind == 'decision')).all()


def test_scene_event_still_runs_when_primary_classes_are_off(client):
    runtime, capture = make_capture(client)
    calls = []
    runtime.sva = SimpleNamespace(name='scene', version='scene', scene=True,
        manifest={'classes': ['normal', 'fighting'], 'domain': capture.environment, 'synthetic': True},
        infer=lambda history, *args, **kwargs: calls.append(len(history)) or {'decision': 'abstain'})
    runtime.set_detection_selection(capture.id, DetectionSelection(disabled_labels=['synthetic_entity']))
    assert runtime.has_detection_work(capture)
    runtime.infer(capture)
    assert calls == [1] and capture.scene_history and not capture.entity_history
    assert capture.inference_count == 0  # No object detector forward.
    runtime.set_detection_selection(capture.id, DetectionSelection(disabled_labels=['synthetic_entity', 'fighting']))
    assert not runtime.has_detection_work(capture)
    capture.sequence += 1
    runtime.infer(capture)
    assert calls == [1] and not capture.scene_history


def test_disabled_specialists_and_events_never_compete_for_service(client):
    runtime, capture = make_capture(client)
    calls = []
    def expert(label, scene=False):
        return SimpleNamespace(name=label, version=label, scene=scene,
            manifest={'classes': ['normal', label] if scene else [label], 'domain': capture.environment, 'synthetic': True},
            infer=lambda *args, **kwargs: calls.append(label) or [])
    runtime.experts = {'detection:fire': expert('fire'), 'detection:person': expert('person'),
                       'event:fighting': expert('fighting', True)}
    runtime.set_detection_selection(capture.id, DetectionSelection(disabled_labels=['person', 'fighting']))
    for _ in range(3):
        capture.sequence += 1
        runtime.infer(capture)
    assert calls == ['fire'] * 3
    assert capture.selection_work['enabled_detection_specialists'] == 1
    assert capture.selection_work['enabled_event_models'] == 0
    assert not capture.entity_history and not capture.scene_history


def test_disabled_baseline_never_executes_on_primary_failure(client):
    runtime, capture = make_capture(client)
    capture.synthetic = False
    def fail(*args):
        raise RuntimeError('primary failed')
    def forbidden(*args):
        raise AssertionError('Disabled HOG ran')
    runtime.yolo = SimpleNamespace(name='unknown-model', infer=fail)
    runtime.baseline = SimpleNamespace(name='opencv-hog-person-baseline', infer=forbidden)
    runtime.set_detection_selection(capture.id, DetectionSelection(disabled_labels=['person']))
    runtime.infer(capture)
    assert not capture.outputs and capture.inference_count == 0
    assert capture.selection_work['primary_skipped']


def test_idle_camera_does_not_reserve_scheduler_budget(client, monkeypatch):
    import surveilx.runtime as runtime_module
    runtime, capture = make_capture(client)
    runtime.set_detection_selection(capture.id, DetectionSelection(disabled_labels=['synthetic_entity']))
    capture.status = 'online'
    monkeypatch.setattr(runtime_module.IntervalGate, 'due', lambda *args: False)
    calls = []
    allocate = runtime.scheduler.allocate
    def one_tick(candidates, *args):
        calls.extend(candidates)
        runtime.stop_event.set()
        return allocate(candidates, *args)
    monkeypatch.setattr(runtime.scheduler, 'allocate', one_tick)
    runtime.run()
    assert len(calls) == 1 and not calls[0].available
    assert not runtime.decision['selected']
    assert capture.consumed == capture.sequence
    assert capture.inference_count == 0 and capture.frame is not None
