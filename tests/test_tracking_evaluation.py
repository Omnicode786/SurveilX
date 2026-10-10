import itertools

import numpy as np
import pytest

from surveilx.matching import maximum_assignment
from surveilx.vision import Detection, Tracker
from training.tracking_evaluation import evaluate_tracking


def obj(identity=1, label="person", box=None):
    return {"track_id": identity, "label": label, "box": box or [.1, .1, .5, .5]}


def payload(frames):
    return {"schema_version": 1, "synthetic": True, "sequences": [{"name": "fixture", "frames": [
        {"index": i, "ground_truth": gt, "predictions": pred} for i, (gt, pred) in enumerate(frames)]}]}


def test_fallback_assignment_equals_exhaustive_optimum_and_native():
    rng = np.random.default_rng(12)
    for n, m in ((1, 4), (3, 4), (4, 3), (3, 3)):
        for _ in range(6):
            weights = rng.integers(-4, 10, (n, m))
            small = weights if n <= m else weights.T
            optimum = max(sum(small[row, col] for row, col in enumerate(cols)) for cols in itertools.permutations(range(small.shape[1]), small.shape[0]))
            for native in (False, True):
                pairs = maximum_assignment(weights, native=native)
                assert sum(weights[row, col] for row, col in pairs) == optimum
                assert len({r for r, _ in pairs}) == len({c for _, c in pairs}) == min(n, m)
    assert maximum_assignment(np.zeros((0, 3))) == []
    with pytest.raises(ValueError, match="finite"):
        maximum_assignment([[float("nan")]])


def test_global_tracker_preserves_best_ids_independent_of_detection_order():
    for reverse in (False, True):
        tracker = Tracker()
        tracker.update([Detection([.1, .1, .5, .5], "person", .9), Detection([.3, .1, .7, .5], "person", .9)], now=0)
        frames = [Detection([.15, .1, .55, .5], "person", .9), Detection([.1, .1, .5, .5], "person", .9)]
        result = tracker.update(frames[::-1] if reverse else frames, now=1)
        assert {tuple(d.box): d.track_id for d in result} == {tuple(frames[0].box): 2, tuple(frames[1].box): 1}
        changed = tracker.update([Detection([.1, .1, .5, .5], "car", .9)], now=1.5)[0]
        assert changed.track_id == 3
        expired = tracker.update([Detection([.1, .1, .5, .5], "person", .9)], now=4)[0]
        assert expired.track_id == 4


def test_tracking_perfect_switches_gap_duplicates_and_wrong_classes():
    result = evaluate_tracking(payload([([obj()], [obj(8)]), ([obj()], [obj(8)])]))["metrics"]
    assert result["mota"] == result["idf1"] == result["motp_mean_iou"] == 1
    result = evaluate_tracking(payload([([obj()], [obj(8)]), ([obj()], [obj(9)])]))["metrics"]
    assert result["id_switches"] == 1 and result["mota"] == .5 and result["idf1"] == .5
    result = evaluate_tracking(payload([([obj()], [obj(8)]), ([obj()], []), ([obj()], [obj(9)])]))["metrics"]
    assert result["false_negatives"] == 1 and result["id_switches"] == 1
    result = evaluate_tracking(payload([([obj()], [obj(8), obj(9), obj(10, "car")])]))
    assert result["metrics"]["false_positives"] == 2 and result["metrics"]["idf1"] == .5
    assert result["per_class"]["car"]["gt"] == 0 and result["per_class"]["car"]["mota"] is None


def test_identity_global_assignment_uses_all_eligible_pairs_not_greedy_matches():
    # Ambiguous overlapping boxes: ID assignment considers all valid pairs.
    data = payload([([obj(1), obj(2)], [obj(10), obj(20)]), ([obj(1), obj(2)], [obj(20), obj(10)])])
    assert evaluate_tracking(data)["metrics"]["idf1"] == 1
    duplicate = payload([([obj()], [obj(10), obj(10)])])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_tracking(duplicate)
    data["sequences"].append({"name": "second", "frames": [{"index": 0, "ground_truth": [obj()], "predictions": [obj(10)]}]})
    result = evaluate_tracking(data)
    assert result["metrics"]["idf1"] == 1 and result["metrics"]["gt"] == 5


def test_tracking_empty_missing_frame_and_invalid_box_are_explicit():
    assert evaluate_tracking(payload([([], [])]))["metrics"]["idf1"] is None
    assert evaluate_tracking(payload([([obj()], [])]))["metrics"]["idf1"] == 0
    assert evaluate_tracking(payload([([], [obj()])]))["metrics"]["false_positives"] == 1
    data = payload([([obj()], [obj()]), ([obj()], [obj()])])
    data["sequences"][0]["frames"][1]["index"] = 2
    with pytest.raises(ValueError, match="consecutive"):
        evaluate_tracking(data)
    with pytest.raises(ValueError, match="normalized"):
        evaluate_tracking(payload([([obj(box=[0, 0, 1, float("nan")])], [])]))
