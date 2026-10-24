import copy
from types import SimpleNamespace

import pytest
import torch

from training.yolo_model import (SceneZoneAdapter, LocalDetailAdapter, ConditionedBlock,
                                 install_adapters, upgrade_adapters, adapter_version)


def test_local_upgrade_preserves_every_legacy_tensor_and_forward():
    from ultralytics.nn.tasks import DetectionModel
    torch.set_num_threads(2)
    old = install_adapters(DetectionModel('yolo11n.yaml', nc=4, verbose=False)).eval()
    # Include learned nonzero adapter state, not just an identity fixture.
    with torch.no_grad():
        for block in old.modules():
            if isinstance(block, SceneZoneAdapter):
                block.channel[-1].weight.normal_(0, .02)
                block.spatial[-1].weight.normal_(0, .02)
    new = upgrade_adapters(copy.deepcopy(old), 'local-v2')
    assert adapter_version(old) == 'v1' and adapter_version(new) == 'local-v2'
    for name, expected in old.state_dict().items():
        assert torch.equal(new.state_dict()[name], expected), name
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(old(x)[0], new(x)[0])
    added = sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters())
    assert 0 < added < 10_000
    assert upgrade_adapters(new, 'local-v2') is new
    with pytest.raises(ValueError, match='discard'):
        upgrade_adapters(new, 'v1')


def test_new_branch_has_gradients_and_bounded_residual():
    adapter = LocalDetailAdapter(8)
    original = SceneZoneAdapter(8)
    original.load_state_dict({k: v for k, v in adapter.state_dict().items() if k != 'local_detail.weight'})
    x = torch.randn(2, 8, 7, 9, requires_grad=True)
    assert torch.equal(adapter(x), original(x))
    adapter(x).square().mean().backward()
    assert torch.isfinite(adapter.local_detail.weight.grad).all()
    assert adapter.local_detail.weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.local_detail.weight.fill_(100)
        assert torch.all((adapter(x) - original(x)).abs() <= .10001 * x.abs() + 1e-6)


def test_trainer_restores_trained_local_branch(monkeypatch):
    from ultralytics.nn.tasks import DetectionModel
    from training.yolo_pipeline import ConditionedTrainer, DetectionTrainer
    original = DetectionModel('yolo11n.yaml', nc=4, verbose=False)
    parent = install_adapters(copy.deepcopy(original), 'local-v2')
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.add_(.01)
    monkeypatch.setattr(DetectionTrainer, 'get_model', lambda *args: copy.deepcopy(original))
    restored = ConditionedTrainer.get_model(object.__new__(ConditionedTrainer), weights=parent, verbose=False)
    assert adapter_version(restored) == 'local-v2'
    assert sum(isinstance(block, ConditionedBlock) for block in restored.modules()) == 3
    for name, expected in parent.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected), name


def test_campaign_command_preserves_explicit_architecture(monkeypatch, tmp_path):
    from scripts import train_accuracy_campaign as campaign
    monkeypatch.setattr(campaign, 'training_interpreter', lambda: 'python')
    monkeypatch.setattr(campaign, 'settings', SimpleNamespace(data_dir=tmp_path))
    command = campaign.command_for({'pipeline': 'yolo', 'dataset': 'data', 'parent': 'parent',
                                   'version': 'new', 'options': {'adapter-version': 'local-v2'}})
    assert command[-2:] == ['--adapter-version', 'local-v2']


def test_comparison_uses_matching_geometry_and_rejects_foreign_baseline():
    from scripts.train_accuracy_campaign import comparison
    parent = {'weights_sha256': 'parent', 'metrics': {'map50': .3, 'per_class': []}}
    candidate = {'task': 'detection', 'classes': [], 'dataset_sha256': 'dataset',
        'config': {'image_size': 640, 'preprocessing': 'letterbox'}, 'metrics': {'map50': .5},
        'parent_evaluation': {'dataset_sha256': 'dataset', 'weights_sha256': 'parent',
            'image_size': 640, 'preprocessing': 'letterbox', 'metrics': {'map50': .45}}}
    measured = comparison(parent, candidate)
    assert measured['delta'] == pytest.approx(.05) and measured['parent_re_evaluated']
    candidate['parent_evaluation']['image_size'] = 320
    with pytest.raises(ValueError, match='contract mismatch'):
        comparison(parent, candidate)


def test_class_balance_uses_training_labels_and_preserves_mean_loss_weight():
    import numpy as np
    from training.yolo_pipeline import ConditionedTrainer
    trainer = object.__new__(ConditionedTrainer)
    trainer.args = SimpleNamespace(cls_pw=.35)
    trainer.data = {'nc': 2}
    trainer.device = torch.device('cpu')
    trainer.model = torch.nn.Linear(2, 2)
    trainer.train_loader = SimpleNamespace(dataset=SimpleNamespace(labels=[
        {'cls': np.array([[0]] * 38 + [[1]], dtype=np.float32)}]))
    trainer.set_class_weights()
    weights = trainer.model.class_weights
    assert weights.mean().item() == pytest.approx(1.)
    assert weights[1] / weights[0] == pytest.approx(38 ** .35)
    assert weights.max() < 2
