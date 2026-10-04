import pytest
import torch
from torch.nn import functional as F

from training.detector_model import classification_loss


def test_focal_classification_legacy_and_hard_example_gradients():
    logits = torch.tensor([0.0, 5.0], requires_grad=True)
    targets = torch.ones(2)
    assert torch.equal(
        classification_loss(logits, targets), F.binary_cross_entropy_with_logits(logits, targets)
    )
    loss = classification_loss(logits, targets, gamma=2, alpha=0.25)
    expected = (
        0.25
        * (1 - logits.sigmoid()).square()
        * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    ).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert abs(logits.grad[0]) > 100 * abs(logits.grad[1])


@pytest.mark.parametrize("gamma,alpha", [(-1, 0.25), (float("nan"), 0.25), (2, 0), (2, 1)])
def test_focal_classification_rejects_invalid_recipe(gamma, alpha):
    with pytest.raises(ValueError):
        classification_loss(torch.zeros(1), torch.zeros(1), gamma, alpha)
