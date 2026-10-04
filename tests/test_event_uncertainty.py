import numpy as np
import pytest

from training.event_uncertainty import event_uncertainty, wilson


def test_perfect_ten_clips_have_nonperfect_population_bound():
    labels = np.arange(10) % 2
    result = event_uncertainty(np.eye(2)[labels], labels, [str(i) for i in range(10)])
    assert result["accuracy_cluster_bootstrap"]["degenerate"]
    assert result["group_all_correct"]["interval"]["lower"] == pytest.approx(0.7224672)
    assert result["group_all_correct"]["interval"]["upper"] == pytest.approx(1)


def test_paired_views_count_as_one_group_and_missing_class_has_no_interval():
    labels = np.zeros(20, dtype=int)
    result = event_uncertainty(np.tile([1, 0], (20, 1)), labels, [str(i // 2) for i in range(20)])
    assert result["recorded_groups"] == 10
    assert result["group_all_correct"]["trials"] == 10
    assert result["per_class_recall"][1]["interval"] is None
    result = event_uncertainty(np.tile([1, 0], (20, 1)), labels, ["one"] * 20)
    assert result["status"] == "insufficient_clusters"


def test_event_uncertainty_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        event_uncertainty([[float("nan"), 1]], [0], ["one"])
    with pytest.raises(ValueError):
        wilson(11, 10)
