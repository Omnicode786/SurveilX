import numpy as np

from training.uncertainty import cluster_bootstrap_detection
from scripts.report_detection_uncertainty import markdown


def perfect_rows(count):
    predictions, targets = [], []
    for _ in range(count):
        box = np.array([[0.1, 0.1, 0.9, 0.9]])
        predictions.append({"boxes": box, "scores": np.array([0.9]), "labels": np.array([0])})
        targets.append({"boxes": box, "labels": np.array([0])})
    return predictions, targets


def test_cluster_bootstrap_preserves_perfect_ap():
    predictions, targets = perfect_rows(10)
    result = cluster_bootstrap_detection(
        predictions, targets, [f"group-{i}" for i in range(10)], 1, 0.5, iterations=50
    )
    assert result["status"] == "estimated"
    assert result["map50"]["lower"] == 1
    assert result["map50"]["upper"] == 1
    assert result["selection_criterion"] is False


def test_cluster_bootstrap_refuses_too_few_groups():
    predictions, targets = perfect_rows(4)
    result = cluster_bootstrap_detection(predictions, targets, ["a", "a", "b", "b"], 1, 0.5)
    assert result == {
        "status": "insufficient_clusters",
        "clusters": 2,
        "minimum_clusters": 8,
        "iterations": 0,
    }


def test_uncertainty_markdown_names_limits_and_classes():
    text = markdown(
        {
            "version": "candidate",
            "dataset": "dataset",
            "point_map50": 0.6,
            "status": "estimated",
            "clusters": 10,
            "iterations": 50,
            "map50": {"lower": 0.5, "upper": 0.7},
            "classes": ["fire"],
            "point_per_class": [{"class_id": 0, "ap50": 0.61}],
            "per_class": [{"class_id": 0, "lower": 0.51, "upper": 0.71}],
            "warning": "Source independence is unavailable",
        }
    )
    assert "fire" in text and "0.5100–0.7100" in text
    assert "not used for model selection" in text


def test_rare_and_absent_classes_do_not_crash_or_change_macro_taxonomy():
    predictions, targets = perfect_rows(10)
    predictions[0]["labels"] = np.array([1])
    targets[0]["labels"] = np.array([1])
    result = cluster_bootstrap_detection(predictions, targets, [str(i) for i in range(10)], 3, 0.5,
                                         iterations=100)
    assert result["map50"]["lower"] == 1
    assert 0 < result["map50"]["valid_resamples"] < 100
    assert result["per_class"][1]["lower"] == 1
    assert result["per_class"][2]["lower"] is None
    assert result["per_class"][2]["status"] == "unavailable"
