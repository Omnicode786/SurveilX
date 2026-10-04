from scripts.prepare_sh17_balanced import select_additions, split_for


def test_selection_prioritizes_missing_labels_and_preserves_splits():
    existing = [{"source_member": "old", "split": "train", "labels": [0]}]
    records = [
        {"source_member": "old", "split": "train", "labels": [0]},
        {"source_member": "common", "split": "train", "labels": [0]},
        {"source_member": "rare", "split": "train", "labels": [4]},
        {"source_member": "test-rare", "split": "test", "labels": [4]},
    ]
    chosen = select_additions(records, existing, {"train": 1, "test": 1}, 3)
    assert {row["source_member"] for row in chosen} == {"rare", "test-rare"}
    assert next(row for row in chosen if row["source_member"] == "test-rare")["split"] == "test"
    assert existing == [{"source_member": "old", "split": "train", "labels": [0]}]


def test_group_assignment_is_stable_and_image_independent():
    group, split = split_for(123456)
    assert group == "sh17-photographer-123456"
    assert split in {"train", "validation", "calibration", "test"}
    assert split_for(123456) == (group, split)
