import pytest

from scripts.prepare_dfire import filename_bands, parse_labels


def test_labels_preserve_negatives_and_reject_invalid_geometry():
    assert parse_labels("") == ([], [])
    boxes, labels = parse_labels("1 0.5 0.5 0.4 0.2")
    assert labels == [1]
    assert boxes[0] == pytest.approx([0.3, 0.4, 0.7, 0.6])
    for invalid in ("2 0.5 0.5 0.2 0.2", "1 nan 0.5 0.2 0.2", "0 0 0 1 1", "0 0.5 0.5 0 0"):
        with pytest.raises(ValueError):
            parse_labels(invalid)


def test_filename_groups_have_guard_bands_and_never_reuse_images():
    names = [f"data/train/images/PublicDataset{i:05}.jpg" for i in range(200)]
    bands = filename_bands(names)
    selected = [name for members in bands.values() for name in members]
    assert len(selected) == len(set(selected)) == 104
    assert len(bands) == 4
    for previous, following in zip(list(bands.values())[:-1], list(bands.values())[1:]):
        assert names.index(following[0]) - names.index(previous[-1]) == 33


def test_development_groups_are_not_accepted_as_independent(monkeypatch):
    from training import acceptance

    monkeypatch.setattr(
        acceptance,
        "validate_manifest",
        lambda path: ({"provenance": {"source_independence_verified": False}}, {}),
    )
    with pytest.raises(ValueError, match="Development-only"):
        acceptance.verify_independence("unused", [])
