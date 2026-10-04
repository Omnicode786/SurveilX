from collections import Counter, defaultdict
import io

import pytest

from scripts.prepare_airtlab import PREFIX, actions, fetch, git_blob_sha, plan


def test_airtlab_plan_keeps_views_together_and_filters_other_violence():
    normal = {f"{i}.mp4": {"hug"} for i in range(60)}
    violent = {f"{i}.mp4": {"fight"} for i in range(40)}
    violent.update({"other.mp4": {"stab"}, "mixed.mp4": {"fight", "gunshot"}})
    entries = [
        {"path": f"{PREFIX}{folder}/{camera}/{name}"}
        for folder, names in (("non-violent", normal), ("violent", violent))
        for camera in ("cam1", "cam2")
        for name in names
    ]
    selected = plan(entries, normal, violent)
    assert Counter(item["split"] for item in selected) == {
        "train": 100,
        "validation": 20,
        "calibration": 20,
        "test": 20,
    }
    grouped = defaultdict(set)
    for item in selected:
        grouped[item["group"]].add(item["split"])
        assert "other.mp4" not in item["entry"]["path"]
        assert "mixed.mp4" not in item["entry"]["path"]
    assert len(grouped) == 80 and all(len(splits) == 1 for splits in grouped.values())
    assert actions("FILE; ACTION CLASSES\n1.mp4;fight,punch\n") == {"1.mp4": {"fight", "punch"}}


def test_download_is_pinned_verified_and_reuses_only_intact_bytes(tmp_path):
    data = b"source video bytes"
    entry = {"path": "videos/1.mp4", "size": len(data), "sha": git_blob_sha(data)}
    calls = []

    def opener(url, timeout):
        calls.append(url)
        return io.BytesIO(data)

    path = fetch(entry, tmp_path, opener)
    assert fetch(entry, tmp_path, opener) == path
    assert len(calls) == 1 and "/1f7747e104301ccaa82ef5a2f6804b51ced1c398/" in calls[0]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        fetch(entry, tmp_path, opener)
    with pytest.raises(ValueError, match="Unsafe"):
        fetch({**entry, "path": "../escape.mp4"}, tmp_path, opener)
