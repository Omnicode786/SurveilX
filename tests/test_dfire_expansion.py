import hashlib
import json
import zipfile

import cv2
import numpy as np
import pytest

from scripts.expand_dfire_training import expand
from scripts.prepare_dfire import filename_bands
from training.datasets import digest


def test_train_expansion_preserves_holdouts_and_skips_duplicate_content(tmp_path):
    names = [f"data/train/images/AoF{index:05}.jpg" for index in range(200)]
    bands = filename_bands(names)
    contents = {}
    for index, name in enumerate(names):
        image = np.random.default_rng(index).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        contents[name] = cv2.imencode(".jpg", image)[1].tobytes()
    duplicate = bands[("AoF", "train")][1]
    contents[duplicate] = contents[bands[("AoF", "test")][0]]
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in contents.items():
            bundle.writestr(name, content)
            bundle.writestr(
                name.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt", "1 0.5 0.5 0.2 0.2"
            )
    sha = digest(archive)
    (tmp_path / "acquisition.json").write_text(json.dumps({"state": "verified_download", "sha256": sha}))
    base = tmp_path / "base"
    (base / "images").mkdir(parents=True)
    samples = []
    for (_, split), members in bands.items():
        content = contents[members[0]]
        checksum = hashlib.sha256(content).hexdigest()
        relative = f"images/{checksum}.jpg"
        (base / relative).write_bytes(content)
        samples.append(
            {
                "image": relative,
                "sha256": checksum,
                "source_member": members[0],
                "split": split,
                "group": f"dfire-AoF-development-band-{split}",
                "boxes": [[0.4, 0.4, 0.6, 0.6]],
                "labels": [1],
            }
        )
    manifest = {
        "schema_version": 1,
        "task": "detection",
        "domain": "fire_safety",
        "license": "Generated test fixture",
        "classes": ["smoke", "fire"],
        "samples": samples,
        "provenance": {"archive_sha256": sha},
    }
    (base / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "expanded"
    result = expand(archive, base, output, per_family=60)
    expanded = json.loads((output / "manifest.json").read_text())
    assert result["heldout_records_unchanged"] is True
    assert [row for row in expanded["samples"] if row["split"] != "train"] == samples[1:]
    for row in samples:
        assert (output / row["image"]).read_bytes() == (base / row["image"]).read_bytes()
    additions = expanded["samples"][len(samples) :]
    assert len(additions) == len(bands[("AoF", "train")]) - 2
    assert all(row["split"] == "train" and row["source_member"] != duplicate for row in additions)
    assert not {row["sha256"] for row in additions} & {row["sha256"] for row in samples}
    with pytest.raises(ValueError, match="Preserve existing"):
        expand(archive, base, output)
