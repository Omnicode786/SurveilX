import json

import numpy as np
import pytest
import cv2

from training.datasets import digest, generate, validate_manifest


def scene_fixture(tmp_path):
    manifest_path = generate(tmp_path / "clips", count=80)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        task="event",
        representation="scene_clip",
        input_contract={"frames": 8, "entities": 0, "image_size": 64, "clip_seconds": 2},
    )
    for sample in manifest["samples"]:
        path = manifest_path.parent / sample["file"]
        with np.load(path) as data:
            clip, context = data["clip"], data["context"]
        np.savez_compressed(path, clip=clip, context=context)
        sample["sha256"] = digest(path)
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_scene_training_calibration_runtime_and_cadence(tmp_path):
    from training.pipeline import train
    from surveilx.expert import SVAExpert

    path = scene_fixture(tmp_path)
    metadata = train(path, tmp_path / "run", epochs=1)
    assert metadata["representation"] == "scene_clip"
    assert metadata["calibrated"] and not metadata["deployment_eligible"]
    expert = SVAExpert(tmp_path / "run")
    frames = [(np.zeros((64, 64, 3), dtype=np.uint8), []) for _ in range(8)]
    assert expert.infer(frames, True, "synthetic-motion")["decision"] == "abstain"
    result = expert.infer(frames, True, "synthetic-motion", timestamps=np.linspace(0, 2, 8))
    assert result["decision"] == "experimental_evidence"
    assert len(result["probabilities"]) == 2
    assert (
        expert.infer(frames, True, "synthetic-motion", timestamps=np.arange(8) * 5)["decision"] == "abstain"
    )
    ok, encoded = cv2.imencode(".jpg", frames[0][0])
    assert ok
    compressed = [(encoded.tobytes(), []) for _ in range(8)]
    assert (
        expert.infer(compressed, True, "synthetic-motion", timestamps=np.linspace(0, 2, 8))["decision"]
        == "experimental_evidence"
    )


def test_same_source_video_cannot_leak_between_scene_splits(tmp_path):
    path = scene_fixture(tmp_path)
    data = json.loads(path.read_text())
    data["samples"][0]["source_video_sha256"] = "same-source"
    data["samples"][-1]["source_video_sha256"] = "same-source"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Source video leaks"):
        validate_manifest(path)


def test_scene_import_reads_reviewed_video_intervals(tmp_path):
    import cv2
    from training.import_scene import import_scene

    videos = tmp_path / "videos"
    videos.mkdir()
    clips = []
    for index in range(40):
        name = f"source-{index}.avi"
        writer = cv2.VideoWriter(str(videos / name), cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 64))
        assert writer.isOpened()
        for frame in range(12):
            image = np.full((64, 64, 3), index * 5, dtype=np.uint8)
            image[frame : frame + 5, :, 1] = 220
            writer.write(image)
        writer.release()
        clips.append(
            {
                "video": name,
                "start_seconds": 0,
                "end_seconds": 1,
                "label": "normal" if index % 2 else "event",
                "group": f"source-{index}",
                "split": ["train", "validation", "calibration", "test"][index // 10],
            }
        )
    descriptor = tmp_path / "descriptor.json"
    specification = {
        "source_root": "videos",
        "domain": "test-scene",
        "synthetic": True,
        "license": "Locally generated test",
        "classes": ["normal", "event"],
        "input_contract": {"frames": 4, "entities": 0, "image_size": 32, "clip_seconds": 1},
        "clips": clips,
    }
    descriptor.write_text(json.dumps(specification))
    manifest = import_scene(descriptor, tmp_path / "imported")
    _, counts = validate_manifest(manifest)
    assert counts == {split: 10 for split in ("train", "validation", "calibration", "test")}
    specification["clips"][-1]["video"] = clips[0]["video"]
    descriptor.write_text(json.dumps(specification))
    with pytest.raises(ValueError, match="Whole source videos"):
        import_scene(descriptor, tmp_path / "leaking")
