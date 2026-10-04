import cv2
import numpy as np
import pytest

from scripts.prepare_scene_event_directory import prepare
from training.datasets import SPLITS, validate_manifest


def write_video(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 40))
    try:
        for index in range(12):
            frame = np.full((40, 40, 3), value + index, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_prepare_scene_event_directory_imports_split_class_videos(tmp_path):
    source = tmp_path / "source"
    for split_index, split in enumerate(SPLITS):
        for video_index in range(5):
            write_video(
                source / split / "normal" / f"{split}-normal-{video_index}.avi",
                10 + split_index * 20 + video_index,
            )
            write_video(
                source / split / "collision" / f"{split}-collision-{video_index}.avi",
                120 + split_index * 20 + video_index,
            )

    manifest_path = prepare(
        source,
        tmp_path / "datasets" / "collision-development-v1",
        ["normal", "collision"],
        "traffic-collision-development",
        "user-provided development data; redistribution rights not assumed",
        ["traffic_collision"],
        frames=4,
        image_size=32,
        clip_seconds=0.5,
        max_clips_per_video=1,
        max_clips_per_class_split=5,
    )

    manifest, counts = validate_manifest(manifest_path)
    assert manifest["classes"] == ["normal", "collision"]
    assert manifest["capability_ids"] == ["traffic_collision"]
    assert counts == {split: 10 for split in SPLITS}
    assert (tmp_path / "datasets" / "collision-development-v1-descriptor.json").exists()
    assert manifest["provenance"]["label_scope"] == "inherited_video_label"
    with pytest.raises(ValueError, match="preparation differs"):
        prepare(source, manifest_path.parent, ["normal", "collision"], "changed-domain", "rights")


def test_multiple_windows_end_on_decodable_frame(tmp_path):
    from scripts.prepare_scene_event_directory import clip_starts, video_duration
    path = tmp_path / "clip.avi"
    write_video(path, 20)
    assert video_duration(path) == pytest.approx(1.1)
    starts = clip_starts(video_duration(path), 0.5, 3)
    assert len(starts) == 3
    assert starts[-1] + 0.5 <= 1.1


def test_sequential_interval_sampling_matches_exact_frame_seeks(tmp_path):
    from training.import_scene import sampled_frames
    path = tmp_path / "clip.avi"
    write_video(path, 20)
    indices = np.array([1, 3, 7, 11])
    video = cv2.VideoCapture(str(path))
    try:
        actual = list(sampled_frames(video, indices))
        for index, observed in zip(indices, actual, strict=True):
            video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, expected = video.read()
            assert ok and np.array_equal(observed, expected)
    finally:
        video.release()
