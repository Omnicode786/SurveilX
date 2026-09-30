import json

import cv2
import numpy as np
import pytest

from training.import_detection import convert


@pytest.mark.parametrize("format_name", ["coco", "yolo", "voc"])
def test_formats_map_same_geometry_and_preserve_groups(tmp_path, format_name):
    sources = []
    for index, split in enumerate(("train", "validation", "calibration", "test")):
        root = tmp_path / split
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        cv2.imwrite(str(root / "images" / "frame.png"), np.full((20, 40, 3), index * 30, np.uint8))
        annotations = f"{split}/labels"
        if format_name == "coco":
            annotations = f"{split}/annotations.json"
            (tmp_path / annotations).write_text(
                json.dumps(
                    {
                        "images": [{"id": 17, "file_name": "frame.png", "width": 40, "height": 20}],
                        "categories": [{"id": 83, "name": "person"}],
                        "annotations": [{"image_id": 17, "category_id": 83, "bbox": [4, 2, 20, 12]}],
                    }
                )
            )
        elif format_name == "yolo":
            (root / "labels" / "frame.txt").write_text("0 .35 .4 .5 .6")
        else:
            (root / "labels" / "frame.xml").write_text(
                "<annotation><filename>frame.png</filename><size><width>40</width><height>20</height></size>"
                "<object><name>person</name><bndbox><xmin>5</xmin><ymin>3</ymin><xmax>24</xmax>"
                "<ymax>14</ymax></bndbox></object></annotation>"
            )
        sources.append(
            {
                "split": split,
                "group": f"camera-{index}",
                "images": f"{split}/images",
                "annotations": annotations,
            }
        )
    descriptor = tmp_path / "import.json"
    descriptor.write_text(
        json.dumps(
            {
                "format": format_name,
                "classes": ["person"],
                "domain": "test",
                "license": "test fixture",
                "sources": sources,
            }
        )
    )
    output = tmp_path / "converted"
    result = convert(descriptor, output)
    assert sum(result["counts"].values()) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    for sample in manifest["samples"]:
        assert sample["boxes"][0] == pytest.approx([0.1, 0.1, 0.6, 0.7])
        assert sample["labels"] == [0]
    assert len({s["group"] for s in manifest["samples"]}) == 4
    with pytest.raises(ValueError, match="new dataset"):
        convert(descriptor, output)


def test_event_model_supports_more_than_two_entities_and_classes():
    import torch
    from training.models import SVANet

    torch.set_num_threads(2)
    model = SVANet(classes=5)
    output = model(torch.rand(2, 6, 3, 48, 48), torch.rand(2, 6, 4, 4), torch.rand(2, 4))
    assert output["event"].shape == (2, 5)
    assert output["states"].shape == (2, 4, 5)
    torch.nn.functional.cross_entropy(output["event"], torch.tensor([3, 4])).backward()
    assert model.event_head[-1].weight.grad.abs().sum() > 0
