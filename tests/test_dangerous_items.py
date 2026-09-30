import pytest

from scripts.prepare_dangerous_items import CLASSES, parse_annotation, split_bucket


def annotation(label="Rifle", difficult="0"):
    return f"""<annotation><filename>sample.jpg</filename><size><width>100</width><height>50</height></size>
    <object><name>{label}</name><difficult>{difficult}</difficult><bndbox>
    <xmin>1</xmin><ymin>1</ymin><xmax>100</xmax><ymax>50</ymax>
    </bndbox></object></annotation>""".encode()


def test_firearm_mapping_and_voc_edges():
    filename, width, height, boxes, labels = parse_annotation(annotation())
    assert (filename, width, height) == ("sample.jpg", 100, 50)
    assert boxes == [[0.0, 0.0, 1.0, 1.0]]
    assert labels == [CLASSES.index("firearm")]


def test_unknown_or_difficult_labels_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        parse_annotation(annotation("Sword"))
    with pytest.raises(ValueError, match="Difficult"):
        parse_annotation(annotation(difficult="1"))


def test_split_bucket_is_complete_and_four_way():
    assigned = split_bucket(list(range(40)))
    assert [sum(split == name for _, split in assigned) for name, _ in (
        ("train", 0), ("validation", 0), ("calibration", 0), ("test", 0)
    )] == [24, 6, 4, 6]
    assert [item for item, _ in assigned] == list(range(40))
