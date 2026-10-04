from scripts.diagnose_scratch import diagnostic_samples


def test_diagnostic_covers_classes_without_duplicate_images_or_first_class_bias():
    samples = [{"image": f"{label}-{index}", "labels": [label]}
               for label in range(4) for index in range(20)]
    samples.append({"image": "multi", "labels": [0, 1, 2, 3]})
    selected = diagnostic_samples(samples, 4, 8)
    assert len(selected) == len({row["image"] for row in selected}) == 8
    assert {label for row in selected for label in row["labels"]} == {0, 1, 2, 3}
    assert selected == diagnostic_samples(samples, 4, 8)
