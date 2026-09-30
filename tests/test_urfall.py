from scripts import prepare_urfall


def test_urfall_plan_is_balanced_disjoint_and_transition_centered(monkeypatch):
    fall = []
    adl = []
    for number in range(1, 21):
        sequence = f"fall-{number:02d}"
        fall.extend([[sequence, str(frame), str(0 if 55 <= frame <= 65 else -1)] for frame in range(1, 121)])
        sequence = f"adl-{number:02d}"
        adl.extend([[sequence, str(frame), "-1"] for frame in range(1, 121)])
    monkeypatch.setattr(
        prepare_urfall,
        "fetch_rows",
        lambda name: fall if name == "urfall-cam0-falls.csv" else adl,
    )
    plan = prepare_urfall.source_plan()
    assert len(plan) == 40
    assert len({row[0] for row in plan}) == 40
    for split in prepare_urfall.SPLITS:
        selected = [row for row in plan if row[1] == split]
        assert len(selected) == 10
        assert {row[2] for row in selected} == {0, 1}
    positive = next(row for row in plan if row[0] == "fall-01")
    assert positive[3][0] <= 60 <= positive[3][-1]
    assert len(set(positive[3])) == prepare_urfall.CONTRACT["frames"]
