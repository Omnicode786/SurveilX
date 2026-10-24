from surveilx.runtime import IntervalGate


def test_fast_ticks_do_not_multiply_maintenance_or_hardware_probes():
    gates = IntervalGate()
    counts = {"hardware": 0, "maintenance": 0, "drift": 0, "retention": 0}
    for tick in range(2401):
        now = tick / 4
        for name, seconds in zip(counts, (1, 10, 60, 600)):
            counts[name] += gates.due(name, seconds, now)
    assert counts == {"hardware": 601, "maintenance": 61, "drift": 11, "retention": 2}


def test_delayed_ticks_do_not_queue_catchup_work():
    gates = IntervalGate()
    assert gates.due("maintenance", 10, 0)
    assert gates.due("maintenance", 10, 100)
    assert not gates.due("maintenance", 10, 100.25)
    assert gates.due("maintenance", 10, 110)


def test_slow_drift_is_single_flight_and_does_not_block_inference(monkeypatch):
    import threading
    from surveilx.runtime import Runtime, drift_monitor

    entered, release = threading.Event(), threading.Event()
    def slow_evaluation():
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(drift_monitor, 'evaluate', slow_evaluation)
    runtime = object.__new__(Runtime)
    runtime.drift_thread = None
    runtime.error = None
    try:
        assert runtime.request_drift_check()
        assert entered.wait(2)
        assert not runtime.request_drift_check()
        assert runtime.drift_thread.is_alive()
    finally:
        release.set()
        runtime.drift_thread.join(2)
    assert runtime.error is None


def test_runtime_budget_scale_rejects_unbounded_workload():
    import pytest
    from pydantic import ValidationError
    from surveilx.config import Settings
    assert Settings(_env_file=None, runtime_budget_scale=1.75).runtime_budget_scale == 1.75
    with pytest.raises(ValidationError):
        Settings(_env_file=None, runtime_budget_scale=100)
