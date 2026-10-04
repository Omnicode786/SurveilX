import pytest

from surveilx.model_profiles import runtime_profile, select_tier_profiles


def trial(size, score, latency, status="fitted"):
    return {
        "image_size": size,
        "validation_map50": score,
        "latency_ms": latency,
        "calibration": {"status": status, "threshold": 0.4},
    }


def test_tiers_trade_only_bounded_validation_accuracy_for_latency():
    selected = select_tier_profiles(
        [trial(192, 0.56, 10), trial(256, 0.58, 16), trial(320, 0.59, 29)]
    )
    assert selected["economy"]["image_size"] == 192
    assert selected["balanced"]["image_size"] == 256
    assert selected["performance"]["image_size"] == 320


def test_profile_selection_rejects_uncalibrated_or_nonfinite_trials():
    with pytest.raises(ValueError, match="No calibrated"):
        select_tier_profiles([trial(256, 0.9, 10, "insufficient")])
    selected = select_tier_profiles([trial(192, 0.5, 8), trial(256, float("nan"), 3)])
    assert selected["performance"]["image_size"] == 192


def test_runtime_uses_only_named_fitted_profiles():
    manifest = {"inference_profiles": [{"tier": "economy", **trial(192, 0.5, 8)}]}
    assert runtime_profile(manifest, 320)["image_size"] == 192
    assert runtime_profile(manifest, 640) is None
    assert runtime_profile({}, 320) is None
