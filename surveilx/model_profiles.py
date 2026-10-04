"""Validated accuracy/latency profile selection for calibrated detector artifacts."""

import math


TIER_DROPS = {"economy": 0.03, "balanced": 0.01, "performance": 0.0}
GOVERNOR_TIERS = {320: "economy", 480: "balanced", 640: "performance"}


def select_tier_profiles(trials, drops=None):
    """Choose the fastest calibrated profile within each validation-score tolerance."""
    drops = TIER_DROPS if drops is None else drops
    usable = [
        dict(row)
        for row in trials
        if row.get("calibration", {}).get("status") == "fitted"
        and math.isfinite(row.get("validation_map50", float("nan")))
        and math.isfinite(row.get("latency_ms", float("nan")))
        and row["latency_ms"] > 0
    ]
    if not usable:
        raise ValueError("No calibrated finite validation/latency trials are available")
    best = max(row["validation_map50"] for row in usable)
    selected = {}
    for tier in ("economy", "balanced", "performance"):
        floor = best - drops[tier]
        eligible = [row for row in usable if row["validation_map50"] + 1e-12 >= floor]
        selected[tier] = min(
            eligible,
            key=lambda row: (row["latency_ms"], -row["validation_map50"], row["image_size"]),
        )
    return selected


def runtime_profile(manifest, requested_resolution):
    """Return a profile only when the artifact contains calibration for that tier."""
    profiles = manifest.get("inference_profiles", [])
    if not profiles:
        return None
    requested = min(GOVERNOR_TIERS, key=lambda value: abs(value - requested_resolution))
    tier = GOVERNOR_TIERS[requested]
    profile = next((row for row in profiles if row.get("tier") == tier), None)
    if not profile or profile.get("calibration", {}).get("status") != "fitted":
        return None
    return profile
