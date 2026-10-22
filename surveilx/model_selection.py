"""Task/category recommendations; development evidence never grants deployment approval."""

import json
import csv
import math
import re
from pathlib import Path
from functools import lru_cache
from training.datasets import digest


@lru_cache(maxsize=512)
def _weight_hash(path, size, modified_ns):
    # Runtime loading/acceptance still verify independently; the dashboard avoids
    # repeatedly reading immutable completed checkpoints on every refresh.
    return digest(Path(path))


@lru_cache(maxsize=512)
def _json_cache(path, size, modified_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json(path):
    stat = path.stat()
    return _json_cache(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=256)
def _yolo_curve(path, size, modified_ns):
    from surveilx.generations import yolo_epoch

    with Path(path).open(encoding="utf-8") as stream:
        return [value for row in csv.DictReader(stream) if (value := yolo_epoch(row))]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def contract(manifest):
    return json.dumps(
        [
            manifest.get("task"),
            manifest.get("domain"),
            bool(manifest.get("synthetic")),
            manifest.get("dataset_sha256"),
            manifest.get("classes"),
            manifest.get("representation", "entity_clip"),
            manifest.get("input_contract"),
        ],
        sort_keys=True,
    )


def evaluation_for(data, version, manifest):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", version):
        return None
    path = Path(data) / "model-evaluations" / (version + ".json")
    try:
        report = read_json(path)
        if (
            report.get("split") != "validation"
            or report.get("weights_sha256") != manifest.get("weights_sha256")
            or report.get("dataset_sha256") != manifest.get("dataset_sha256")
            or report.get("classes") != manifest.get("classes")
            or report.get("task") != manifest.get("task")
            or report.get("manifest_sha256") != digest(Path(data) / "runs" / version / "manifest.json")
        ):
            return None
        return report
    except (OSError, ValueError):
        return None


def candidate_evaluation(data, version, manifest):
    """Read-only class evidence; never participates in calibrated/approved routing."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", version):
        return None
    directory = Path(data) / "runs" / version
    report = manifest.get("trained_candidate")
    sidecar = Path(data) / "candidate-evaluations" / (version + ".json")
    if report is None and sidecar.exists():
        report = read_json(sidecar)
        if report.get("manifest_sha256") != digest(directory / "manifest.json"):
            return None
    if not report or report.get("dataset_sha256") != manifest.get("dataset_sha256"):
        return None
    if report.get("weights_sha256") == manifest.get("weights_sha256") and manifest.get("calibrated"):
        return None  # already represented by the completed calibrated artifact
    checkpoint = (directory / report.get("checkpoint", "")).resolve()
    if not checkpoint.is_relative_to(directory.resolve()) or not checkpoint.is_file():
        return None
    stat = checkpoint.stat()
    if _weight_hash(str(checkpoint), stat.st_size, stat.st_mtime_ns) != report.get("weights_sha256"):
        return None
    measured = report.get("validation_metrics", {})
    previous = report.get("parent_validation_metrics", {})
    changes = []
    for row in measured.get("per_class", []):
        index = row.get("class_id")
        if not isinstance(index, int) or not 0 <= index < len(manifest.get("classes", [])):
            continue
        before = next((r.get("ap50") for r in previous.get("per_class", []) if r.get("class_id") == index), None)
        if finite(before) and finite(row.get("ap50")) and row["ap50"] > before:
            changes.append({"class": manifest["classes"][index], "support": row.get("ground_truth"),
                "parent_ap50": before, "candidate_ap50": row["ap50"], "delta": row["ap50"] - before})
    return {"version": version, "checkpoint": report["checkpoint"], "weights_sha256": report["weights_sha256"],
        "domain": manifest.get("domain"), "dataset_sha256": manifest.get("dataset_sha256"),
        "class_improvements": changes, "calibrated": False, "deployment_eligible": False,
        "scope": "Unpromoted validation-only trained candidate; parent compared at the same geometry; not a test gain"}


def rank_categories(models):
    """Rank only within an identical task/domain/dataset/taxonomy/clip contract.

    AP (or event F1), recall and F1 are balanced; missing evidence is explicit.
    Latency only breaks ties when measured in the same runtime/device contract.
    """
    groups = {}
    for model in models:
        manifest, report = model["manifest"], model.get("evaluation")
        calibrated = manifest.get("calibrated") and (
            manifest.get("task") == "event" or (manifest.get("calibration") or {}).get("status") == "fitted"
        )
        if not calibrated or not report or report.get("split") != "validation":
            continue
        for row in report.get("metrics", {}).get("per_class", []):
            index = row.get("class_id")
            if not isinstance(index, int) or not 0 <= index < len(manifest["classes"]):
                continue
            if row.get("ground_truth", row.get("support", 0)) < 1:
                continue
            category = manifest["classes"][index]
            primary = row.get("ap50") if manifest["task"] == "detection" else row.get("f1")
            if not finite(primary):
                continue
            critical = category in {"firearm", "knife", "machete", "fire", "smoke", "fall", "fighting"}
            terms = [
                (primary, 0.40 if critical else 0.55),
                (row.get("recall"), 0.40 if critical else 0.25),
                (row.get("f1"), 0.20),
            ]
            score = sum(v * w for v, w in terms if finite(v)) / sum(w for v, w in terms if finite(v))
            ece = report["metrics"].get("reliability", report["metrics"]).get("ece")
            if finite(ece):
                score -= 0.05 * min(1, max(0, ece))
            latency = (
                report.get("runtime_latency_ms")
                if (
                    report.get("latency_usable_for_routing")
                    and report.get("device") == model.get("runtime_device", report.get("device"))
                )
                else None
            )
            groups.setdefault((contract(manifest), category), []).append(
                {
                    "version": model["version"],
                    "id": model.get("id"),
                    "stage": model.get("stage", "candidate"),
                    "score": score,
                    "metrics": row,
                    "calibration_ece": ece,
                    "latency_ms": latency,
                    "device": report.get("device"),
                    "weights_sha256": manifest.get("weights_sha256"),
                }
            )
    recommendations = []
    for (scope, category), candidates in sorted(groups.items()):
        candidates.sort(key=lambda r: (-r["score"], r["version"]))
        # Do not compare CPU and GPU timings or cold-load measurements as tie breakers.
        tied = [r for r in candidates if abs(r["score"] - candidates[0]["score"]) < 1e-8]
        if tied and len({r["device"] for r in tied}) == 1 and all(finite(r["latency_ms"]) for r in tied):
            winner = min(tied, key=lambda r: (r["latency_ms"], r["version"]))
            candidates.remove(winner)
            candidates.insert(0, winner)
        recommendations.append(
            {
                "task": json.loads(scope)[0],
                "domain": json.loads(scope)[1],
                "dataset_sha256": json.loads(scope)[3],
                "contract": scope,
                "category": category,
                "selected": candidates[0],
                "fallbacks": candidates[1:],
                "reason": "Compatible validation-only class metrics: AP50 (event:F1), recall and F1, with a small ECE penalty; critical hazards prioritize recall (40/40/20 versus55/25/20). Matched-device warm latency only breaks ties. Approval is separate.",
                "production_approved": candidates[0]["stage"] == "production",
            }
        )
    return recommendations


def catalog(data, registered=(), active=()):
    data = Path(data)
    registry = {r["version"]: r for r in registered}
    licenses = {}
    for dataset in (data / "datasets").glob("*/manifest.json"):
        try:
            stat = dataset.stat()
            licenses[_weight_hash(str(dataset.resolve()), stat.st_size, stat.st_mtime_ns)] = read_json(
                dataset
            ).get("license")
        except (OSError, ValueError):
            continue
    result = []
    candidates = []
    errors = []
    for path in sorted((data / "runs").glob("*/manifest.json")):
        try:
            manifest = read_json(path)
            version = path.parent.name
            weights = path.parent / "weights.pt"
            stat = weights.stat()
            if _weight_hash(str(weights.resolve()), stat.st_size, stat.st_mtime_ns) != manifest.get(
                "weights_sha256"
            ):
                raise ValueError("Weight checksum mismatch")
            registered_model = registry.get(version, {})
            curve = []
            history_path = path.parent / "history.json"
            if not history_path.exists() and (path.parent / "native_history.json").exists():
                history_path = path.parent / "native_history.json"
            if history_path.exists():
                history = read_json(history_path)
                curve = [
                    {
                        k: row.get(k)
                        for k in (
                            "epoch",
                            "train_loss",
                            "validation_loss",
                            "validation_map50",
                            "training",
                            "validation",
                            "validation_metrics",
                            "learning_rate",
                            "elapsed_seconds",
                            "optimizer_step_attempts",
                            "learning_rate_groups",
                            "nominal_effective_batch_size",
                            "physical_batch_size",
                        )
                    }
                    for row in history
                    if isinstance(row, dict)
                ]
            csv_path = path.parent / "fit" / "results.csv"
            if not curve and csv_path.exists():
                stat = csv_path.stat()
                curve = _yolo_curve(str(csv_path.resolve()), stat.st_size, stat.st_mtime_ns)
            result.append(
                {
                    "version": version,
                    "id": registered_model.get("id"),
                    "stage": registered_model.get("stage", "candidate"),
                    "active": registered_model.get("id") in active if registered_model else False,
                    "manifest": {
                        k: manifest.get(k)
                        for k in (
                            "task",
                            "domain",
                            "synthetic",
                            "classes",
                            "representation",
                            "input_contract",
                            "dataset_sha256",
                            "weights_sha256",
                            "architecture",
                            "model_config",
                            "calibrated",
                            "calibration",
                            "temperature",
                            "metrics",
                            "acceptance",
                            "deployment_eligible",
                            "gate_reason",
                        )
                    },
                    "evaluation": evaluation_for(data, version, manifest),
                    "learning_curve": curve,
                    "dataset_license": licenses.get(manifest.get("dataset_sha256")),
                }
            )
            candidate = candidate_evaluation(data, version, manifest)
            if candidate is not None:
                candidates.append(candidate)
        except (OSError, ValueError) as exc:
            errors.append({"version": path.parent.name, "error": str(exc)})
    partial = sorted(
        {
            p.parent.name
            for p in (data / "runs").glob("*/config.json")
            if not (p.parent / "manifest.json").exists()
        }
        | {
            p.parent.parent.name
            for p in (data / "runs").glob("*/fit/args.yaml")
            if not (p.parent.parent / "manifest.json").exists()
        }
    )
    return {
        "models": result,
        "unpromoted_candidates": candidates,
        "recommendations": rank_categories(result),
        "partial_runs": partial,
        "errors": errors,
        "selection_split": "validation",
        "automatic_activation": False,
        "unavailable_measurements": [
            "Independent false alarms per camera-hour and critical event delay require timestamped reviewed streams.",
            "HOTA requires labeled identity trajectories; no real identity corpus is available.",
            "Challenging-condition robustness requires reviewed condition labels; not inferred from brightness alone.",
        ],
    }
