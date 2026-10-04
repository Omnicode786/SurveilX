"""Report installed development evidence and interrupted runs without changing models."""

import json
from pathlib import Path

from surveilx.config import settings
from surveilx.task_profiles import POLICY_PRIMITIVES, PROFILES, declared_profiles, development_evidence
from training.datasets import digest


REPLACEMENTS = {
    "accuracy-g3-weapons-scratch": "accuracy-g3b-weapons-scratch",
    "accuracy-ppe-g3-scratch": "accuracy-ppe-g3b-scratch",
}


def audit():
    datasets = {}
    for path in sorted((settings.data_dir / "datasets").glob("*/manifest.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        datasets[digest(path)] = {"name": path.parent.name, "metadata": metadata}
    artifacts, errors = [], []
    for path in sorted((settings.data_dir / "runs").glob("*/manifest.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            weights = path.parent / "weights.pt"
            intact = weights.is_file() and digest(weights) == metadata.get("weights_sha256")
            source = datasets.get(metadata.get("dataset_sha256"))
            artifacts.append(
                {
                    "version": path.parent.name,
                    "task": metadata.get("task"),
                    "domain": metadata.get("domain"),
                    "capabilities": declared_profiles(metadata),
                    "weights_intact": intact,
                    "dataset_resolved": source is not None,
                    "dataset": source["name"] if source else None,
                    "source_provenance": source["metadata"].get("provenance", {}) if source else {},
                    "calibrated": metadata.get("calibrated", False),
                    "inherited": metadata.get("training_status") == "external_pretrained",
                    "development": development_evidence(metadata),
                }
            )
        except (ValueError, OSError, KeyError) as exc:
            errors.append({"version": path.parent.name, "error": str(exc)})
    by_version = {row["version"]: row for row in artifacts}
    interrupted = []
    for path in sorted((settings.data_dir / "runs").iterdir()):
        if not path.is_dir() or (path / "manifest.json").exists() or not (path / "config.json").exists():
            continue
        replacement = REPLACEMENTS.get(path.name)
        record = by_version.get(replacement, {})
        interrupted.append(
            {
                "version": path.name,
                "replacement": replacement,
                "replacement_complete_and_intact": bool(
                    record.get("weights_intact") and record.get("dataset_resolved")
                ),
                "original_preserved": True,
            }
        )
    families = []
    for profile in PROFILES:
        matching = [row for row in artifacts if profile["id"] in row["capabilities"]]
        compatible_data = [
            row["name"] for row in datasets.values() if profile["id"] in declared_profiles(row["metadata"])
        ]
        families.append(
            {
                "id": profile["id"],
                "title": profile["title"],
                "task": profile["task"],
                "datasets": compatible_data,
                "artifact_versions": [r["version"] for r in matching],
                "evaluated_candidates": sum(r["development"]["score"] is not None for r in matching),
                "all_class_point_target_candidates": [
                    r["version"] for r in matching if r["development"]["all_classes_target_met"]
                ],
                "domain_reliability": "Not established by development metrics; requires reviewed independent domain evidence",
            }
        )
    report = {
        "scope": "Artifact integrity and development coverage audit; no training, activation or production acceptance",
        "families": families,
        "artifacts": artifacts,
        "interrupted_runs": interrupted,
        "errors": errors,
        "excluded": [
            "research paper",
            "complete acceleration deployment",
            "full hardware acceleration coverage",
            "complete production verification",
            "full production infrastructure/deployment completion",
        ],
    }
    output = Path("reports/product-readiness.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Product development evidence",
        "",
        report["scope"] + ".",
        "",
        "A successful training run or aggregate score is not proof of reliable multi-domain detection.",
        "",
        "| Family | Imported datasets | Evaluated candidates | Every model class >=0.50 | Implemented policy rules |",
        "|---|---:|---:|---:|---|",
    ]
    for row in families:
        rules = ", ".join(POLICY_PRIMITIVES.get(row["id"], [])) or "N/A"
        counts = "N/A | N/A | N/A" if row["task"] == "policy" else (
            f"{len(row['datasets'])} | {row['evaluated_candidates']} | {len(row['all_class_point_target_candidates'])}"
        )
        lines.append(f"| {row['title']} | {counts} | {rules} |")
    lines.extend(["", "## Interrupted-run disposition", ""])
    for row in interrupted:
        lines.append(
            f"- `{row['version']}`: preserved; replacement `{row['replacement']}` complete and intact: {row['replacement_complete_and_intact']}."
        )
    lines.extend(
        [
            "",
            "Detailed scores, source limits, hashes and errors are in product-readiness.json. Inherited models without matching local data are not local training evidence.",
            "",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return output, report


if __name__ == "__main__":
    output, report = audit()
    print(
        json.dumps(
            {
                "report": str(output),
                "artifacts": len(report["artifacts"]),
                "integrity_failures": [
                    row["version"] for row in report["artifacts"] if not row["weights_intact"]
                ],
                "interrupted": report["interrupted_runs"],
                "errors": report["errors"],
            },
            indent=2,
        )
    )
