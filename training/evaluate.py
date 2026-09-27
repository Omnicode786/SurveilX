import argparse
import json
from pathlib import Path

from surveilx.controller import Candidate, Scheduler
from training.pipeline import train


def scheduling_simulation():
    """Deterministic offered-load experiment. Costs are simulated, not device measurements."""
    outcomes = {}
    for policy in ["round_robin", "fifo", "static_priority", "greedy_risk", "mcrs"]:
        last = [-5.0] * 4
        service = [0] * 4
        violations = 0
        max_age = 0
        for tick in range(120):
            candidates = [Candidate(str(i), 5 if i == 0 else 1, tick - last[i], 60) for i in range(4)]
            if policy == "mcrs":
                selected = [int(x) for x in Scheduler().allocate(candidates, 120, 5)["selected"]]
            elif policy == "round_robin":
                selected = [(tick * 2) % 4, (tick * 2 + 1) % 4]
            elif policy == "fifo":
                selected = sorted(range(4), key=lambda i: last[i])[:2]
            else:
                selected = [0, 1]
            for i in range(4):
                age = tick - last[i]
                max_age = max(max_age, age)
                violations += age > 5
                if i in selected:
                    last[i], service[i] = tick, service[i] + 1
        outcomes[policy] = {
            "service_count": service,
            "deadline_violations": violations,
            "max_service_age": max_age,
            "jain_fairness": sum(service) ** 2 / (4 * sum(x * x for x in service)),
        }
    return {
        "scope": "simulated constant costs, four always-ready streams, not real incident accuracy",
        "epochs": 120,
        "budget_ms": 120,
        "action_cost_ms": 60,
        "results": outcomes,
    }


def suite(manifest, output, epochs=25):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in ["full", "no_motion", "no_context", "no_interactions", "no_temporal"]:
        result = train(manifest, output / variant, epochs=epochs, ablation=variant)
        results[variant] = {
            "metrics": result["metrics"],
            "duration_seconds": result["duration_seconds"],
            "synthetic": result["synthetic"],
        }
    report = {
        "scope": "Exploratory architecture ablations; same held-out set used across variants; not a final unbiased selection test",
        "ablations": results,
        "scheduling": scheduling_simulation(),
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(suite(args.manifest, args.output, args.epochs), indent=2))
