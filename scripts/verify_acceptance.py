"""Exercise frozen synthetic acceptance through the running API without production approval."""

import json
import time
from pathlib import Path

import httpx

from surveilx.config import settings
from training.datasets import generate


def main():
    name = "acceptance-generated-20260928"
    directory = settings.data_dir / "datasets" / name
    if not (directory / "manifest.json").exists():
        generate(directory, count=400, seed=2026092801)
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=90) as client:
        password = (
            settings.admin_password or (settings.data_dir / "initial-admin-password.txt").read_text().strip()
        )
        client.post("/api/auth/login", json={"username": "admin", "password": password}).raise_for_status()
        model = next(m for m in client.get("/api/models").json() if m["version"] == "synthetic-candidate-v1")
        response = client.post(f"/api/models/{model['id']}/evaluate", json={"dataset": name})
        response.raise_for_status()
        job_id = response.json()["id"]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            result = next(r for r in client.get("/api/acceptance").json() if r["id"] == job_id)
            state = result["payload"]["state"]
            if state not in {"queued", "running"}:
                break
            time.sleep(2)
        else:
            raise TimeoutError("Acceptance evaluation did not finish within ten minutes")
        assert state == "completed", result
        assert result["payload"]["report"]["deployment_eligible"] is False
        assert result["payload"]["report"]["synthetic"] is True
        # A direct approval request must also reject synthetic reports even if metrics pass.
        rejected = client.post(f"/api/acceptance/{job_id}/approve")
        assert rejected.status_code == 409, rejected.text
        report = {"evaluation": result, "synthetic_approval_http_status": rejected.status_code}
        Path("reports/acceptance-workflow.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "id": job_id,
                    "state": state,
                    "passed": result["payload"]["report"]["passed"],
                    "deployment_eligible": False,
                    "metrics": result["payload"]["report"]["metrics"],
                }
            )
        )


if __name__ == "__main__":
    main()
