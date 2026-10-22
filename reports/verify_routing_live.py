"""Verify new demo evidence without modifying pre-existing cameras/incidents."""

import asyncio
import io
import json
import time
import zipfile
from pathlib import Path

import httpx
from websockets.asyncio.client import connect


async def live_system(cookie):
    async with connect(
        "ws://127.0.0.1:8000/live/system",
        origin="http://127.0.0.1:8000",
        additional_headers={"Cookie": "surveilx_session=" + cookie},
    ) as socket:
        payload = json.loads(await asyncio.wait_for(socket.recv(), 15))
        return payload["channel"] == "system"


def main():
    report = {"scope": "Generated-source application plumbing only, not model accuracy or real-camera acceptance"}
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=45) as client:
        client.post("/api/auth/login", json={"username": "admin", "password": Path(
            "data/initial-admin-password.txt").read_text(encoding="utf-8").strip()}).raise_for_status()
        camera = client.post("/api/cameras", json={
            "name": "G7 integration verification (generated)", "source": "demo://5",
            "environment": "custom", "zones": [[0, 0, 1, 1]], "enabled": True,
        })
        camera.raise_for_status()
        camera_id = camera.json()["id"]
        report["camera_id"] = camera_id
        try:
            deadline = time.monotonic() + 85
            first = None
            event = None
            while time.monotonic() < deadline:
                health = client.get(f"/api/cameras/{camera_id}/health").json()
                if health.get("frames", 0) and first is None:
                    first = health
                events = client.get("/api/incidents").json()
                event = next((e for e in events if e["camera_id"] == camera_id and e.get("evidence_key")), None)
                if event and first and health["frames"] > first["frames"]:
                    break
                time.sleep(1)
            assert first and health["frames"] > first["frames"], "New capture must advance"
            assert event, "New camera must create evidence"
            image = client.get(f"/api/cameras/{camera_id}/stream")
            image.raise_for_status()
            assert image.content.startswith(b"\xff\xd8")
            evidence = client.get(f"/api/incidents/{event['id']}/evidence")
            evidence.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(evidence.content)) as bundle:
                assert any(name.endswith(".jpg") for name in bundle.namelist())
            frames = client.get(f"/api/incidents/{event['id']}/frames").json()["frames"]
            assert frames
            review = client.get(frames[0]["review_url"])
            review.raise_for_status()
            assert review.content.startswith(b"\xff\xd8")
            Path("reports/routing-live-titled-frame.jpg").write_bytes(review.content)
            labels = event["details"].get("observed_labels", [])
            assert labels == ["synthetic_entity"]
            # Only the newly created fixture incident is acknowledged.
            client.post(f"/api/incidents/{event['id']}/acknowledge").raise_for_status()
            report.update(frames_advance=True, jpeg_stream=True, encrypted_evidence_roundtrip=True,
                titled_review_frame=True, observed_labels=labels, incident_id=event["id"],
                acknowledgement=True, websocket=asyncio.run(live_system(client.cookies.get("surveilx_session"))),
                detector=health.get("model"), frame_count=health["frames"])
        finally:
            disabled = client.patch(f"/api/cameras/{camera_id}", json={"enabled": False})
            disabled.raise_for_status()
            report["test_camera_disabled"] = True
            Path("reports/routing-live-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
