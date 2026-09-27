import asyncio
import json
import time
from pathlib import Path

import httpx
from websockets.asyncio.client import connect


async def websocket(cookie):
    async with connect(
        "ws://127.0.0.1:8000/live/system",
        origin="http://127.0.0.1:8000",
        additional_headers={"Cookie": f"surveilx_session={cookie}"},
    ) as socket:
        payload = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
        assert payload["channel"] == "system"
        assert len(payload["data"]["cameras"]) >= 4
        return True


def main():
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
        password = Path("data/initial-admin-password.txt").read_text().strip()
        response = client.post("/api/auth/login", json={"username": "admin", "password": password})
        response.raise_for_status()
        first = client.get("/api/cameras").json()
        assert len(first) >= 4 and all(c["status"] == "online" for c in first)
        images = [client.get(f"/api/cameras/{c['id']}/stream") for c in first]
        assert all(image.status_code == 200 and image.content[:2] == b"\xff\xd8" for image in images)
        time.sleep(3)
        second = client.get("/api/cameras").json()
        assert all(new["frames"] > old["frames"] for old, new in zip(first, second))
        events = client.get("/api/incidents").json()
        event = next(e for e in events if e["state"] == "VERIFYING")
        evidence = client.get(f"/api/incidents/{event['id']}/evidence")
        assert evidence.status_code == 200 and evidence.content.startswith(b"PK")
        assert client.post(f"/api/incidents/{event['id']}/acknowledge").status_code == 200
        feedback = client.post(
            "/api/feedback",
            json={
                "incident_id": event["id"],
                "label": "true_event",
                "notes": "Automated integration verification on synthetic source",
            },
        )
        assert feedback.status_code == 200
        ws_ok = asyncio.run(websocket(client.cookies.get("surveilx_session")))
        controller = client.get("/api/controller/status").json()
        report = {
            "timestamp": time.time(),
            "four_generated_streams": True,
            "frames_advance": True,
            "jpeg_streams": True,
            "evidence_roundtrip": True,
            "acknowledgement": True,
            "feedback": True,
            "websocket": ws_ok,
            "incidents_persisted": len(events),
            "runtime": controller,
            "scope": "Local generated sources and SQLite; no real RTSP/network camera or PostgreSQL claim",
        }
        Path("reports").mkdir(exist_ok=True)
        Path("reports/live-verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
