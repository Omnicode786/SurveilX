import argparse
import json
import re
from pathlib import Path

from sqlalchemy import select

from surveilx.config import settings
from surveilx.database import Base, ModelVersion, Record, audit, engine, transaction
from training.datasets import digest


def register(version):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", version):
        raise ValueError("Run version must be a single directory name")
    directory = settings.data_dir / "runs" / version
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(directory / "weights.pt") != manifest["weights_sha256"]:
        raise ValueError("Checkpoint integrity check failed")
    Base.metadata.create_all(engine)
    with transaction() as session:
        existing = session.scalar(select(ModelVersion).where(ModelVersion.version == version))
        if existing:
            return existing.id
        model = ModelVersion(name=manifest["name"], version=version, manifest=manifest)
        session.add(model)
        session.flush()
        session.add(Record(kind="experiment", payload={"state":"completed", "version":version,"result":manifest}))
        audit(session, "local-cli", "candidate_registered", model.id)
        return model.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    print(register(parser.parse_args().version))
