import time
import uuid
from contextlib import contextmanager

from sqlalchemy import JSON, Boolean, Float, ForeignKey, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from surveilx.config import settings


def uid():
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    username: Mapped[str] = mapped_column(String, unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String, default="viewer")


class Session(Base):
    __tablename__ = "sessions"
    token_hash: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    expires: Mapped[float] = mapped_column(Float)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String)
    source_cipher: Mapped[str] = mapped_column(Text)
    environment: Mapped[str] = mapped_column(String, default="custom")
    priority: Mapped[int] = mapped_column(default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    profile_version: Mapped[int] = mapped_column(default=1)
    zones: Mapped[list] = mapped_column(JSON, default=list)
    masks: Mapped[list] = mapped_column(JSON, default=list)


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"))
    created: Mapped[float] = mapped_column(Float, default=time.time)
    updated: Mapped[float] = mapped_column(Float, default=time.time)
    event_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String, default="review")
    state: Mapped[str] = mapped_column(String, default="VERIFYING")
    evidence_key: Mapped[str | None] = mapped_column(String, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"), unique=True)
    created: Mapped[float] = mapped_column(Float, default=time.time)
    status: Mapped[str] = mapped_column(String, default="delivered_in_app")
    acknowledged_by: Mapped[str | None] = mapped_column(String, nullable=True)


class Audit(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    timestamp: Mapped[float] = mapped_column(Float, default=time.time, index=True)
    actor: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    target: Mapped[str] = mapped_column(String, default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class Feedback(Base):
    __tablename__ = "feedback_labels"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"))
    actor: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)
    notes: Mapped[str] = mapped_column(Text, default="")
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created: Mapped[float] = mapped_column(Float, default=time.time)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String, default="candidate")
    manifest: Mapped[dict] = mapped_column(JSON, default=dict)


class Record(Base):
    __tablename__ = "runtime_records"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    kind: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[float] = mapped_column(Float, default=time.time, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


def make_engine(url):
    options = {"check_same_thread": False, "timeout": 15} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=options, pool_pre_ping=True)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def pragmas(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")

    return engine


settings.data_dir.mkdir(parents=True, exist_ok=True)
engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


@contextmanager
def transaction():
    with SessionLocal.begin() as session:
        yield session


def audit(session, actor, action, target="", **details):
    session.add(Audit(actor=actor, action=action, target=target, details=details))


def serialize(row):
    return {
        column.name: getattr(row, column.name)
        for column in row.__table__.columns
        if column.name not in {"source_cipher", "password_hash", "token_hash"}
    }
