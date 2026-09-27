import hashlib
import secrets
import time

from argon2 import PasswordHasher
from cryptography.fernet import Fernet
from fastapi import HTTPException, Request
from sqlalchemy import select

from surveilx.config import settings
from surveilx.database import Session, User, audit, transaction

hasher = PasswordHasher()


def cipher():
    path = settings.data_dir / "encryption.key"
    if not path.exists():
        try:
            with path.open("xb") as handle:
                handle.write(Fernet.generate_key())
            path.chmod(0o600)
        except FileExistsError:
            pass
    return Fernet(path.read_bytes())


def initialize_admin():
    with transaction() as session:
        if session.scalar(select(User).where(User.username == "admin")):
            return
        password = settings.admin_password or secrets.token_urlsafe(20)
        if len(password) < 12:
            raise ValueError("SURVEILX_ADMIN_PASSWORD must contain at least 12 characters")
        session.add(User(username="admin", password_hash=hasher.hash(password), role="admin"))
        if not settings.admin_password:
            path = settings.data_dir / "initial-admin-password.txt"
            path.write_text(password, encoding="utf-8")
            path.chmod(0o600)


def create_session(session, user):
    token = secrets.token_urlsafe(48)
    session.add(
        Session(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires=time.time() + 8 * 3600,
        )
    )
    return token


def identify(token):
    if not token:
        raise HTTPException(401, "Sign in required")
    with transaction() as session:
        record = session.get(Session, hashlib.sha256(token.encode()).hexdigest())
        if not record or record.expires < time.time():
            raise HTTPException(401, "Session expired")
        user = session.get(User, record.user_id)
        if not user:
            raise HTTPException(401, "Unknown user")
        return user


def current_user(request: Request):
    return identify(request.cookies.get("surveilx_session"))


def require(user, *roles):
    if user.role != "admin" and user.role not in roles:
        raise HTTPException(403, "Role does not permit this action")


def revoke(token):
    with transaction() as session:
        record = session.get(Session, hashlib.sha256((token or "").encode()).hexdigest())
        if record:
            audit(session, record.user_id, "logout")
            session.delete(record)
