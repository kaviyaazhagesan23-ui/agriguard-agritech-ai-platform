from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Role, User
from .security import hash_password, verify_password


def register_user(session: Session, *, name: str, email: str, password: str, role: str, phone: str | None = None) -> User:
    normalized = email.strip().lower()
    if role not in {Role.FARMER.value, Role.BUYER.value}:
        raise ValueError("Role must be farmer or buyer.")
    if session.scalar(select(User).where(User.email == normalized)):
        raise ValueError("An account with this email already exists.")
    user = User(name=name.strip(), email=normalized, phone=phone, password_hash=hash_password(password), role=role, is_active=True)
    session.add(user)
    session.flush()
    return user


def authenticate_user(session: Session, *, email: str, password: str) -> User | None:
    user = session.scalar(select(User).where(User.email == email.strip().lower(), User.is_active.is_(True)))
    if user and verify_password(password, user.password_hash):
        return user
    return None
