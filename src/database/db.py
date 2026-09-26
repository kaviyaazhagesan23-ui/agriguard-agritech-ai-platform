from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

load_dotenv()

DEFAULT_DATABASE_URL = "sqlite:///paddywise_local.db"


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def make_engine(url: str | None = None):
    selected = url or database_url()
    kwargs = {"future": True, "pool_pre_ping": True}
    if selected.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(selected, **kwargs)


def init_database(engine=None) -> None:
    engine = engine or make_engine()
    Base.metadata.create_all(engine)


def session_factory(engine=None):
    return sessionmaker(bind=engine or make_engine(), autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(engine=None) -> Iterator[Session]:
    factory = session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
