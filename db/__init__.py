from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm.session import sessionmaker as sessionmaker_type
from sqlalchemy.pool import QueuePool

from keys import GeneralKeys

Base = declarative_base()


class DatabaseManager:
    _instance: Optional["DatabaseManager"] = None
    _engine = None
    _SessionLocal = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        assert GeneralKeys.DB_URI is not None
        if self._engine is None:
            self._engine = create_engine(
                GeneralKeys.DB_URI,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
            )

            # Create session factory
            self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

    @property
    def engine(self):
        return self._engine

    def get_db(self):
        assert type(self._SessionLocal) is sessionmaker_type
        db = self._SessionLocal()
        try:
            yield db
        finally:
            db.close()


db_manager = DatabaseManager()
