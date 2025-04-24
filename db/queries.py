from typing import Any, Optional

from sqlalchemy.orm import Session

from . import db_manager
from .models import Dataset, ModelBaseType


class QueryEngine:
    def __init__(self, db: Session):
        self.db = db

    def get_dataset_by_id(self, id: int) -> Optional[Any]:
        """Generic method to get a record by ID"""
        return self.db.query(Dataset).filter(Dataset.id == id).first()

    def get_model_by_key(self, key: str) -> Optional[Any]:
        model = self.db.query(ModelBaseType).filter(ModelBaseType.value == key).filter(ModelBaseType.isActive).first()
        if model is not None:
            return model.id
        return -1


def queries() -> QueryEngine:
    """Factory function to get DatasetQueries instance"""
    return QueryEngine(next(db_manager.get_db()))
