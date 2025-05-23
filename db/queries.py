from typing import Any, List, Optional

from sqlalchemy.orm import Session

from . import db_manager
from .models import Dataset, DatasetBaseType, ModelBaseType


class QueryEngine:
    def __init__(self, db: Session):
        self.db = db

    def get_dataset_by_id(self, id: int) -> Optional[Dataset]:
        """Generic method to get a record by ID"""
        return self.db.query(Dataset).filter(Dataset.id == id).first()

    def get_dataset_type_by_value(self, key: str) -> Optional[DatasetBaseType]:
        return (
            self.db.query(DatasetBaseType).filter(DatasetBaseType.value == key).filter(DatasetBaseType.isActive).first()
        )

    def get_model_by_key(self, key: str) -> Optional[Any]:
        model = self.db.query(ModelBaseType).filter(ModelBaseType.value == key).filter(ModelBaseType.isActive).first()
        if model is not None:
            return model.id
        return -1

    def datasets_with_same_links(self, links: List[str]) -> List[Dataset]:
        datasets = self.db.query(Dataset).all()

        def check(x: Dataset) -> bool:
            assert isinstance(x.links, list)
            return set(x.links) == set(links)

        return list(filter(check, datasets))


def queries() -> QueryEngine:
    """Factory function to get DatasetQueries instance"""
    return QueryEngine(next(db_manager.get_db()))
