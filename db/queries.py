from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from .models import (
    Dataset,
)
from . import db_manager

class DatasetQueries:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self,  id: int) -> Optional[Any]:
        """Generic method to get a record by ID"""
        return self.db.query(Dataset).filter(Dataset.id == id).first()


def queries() -> DatasetQueries:
    """Factory function to get DatasetQueries instance"""
    return DatasetQueries(next(db_manager.get_db()))
