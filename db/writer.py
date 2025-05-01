from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .models import Dataset, ModelResults


class DatabaseWriter:
    def __init__(self, session: Session):
        self.session = session

    def create_dataset(
        self,
        datasetTypeId: int,
        name: str,
        tags: List[str],
        links: List[str],
        s3Key: str,
        checksumBlobS3Key: str,
    ) -> Dataset:
        dataset = Dataset(
            datasetTypeId=datasetTypeId,
            name=name,
            tags=tags,
            links=links,
            s3Key=s3Key,
            checksumBlobS3Key=checksumBlobS3Key,
        )
        self.session.add(dataset)
        self.session.commit()
        return dataset

    def create_model_result(
        self,
        dataset_id: int,
        model_type_id: int,
        model_name: str,
        params: Dict[str, Any],
        extras: Dict[str, Any],
        iou_score: Optional[float],
        map50_score: Optional[float],
        map5095_score: Optional[float],
        inference_time: Optional[float],
        tags: List[str],
        results_s3_key: str,
        model_s3_key: str,
        tfjs_s3_key: str,
        is_active: Optional[bool],
    ) -> ModelResults:
        model_result = ModelResults(
            datasetId=dataset_id,
            modelTypeId=model_type_id,
            modelName=model_name,
            params=params,
            extras=extras,
            iouScore=iou_score,
            map50Score=map50_score,
            map5095Score=map5095_score,
            inferenceTime=inference_time,
            tags=tags,
            resultsS3Key=results_s3_key,
            modelS3Key=model_s3_key,
            tfjsS3Key=tfjs_s3_key,
            isActive=is_active,
        )
        self.session.add(model_result)
        self.session.commit()
        return model_result
