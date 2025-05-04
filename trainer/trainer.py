import json
import logging
import os
from dataclasses import asdict, dataclass
import shutil
import time
import traceback
from typing import Tuple, List
from pathlib import Path
import zipfile

from aws_handler import S3Handler, SNSHandler
from db import db_manager
from db.queries import queries
from db.writer import DatabaseWriter
from keys import DatasetKeys, GeneralKeys, PreProcessorKeys, TrainerKeys


@dataclass
class ClassMetrcis:
    name: str
    precision: float
    recall: float
    map50: float
    map5095: float
    iou: float | None


@dataclass
class AllMetrics:
    precision: float
    recall: float
    map50: float
    map5095: float
    iou: float | None
    inference_time: float
    class_metrics: List[ClassMetrcis]


@dataclass
class TrainingResultClass:
    runs_dir: Path
    params: dict
    wandb_logs: str
    best_wt: Path
    tfjs_path: Path | None
    metrics_str: str
    metrics: AllMetrics | None


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
            "file": record.pathname,
            "line": record.lineno,
        }
        return json.dumps(log_entry)


logger = logging.getLogger("sfdt_trainer")
handler = logging.FileHandler("sfdt_trainer.log", encoding="utf-8")

json_formatter = JsonFormatter()
handler.setFormatter(json_formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)

sns = SNSHandler(logger=logger)
s3 = S3Handler(bucket=GeneralKeys.S3_BUCKET_NAME, logger=logger)


def download_dataset_from_s3(s3_key):
    if GeneralKeys.DEPLOYMENT == "dev":
        # random tiny dataset for testing purposes
        from preprocessor.dataset import download_dataset_from_roboflow

        download_dataset_from_roboflow(
            "https://universe.roboflow.com/box-irdnl/boxy-8ddct/dataset/2",
            PreProcessorKeys.ROBOFLOW_YOLOV11,
            PreProcessorKeys.ROBOFLOW_KEY,
            path="data",
        )
        pass
    else:
        s3.download_file(s3_key, "dataset.zip")
        with zipfile.ZipFile("dataset.zip", "r") as z:
            z.extractall("data")


def train(model, extra_keys) -> Tuple[TrainingResultClass | str, bool]:
    model_name = extra_keys["MODEL_NAME"]
    logger.info(f"Started training for {model} | {model_name}")
    if model in TrainerKeys.ULTRALYTICS_TRAINER:
        from trainer.ultralytics_trainer import train_main
    else:
        return f"Model {model} not yet supported", False

    try:
        return train_main(logger, model, extra_keys), True
    except Exception as e:
        return (
            f"Training of model '{model} | {model_name}' failed somewhere, please check manually\n\n\nExcpetion:\n{e}\n\n\nTraceback:\n{traceback.format_exc()}",
            False,
        )


def getDefaultDataset(model):
    dataset = None
    if model in TrainerKeys.ULTRALYTICS_TRAINER:
        dataset = DatasetKeys.YOLO_FORMAT
    dataset = f"dataset/{dataset}.zip"
    return dataset


def run():
    extra_keys = {}
    if ".extra" in os.listdir():
        with open(".extra", "r") as fd:
            for line in fd.readlines():
                key, value = line.split("=")
                extra_keys[key] = value.strip()

    model = os.getenv("MODEL_TO_TRAIN")
    if model is None:
        logger.warning("Model not found")
        sns.send("Training Failed", "Cannot train when model key is not provided. Trainer exited")
        exit()

    tags = []
    if "tags.txt" in os.listdir():
        with open("tags.txt", "r") as fd:
            tags.extend([x.strip() for x in fd.readlines()])
    tags.append(model)
    if "test" not in tags:
        tags.append(GeneralKeys.DEPLOYMENT)

    extra_keys["tags"] = tags

    if "DATASET_ID" in extra_keys.keys():
        dataset_obj = queries().get_dataset_by_id(int(extra_keys["DATASET_ID"]))
        if dataset_obj is None:
            logger.warning("Dataset not found, using default dataset")
            logger.warning("default dataset will be deprecated")
            dataset = getDefaultDataset(model)
        else:
            dataset = dataset_obj.s3Key
    else:
        logger.warning("No dataset ID provided, using default dataset")
        logger.warning("default dataset will be deprecated")
        dataset = getDefaultDataset(model)
    download_dataset_from_s3(dataset)

    sns.send(
        f"Training {model} | {extra_keys['MODEL_NAME']}",
        "\n".join(
            [
                f"Model Name: {extra_keys['MODEL_NAME']}",
                f"Model type:{model}",
                f"Dataset s3 key:{dataset}",
                f"Tags:{tags}",
                f"Started training: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ]
        ),
    )

    time_start = time.time()
    model_results, success = train(model, extra_keys)

    time_taken = time.time() - time_start
    upload_message = []
    if success and GeneralKeys.DEPLOYMENT != "dev":
        assert isinstance(model_results, TrainingResultClass)
        session = next(db_manager.get_db())
        try:
            assert model_results.metrics is not None
            ts = int(time.time())
            msg, s3_key = s3.upload_zip_to_s3(model_results.runs_dir, "runs/", f"{model}_{ts}.zip")
            upload_message.append(msg)
            msg, wt_key = s3.upload_file_to_s3(model_results.best_wt, "runs/", f"{model}_{ts}_weights.pt")
            upload_message.append(msg)

            tfjs_s3_key = ""
            if model_results.tfjs_path is not None:
                shutil.move(model_results.tfjs_path, f"{model}_{ts}_weights_tfjs")
                msg, tfjs_s3_key = s3.upload_folder_to_s3(f"{model}_{ts}_weights_tfjs", "tfjs_models")
                upload_message.append(msg)

            extras = {
                "wandb_logs": model_results.wandb_logs,
                "detailed_metrics": str(asdict(model_results.metrics)),
            }

            if "YAML_URL" in extra_keys.keys():
                extras["YAML_URL"] = extra_keys["YAML_URL"]

            writer = DatabaseWriter(session)
            writer.create_model_result(
                dataset_id=int(extra_keys["DATASET_ID"]),
                model_type_id=int(extra_keys["MODEL_ID"]),
                model_name=str(extra_keys["MODEL_NAME"]),
                params=model_results.params,
                extras=extras,
                iou_score=model_results.metrics.iou,
                inference_time=model_results.metrics.inference_time,
                map50_score=model_results.metrics.map50,
                map5095_score=model_results.metrics.map5095,
                tags=tags,
                model_s3_key=wt_key,
                results_s3_key=s3_key,
                tfjs_s3_key=tfjs_s3_key,
                is_active="test" not in tags,
            )
        except Exception as e:
            logger.info(f"Failed to upload results to database {e}")
            pass
        finally:
            session.close()

    message = ["Training successful!" if success else "Training Failed!"]
    message.append(f"Runtime: {time_taken:.4f} seconds")
    message.append("\n".join(upload_message))
    if isinstance(model_results, TrainingResultClass):
        message.append(model_results.wandb_logs)
        message.append(model_results.metrics_str)
    else:
        message.append(model_results)

    message = "\n\n".join(message)
    sns.send(f"Training {model} | {extra_keys['MODEL_NAME']}", message)
