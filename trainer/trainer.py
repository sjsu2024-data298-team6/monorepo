import shutil
import zipfile
import os
import time
import traceback
from keys import GeneralKeys, PreProcessorKeys, TrainerKeys, DatasetKeys
import logging
import json
from aws_handler import S3Handler, SNSHandler
from db.queries import queries
from preprocessor.dataset import download_dataset_from_roboflow


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


def train(model, extra_keys):
    logger.info(f"Started training for {model}")
    if model in [
        TrainerKeys.MODEL_YOLO,
        TrainerKeys.MODEL_RTDETR,
        TrainerKeys.MODEL_YOLO_CUSTOM,
    ]:
        from trainer.ultralytics_trainer import train_main
    else:
        return ("Model {model} not yet supported", None), False

    try:
        return train_main(logger, model, extra_keys), True
    except Exception as e:
        return (
            (
                f"Training of model '{model}' failed somewhere, please check manually\n\n\nExcpetion:\n{e}\n\n\nTraceback:\n{traceback.format_exc()}",
                None,
            ),
            False,
        )


def getDefaultDataset(model):
    dataset = None
    if model in [
        TrainerKeys.MODEL_RTDETR,
        TrainerKeys.MODEL_YOLO,
        TrainerKeys.MODEL_YOLO_CUSTOM,
    ]:
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
        logger.warning("Model not found, using default model")
        logger.warning("default model will be deprecated")
        model = TrainerKeys.MODEL_YOLO

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
        f"Training {model}",
        f"Model:{model}\n"
        f"Dataset s3 key:{dataset}\n"
        f"Tags:{tags}\n"
        f"Started training: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    time_start = time.time()
    (model_results, runs_dir), success = train(model, extra_keys)

    time_taken = time.time() - time_start
    upload_message = ""
    if success:
        upload_message = s3.upload_zip_to_s3(runs_dir, "runs/", f"{model}.zip")
    message = ["Training successful!" if success else "Training Failed!"]
    message.append(f"Runtime: {time_taken:.4f} seconds")
    message.append(upload_message)
    message.append(model_results)

    message = "\n\n\n".join(message)
    sns.send(f"Training {model}", message)
