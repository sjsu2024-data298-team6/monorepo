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
from db.writer import DatabaseWriter
from db import db_manager
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
        TrainerKeys.MODEL_RTDETR_CUSTOM,
    ]:
        from trainer.ultralytics_trainer import train_main
    else:
        return ("Model {model} not yet supported", None, None), False

    try:
        return train_main(logger, model, extra_keys), True
    except Exception as e:
        return (
            (
                f"Training of model '{model}' failed somewhere, please check manually\n\n\nExcpetion:\n{e}\n\n\nTraceback:\n{traceback.format_exc()}",
                None,
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
        TrainerKeys.MODEL_RTDETR_CUSTOM,
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
        f"Model:{model}\nDataset s3 key:{dataset}\nTags:{tags}\nStarted training: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    time_start = time.time()
    (model_results, runs_dir, the_rest), success = train(model, extra_keys)

    time_taken = time.time() - time_start
    upload_message = ""
    if success:
        assert the_rest is not None
        assert runs_dir is not None
        session = next(db_manager.get_db())
        try:
            results = model_results.splitlines()
            inf = float(results[-1].split(":")[-1].strip())
            iou = float(results[-2].split(":")[-1].strip())

            ts = int(time.time())
            upload_message, s3_key = s3.upload_zip_to_s3(runs_dir, "runs/", f"{model}_{ts}.zip")
            _, wt_key = s3.upload_file_to_s3(the_rest["best_wt"], "runs/", f"{model}_{ts}_weights.pt")

            tfjs_s3_key = ""
            if the_rest["tfjs_path"] != "":
                shutil.move(the_rest["tfjs_path"], f"{model}_{ts}_weights_tfjs")
                _, tfjs_s3_key = s3.upload_folder_to_s3(f"{model}_{ts}_weights_tfjs", "tfjs_models")

            extras = the_rest["extras"]
            if "YAML_URL" in extra_keys.keys():
                extras["YAML_URL"] = extra_keys["YAML_URL"]

            writer = DatabaseWriter(session)
            writer.create_model_result(
                dataset_id=int(extra_keys["DATASET_ID"]),
                model_type_id=int(extra_keys["MODEL_ID"]),
                params=the_rest["params"],
                extras=extras,
                iou_score=iou,
                inference_time=inf,
                map50_score=the_rest["map50"],
                map5095_score=the_rest["map5095"],
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
    message.append(upload_message)
    message.append(model_results)

    message = "\n\n\n".join(message)
    sns.send(f"Training {model}", message)
