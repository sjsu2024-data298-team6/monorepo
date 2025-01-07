import zipfile
import os
import time
import traceback
from keys import GeneralKeys, TrainerKeys, DatasetKeys
import logging
from aws_handler import S3Handler, SNSHandler


logger = logging.getLogger("sfdt_trainer")
logging.basicConfig(
    filename="sfdt_trainer.log",
    encoding="utf-8",
    level=logging.INFO,
)

sns = SNSHandler(logger=logger)
s3 = S3Handler(bucket=GeneralKeys.S3_BUCKET_NAME, logger=logger)


def download_dataset_from_s3(name):
    s3_key = os.path.join("dataset", name)
    s3.download_file(s3_key, name)
    with zipfile.ZipFile(name, "r") as z:
        z.extractall("data")


def train(model):
    if model == TrainerKeys.MODEL_YOLO:
        from trainer.yolo_trainer import train_main
    elif model == TrainerKeys.MODEL_RTDETR:
        from trainer.rtdetr_trainer import train_main
    else:
        return "Model {model} not yet supported", False

    try:
        return train_main(), True
    except Exception as e:
        return (
            f"Training of model '{model}' failed somewhere, please check manually\n\n\nExcpetion:\n{e}\n\n\nTraceback:\n{traceback.format_exc()}",
            False,
        )


def getDataset(model):
    dataset = None
    if model in [
        TrainerKeys.MODEL_RTDETR,
        TrainerKeys.MODEL_YOLO,
    ]:
        dataset = DatasetKeys.YOLO_FORMAT
    return dataset


def run():
    model = os.getenv("MODEL_TO_TRAIN")
    if model is None:
        model = TrainerKeys.MODEL_YOLO

    dataset = getDataset(model)

    if GeneralKeys.DEPLOYMENT != "dev":
        download_dataset_from_s3(f"{dataset}.zip")

    sns.send(
        f"Training {model}",
        f"Model:{model}\nDataset:{dataset}\nStarted training: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    time_start = time.time()
    model_results, success = train(model)
    time_taken = time.time() - time_start
    upload_message = ""
    if success:
        upload_message = s3.upload_zip_to_s3("./runs", "runs/", f"{model}.zip")
    message = ["Training successful!" if success else "Training Failed!"]
    message.append(f"Runtime: {time_taken:.4f} seconds")
    message.append(upload_message)
    message.append(model_results)

    message = "\n\n\n".join(message)
    sns.send(f"Training {model}", message)
