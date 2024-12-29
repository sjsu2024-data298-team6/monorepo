import zipfile
import boto3
import os
from dotenv import load_dotenv
import boto3
import os
import time
import traceback

load_dotenv()

sns = boto3.client("sns", region_name="us-east-1")
s3 = boto3.client("s3")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
SNS_ARN = os.getenv("SNS_ARN")
DEPLOYMENT = os.getenv("DEPLOYMENT")


def download_dataset_from_s3(name):
    s3_key = os.path.join("dataset", name)
    s3.download_file(S3_BUCKET_NAME, s3_key, name)
    with zipfile.ZipFile(name, "r") as z:
        z.extractall("data")


def train(model):
    if model == "yolo":
        from yolo_trainer import train_main
    elif model == "rtdetr":
        from rtdetr_trainer import train_main
    else:
        pass

    try:
        return train_main(), True
    except Exception as e:
        return (
            f"Training of model '{model}' failed somewhere, please check manually\n\n\nExcpetion:\n{e}\n\n\nTraceback:\n{traceback.format_exc()}",
            False,
        )


def upload_to_s3(local_path, s3_path, zip_name="upload.zip"):
    zip_path = os.path.join("/tmp", zip_name)  # Temporary path for the zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, local_path))

    if DEPLOYMENT == "dev":
        print("Not uploading in dev env")
        return ""
    s3_key = os.path.join(s3_path, zip_name)

    s3.upload_file(zip_path, S3_BUCKET_NAME, s3_key)
    ret = f"Uploaded {zip_path} to s3://{S3_BUCKET_NAME}/{s3_key}"
    print(ret)
    return ret


def send_sns(subject, message):
    try:
        sns.publish(
            TargetArn=SNS_ARN,
            Message=message,
            Subject=subject,
        )

    except Exception as e:
        print("Failed to send message")
        pass


if __name__ == "__main__":
    model = os.getenv("MODEL_TO_TRAIN")
    if model is None:
        model = "yolo"

    dataset = None
    if model in ["yolo", "rtdetr"]:
        dataset = "yolo"

    if DEPLOYMENT != "dev":
        download_dataset_from_s3(f"{dataset}.zip")

    send_sns(
        f"Training {model}",
        f"Model:{model}\nDataset:{dataset}\nStarted training: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    time_start = time.time()
    model_results, success = train(model)
    time_taken = time.time() - time_start
    upload_message = ""
    if success:
        upload_message = upload_to_s3("./runs", "runs/", f"{model}.zip")
    message = ["Training successful!" if success else "Training Failed!"]
    message.append(f"Runtime: {time_taken:.4f} seconds")
    message.append(upload_message)
    message.append(model_results)

    message = "\n\n\n".join(message)
    send_sns(f"Training {model}", message)
