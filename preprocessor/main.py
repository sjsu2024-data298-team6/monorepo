from pathlib import Path
import ast
import boto3
import json
import os
import shutil
import time
import wget
import zipfile
import time
from ..keys import GeneralKeys, PreProcessorKeys, DatasetKeys, TrainerKeys
from dataset import *

sqs = boto3.client("sqs", region_name="us-east-1")
s3 = boto3.client("s3")
sns = boto3.client("sns", region_name="us-east-1")


def send_sns(subject, message):
    try:
        sns.publish(
            TargetArn=GeneralKeys.SNS_ARN,
            Message=message,
            Subject=subject,
        )

    except Exception as e:
        print("Failed to send message")
        pass


def print_timestamp(*args, **kwargs):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()):^25}]", end=" ")
    print(*args, **kwargs)


def upload_to_s3(local_path, s3_path, zip_name="upload.zip"):
    zip_path = os.path.join("/tmp", zip_name)  # Temporary path for the zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, local_path))
    print_timestamp(f"Zipped {local_path} to {zip_path}")

    if GeneralKeys.DEPLOYMENT == "dev":
        print_timestamp("Not uploading in dev env")
        return
    s3_key = os.path.join(s3_path, zip_name)

    s3.upload_file(zip_path, GeneralKeys.S3_BUCKET_NAME, s3_key)
    print_timestamp(
        f"Uploaded {zip_path} to s3://{GeneralKeys.S3_BUCKET_NAME}/{s3_key}"
    )


def process_and_upload_dataset(url, dtype, model, names=None):
    send_sns(
        f"Training {model}",
        f"Converting dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    if dtype not in PreProcessorKeys.SUPPORTED_TYPES:
        print_timestamp(f"{dtype} download type not supported")

    if dtype == PreProcessorKeys.TYPE_ROBOFLOW:
        print_timestamp(f"{dtype} support in progress")
        return
        # for dl_format in ROBOFLOW_SUPPORTED_DATASETS:
        #    dataset_dir = download_dataset_from_roboflow(url, dl_format, keys.ROBOFLOW_KEY)
        #     upload_to_s3(dataset_dir, "dataset", zip_name=f"{dl_format}.zip")

    elif dtype == PreProcessorKeys.TYPE_ZIPFILE:
        print_timestamp(f"{dtype} support in progress")
        return

    elif dtype == PreProcessorKeys.TYPE_VISDRONE:
        if names is None:
            print_timestamp("Names are required for visdrone")
            return

        print_timestamp("Downloading original dataset")
        wget.download(url=url, out="visdrone.zip", bar=None)
        with zipfile.ZipFile("visdrone.zip", "r") as zipf:
            dir_name = Path("./" + zipf.namelist()[0])
            print_timestamp("Unzipped to ", dir_name)
            zipf.extractall()

        print_timestamp("Converting to YOLO format")
        visdrone2yolo(dir_name, names)
        upload_to_s3(dir_name, "dataset", zip_name=f"{DatasetKeys.YOLO_FORMAT}.zip")

        splits = ["test", "train", "valid"]
        print_timestamp("Converting to COCO format")
        for split in splits:
            yolo_to_coco(
                dir_name / split / "images",
                dir_name / split / "labels",
                dir_name / split / "images/_annotations.coco.json",
                names,
            )
            shutil.rmtree(dir_name / split / "labels")
            for file_path in (dir_name / split / "images").glob("*"):
                shutil.move(str(file_path), str(dir_name / split))
            os.rmdir(dir_name / split / "images")
        os.remove(dir_name / "data.yaml")
        upload_to_s3(dir_name, "dataset", zip_name=f"{DatasetKeys.COCO_FORMAT}.zip")

        os.remove("visdrone.zip")
        shutil.rmtree(dir_name)
        print_timestamp("Done")

        send_sns(
            f"Training {model}",
            f"Converted dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\ndatasets location: {GeneralKeys.S3_BUCKET_NAME}/datasets/",
        )


def trigger_training(model, params):
    ec2 = boto3.client("ec2", region_name="us-east-1")

    # Define User Data script
    user_data_script = f"""#!/bin/bash
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3-full python3-pip git libgl1 -y
git clone https://ibrahimmkhalid:{GeneralKeys.GITHUB_ACCESS_TOKEN}@github.com/sjsu2024-data298-team6/monorepo /home/ubuntu/trainer
cd /home/ubuntu/trainer
echo "DEPLOYMENT=prod\nS3_BUCKET_NAME={GeneralKeys.S3_BUCKET_NAME}\nSNS_ARN={GeneralKeys.SNS_ARN}\nMODEL_TO_TRAIN={model}\nRUNNER=train" >> .env
echo '{json.dumps(params)}' >> params.json
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
sudo shutdown -h now
    """

    # Launch EC2 instance
    response = ec2.run_instances(
        ImageId="ami-015c62e8068dd8f78",
        InstanceType="g5.2xlarge",
        InstanceInitiatedShutdownBehavior="terminate",
        KeyName="sjsu-fall24-data298-team6-key-pair",
        MinCount=1,
        MaxCount=1,
        UserData=user_data_script,
        IamInstanceProfile={
            "Arn": os.getenv("EC2_INSTANCE_IAM_ARN"),
        },
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "Encrypted": False,
                    "DeleteOnTermination": True,
                    "Iops": 3000,
                    "SnapshotId": "snap-00618611224312cc9",
                    "VolumeSize": 60,
                    "VolumeType": "gp3",
                    "Throughput": 125,
                },
            }
        ],
        NetworkInterfaces=[
            {
                "AssociatePublicIpAddress": True,
                "DeviceIndex": 0,
                "Groups": [
                    "sg-0ae6a08ce3772678c",
                ],
            },
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "Name",
                        "Value": f"sfdt-trainer-{model}",
                    },
                    {
                        "Key": "d298_task_type",
                        "Value": "training",
                    },
                ],
            },
        ],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print_timestamp(f"Trainer EC2 instance launched: {instance_id}")
    send_sns(f"Training {model}", f"Trainer EC2 instance launched: {instance_id}")


def listen_to_sqs():
    while True:
        response = sqs.receive_message(
            QueueUrl=GeneralKeys.SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,
        )

        if "Messages" in response:
            message = response["Messages"][0]
            receipt_handle = message["ReceiptHandle"]
            body = ast.literal_eval(message["Body"])
            try:
                url = body["url"]
                dtype = body["dtype"]
                model = body["model"]
                params = body["params"]
                names = body["names"]

                # Delete message early to avoid over run model training
                sqs.delete_message(
                    QueueUrl=GeneralKeys.SQS_QUEUE_URL, ReceiptHandle=receipt_handle
                )
                print_timestamp("Processed and deleted message from SQS.")

                # Process the dataset
                process_and_upload_dataset(
                    url=url, dtype=dtype, names=names, model=model
                )
                trigger_training(model, params)

            except Exception as e:
                print_timestamp(f"Error processing message: {e}")
                send_sns(
                    "Error | reading request",
                    f"""Project: pipeline
                         Error: {e}""",
                )
                sqs.delete_message(
                    QueueUrl=GeneralKeys.SQS_QUEUE_URL, ReceiptHandle=receipt_handle
                )
                print_timestamp("Deleted message from SQS with errors")
        else:
            print_timestamp("No messages in queue. Waiting...")
        time.sleep(5)  # Poll every 5 seconds


def run():
    if GeneralKeys.DEPLOYMENT == "dev":
        process_and_upload_dataset(
            "file:///mnt/d/datasets/VisDroneSmall.zip",
            dtype=PreProcessorKeys.TYPE_VISDRONE,
            names=[
                "pedestrian",
                "people",
                "bicycle",
                "car",
                "van",
                "truck",
                "tricycle",
                "awning-tricycle",
                "bus",
                "motor",
            ],
            model=TrainerKeys.MODEL_YOLO,
        )
    else:
        listen_to_sqs()
