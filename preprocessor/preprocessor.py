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
from keys import GeneralKeys, PreProcessorKeys, DatasetKeys, TrainerKeys
import logging
from preprocessor.dataset import *
from aws_handler import SNSHandler

logger = logging.getLogger("sfdt_preprocessor")
logging.basicConfig(
    filename="sfdt_preprocessor.log",
    encoding="utf-8",
    level=logging.INFO,
)

sqs = boto3.client("sqs", region_name="us-east-1")
s3 = boto3.client("s3")
sns = SNSHandler(logger=logger)


def upload_to_s3(local_path, s3_path, zip_name="upload.zip"):
    zip_path = os.path.join("/tmp", zip_name)  # Temporary path for the zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, local_path))
    logger.info(f"Zipped {local_path} to {zip_path}")

    if GeneralKeys.DEPLOYMENT == "dev":
        logger.info("Not uploading in dev env")
        return
    s3_key = os.path.join(s3_path, zip_name)

    s3.upload_file(zip_path, GeneralKeys.S3_BUCKET_NAME, s3_key)
    logger.info(f"Uploaded {zip_path} to s3://{GeneralKeys.S3_BUCKET_NAME}/{s3_key}")


def process_and_upload_dataset(url, dtype, model, names=None):
    sns.send(
        f"Training {model}",
        f"Converting dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    if dtype not in PreProcessorKeys.SUPPORTED_TYPES:
        logger.warning(f"{dtype} download type not supported")

    if dtype == PreProcessorKeys.TYPE_ROBOFLOW:
        logger.warning(f"{dtype} support in progress")
        return
        # for dl_format in ROBOFLOW_SUPPORTED_DATASETS:
        #    dataset_dir = download_dataset_from_roboflow(url, dl_format, keys.ROBOFLOW_KEY)
        #     upload_to_s3(dataset_dir, "dataset", zip_name=f"{dl_format}.zip")

    elif dtype == PreProcessorKeys.TYPE_ZIPFILE:
        logger.warning(f"{dtype} support in progress")
        return

    elif dtype == PreProcessorKeys.TYPE_VISDRONE:
        if names is None:
            logger.error("Names are required for visdrone")
            return

        logger.info(f"Started download for dataset at {url}")
        wget.download(url=url, out="visdrone.zip", bar=None)
        with zipfile.ZipFile("visdrone.zip", "r") as zipf:
            dir_name = Path("./" + zipf.namelist()[0])
            logger.info(f"Unzipped dataset to {dir_name}")
            zipf.extractall()

        logger.info("Converting dataset to YOLO format")
        visdrone2yolo(dir_name, names)
        upload_to_s3(dir_name, "dataset", zip_name=f"{DatasetKeys.YOLO_FORMAT}.zip")

        splits = ["test", "train", "valid"]
        logger.info("Converting dataset to COCO format")
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
        logger.info("Dataset conversions complete")

        sns.send(
            f"Training {model}",
            f"Converted dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\ndatasets location: {GeneralKeys.S3_BUCKET_NAME}/datasets/",
        )


def trigger_training(model, params):
    ec2 = boto3.client("ec2", region_name="us-east-1")

    # Define User Data script
    user_data_script = f"""#!/bin/bash
# update and install required packages
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3-full python3-pip git amazon-ec2-utils libgl1 -y
wget https://amazoncloudwatch-agent.s3.amazonaws.com/debian/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# configure and start cloudwatch agent
echo "{{\\"logs\\":{{\\"logs_collected\\":{{\\"files\\":{{\\"collect_list\\":[{{\\"file_path\\":\\"/home/ubuntu/trainer/sfdt_trainer.log\\",\\"log_group_name\\":\\"sfdt-log-group\\",\\"log_stream_name\\":\\"trainer/{model}/instance-$(ec2-metadata -i | awk '{{print $2}}')\\"}}]}}}}}}}}" | sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# get code
git clone https://ibrahimmkhalid:{GeneralKeys.GITHUB_ACCESS_TOKEN}@github.com/sjsu2024-data298-team6/monorepo /home/ubuntu/trainer

# setup environment
cd /home/ubuntu/trainer
echo "DEPLOYMENT=prod\nS3_BUCKET_NAME={GeneralKeys.S3_BUCKET_NAME}\nSNS_ARN={GeneralKeys.SNS_ARN}\nMODEL_TO_TRAIN={model}\nRUNNER=train" >> .env
echo '{json.dumps(params)}' >> params.json
python3 -m venv venv
source venv/bin/activate

# install python packages and run
pip install -r requirements.txt
python3 main.py

# exit
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
    logger.info(f"Trainer EC2 instance launched: {instance_id}")
    sns.send(
        f"Training {model}",
        f"Trainer EC2 instance launched: {instance_id}",
    )


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
                logger.info("Processed and deleted message from SQS.")

                # Process the dataset
                process_and_upload_dataset(
                    url=url, dtype=dtype, names=names, model=model
                )
                trigger_training(model, params)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                sns.send(
                    "Error | reading request",
                    f"""Project: pipeline
                         Error: {e}""",
                )
                sqs.delete_message(
                    QueueUrl=GeneralKeys.SQS_QUEUE_URL, ReceiptHandle=receipt_handle
                )
                logger.info("Deleted message from SQS with errors")
        else:
            logger.info("No messages in queue. Waiting...")
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
        # trigger_training(TrainerKeys.MODEL_YOLO, {})
    else:
        listen_to_sqs()
