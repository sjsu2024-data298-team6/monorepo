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
import traceback
from keys import GeneralKeys, PreProcessorKeys, DatasetKeys, TrainerKeys
import hashlib
import logging
from preprocessor.dataset import *
from aws_handler import S3Handler, SNSHandler

logger = logging.getLogger("sfdt_preprocessor")
logging.basicConfig(
    filename="sfdt_preprocessor.log",
    encoding="utf-8",
    level=logging.INFO,
)

sns = SNSHandler(logger=logger)
s3 = S3Handler(bucket=GeneralKeys.S3_BUCKET_NAME, logger=logger)


def process_and_upload_dataset(url, dtype, names=None):
    sns.send(
        f"Converting {dtype} dataset",
        f"Converting dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    if dtype not in PreProcessorKeys.SUPPORTED_TYPES:
        logger.warning(f"{dtype} download type not supported")
        return

    # check existing dataset
    should_combine = False
    if s3.check_file_exists(f"dataset/{DatasetKeys.YOLO_FORMAT}.zip", logger=logger):
        logger.info(f"Dataset already exists in s3, combining with new dataset")
        s3.download_file(f"dataset/{DatasetKeys.YOLO_FORMAT}.zip", "yolo_old.zip")
        should_combine = True

    dir_name = None

    ### Create YOLO format of datasets
    if dtype == PreProcessorKeys.TYPE_ROBOFLOW:
        if names is not None:
            logger.warning(
                "class names have been provided for roboflow dataset,"
                + " however roboflow does not need them."
                + " Provided class names will be discarded"
            )

        logger.info(f"Started download for roboflow dataset at {url}")
        dir_name = download_dataset_from_roboflow(
            url, PreProcessorKeys.ROBOFLOW_YOLOV11, PreProcessorKeys.ROBOFLOW_KEY
        )
        dir_name = Path(dir_name)
        logger.info(f"Finished download of {url} to {dir_name}")
        with open(dir_name / "data.yaml", "r") as fd:
            yaml_content = yaml.safe_load(fd)
            names = [name.lower() for name in yaml_content["names"]]
            yaml_content["names"] = names

        with open(dir_name / "data.yaml", "w") as fd:
            yaml.safe_dump(yaml_content, fd)

        splits = ["test", "train", "valid"]
        for split in splits:
            labels_path = dir_name / split / "labels"
            for f in labels_path.glob("*.txt"):
                clean = []
                with open(f, "r") as fd:
                    lines = fd.read()
                    lines = lines.strip()
                    lines = lines.splitlines()
                for line in lines:
                    tmp = line.strip().split()
                    if len(tmp) > 5:
                        tmp = convert_mask_to_bbox(tmp)
                        tmp = [str(t) for t in tmp]
                    tmp = " ".join(tmp)
                    clean.append(tmp)
                clean = "\n".join(clean)
                with open(f, "w") as fd:
                    fd.write(clean)

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

        os.remove("visdrone.zip")

    ### Make sure stuff is working
    try:
        assert isinstance(dir_name, Path)
        assert isinstance(names, list)
    except AssertionError:
        logger.error("Something went wrong converting to YOLO")
        return

    checksum_blob = []
    if s3.check_file_exists("dataset/checksum_blob.txt", logger=logger):
        s3.download_file("dataset/checksum_blob.txt", "checksum_blob.txt")
        with open("checksum_blob.txt", "r") as fd:
            checksum_blob = fd.read().splitlines()

    if should_combine:
        logger.info("Combining datasets")

        with zipfile.ZipFile("yolo_old.zip", "r") as z:
            z.extractall("old")
        old_dir = Path("old")

        with open(old_dir / "data.yaml", "r") as fd:
            old_yaml = yaml.safe_load(fd)
            old_names = old_yaml["names"]
        with open(dir_name / "data.yaml", "r") as fd:
            new_yaml = yaml.safe_load(fd)
            new_names = new_yaml["names"]

        combined_names = old_names
        class_id_map = {}
        for i, name in enumerate(new_names):
            if name not in old_names:
                combined_names.append(name)
                class_id_map[str(i)] = str(len(combined_names) - 1)
            else:
                class_id_map[str(i)] = str(old_names.index(name))

        with open(old_dir / "data.yaml", "w") as fd:
            new_yaml["names"] = combined_names
            new_yaml["nc"] = len(combined_names)
            yaml.safe_dump(new_yaml, fd)

        current_ts = int(time.time())
        skipped = 0
        for split in ["test", "train", "valid"]:
            for img_path in (dir_name / split / "images").glob("*"):
                img_checksum = hashlib.md5(img_path.read_bytes()).hexdigest()
                if img_checksum in checksum_blob:
                    skipped += 1
                    continue
                checksum_blob.append(img_checksum)

                txt_path = dir_name / split / "labels" / (img_path.stem + ".txt")
                with open(txt_path, "r") as fd:
                    lines = fd.read().strip().splitlines()
                new_lines = []
                for line in lines:
                    tmp = line.split()
                    try:
                        tmp[0] = class_id_map[tmp[0]]
                    except KeyError:
                        tmp[0] = class_id_map[str(int(float(tmp[0])))]
                    new_lines.append(" ".join(tmp))
                with open(txt_path, "w") as fd:
                    fd.write("\n".join(new_lines))

                # ts to avoid name conflicts
                shutil.move(
                    str(img_path),
                    str(old_dir / split / "images" / f"{current_ts}_{img_path.name}"),
                )
                shutil.move(
                    str(txt_path),
                    str(old_dir / split / "labels" / f"{current_ts}_{txt_path.name}"),
                )

        shutil.rmtree(dir_name)
        os.remove("yolo_old.zip")
        dir_name = old_dir
        logger.info(f"Finished combining datasets with {skipped} images skipped")

    else:
        for split in ["test", "train", "valid"]:
            for img_path in (dir_name / split / "images").glob("*"):
                img_checksum = hashlib.md5(img_path.read_bytes()).hexdigest()
                checksum_blob.append(img_checksum)

    with open("./checksum_blob.txt", "w") as fd:
        fd.write("\n".join(checksum_blob))

    s3.upload_file_to_s3("./checksum_blob.txt", "dataset", "checksum_blob.txt")

    ### Zip and upload to s3
    s3.upload_zip_to_s3(dir_name, "dataset", zip_name=f"{DatasetKeys.YOLO_FORMAT}.zip")

    ### Convert to COCO format
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
    s3.upload_zip_to_s3(dir_name, "dataset", zip_name=f"{DatasetKeys.COCO_FORMAT}.zip")
    shutil.rmtree(dir_name)

    logger.info("Dataset conversions complete")
    sns.send(
        f"Converting {dtype} dataset",
        f"Converted dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\ndatasets location: {GeneralKeys.S3_BUCKET_NAME}/datasets/",
    )
    return


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
echo '{params}' >> params.json
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
    return instance_id


def check_instance_terminated(instance_id):
    ec2 = boto3.client("ec2", region_name="us-east-1")
    response = ec2.describe_instance_status(
        InstanceIds=[instance_id], IncludeAllInstances=True
    )
    return response["InstanceStatuses"][0]["InstanceState"]["Name"] == "terminated"


def listen_to_sqs():
    sqs = boto3.client("sqs", region_name="us-east-1")
    instance_id = None
    _counter = 0
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
            data = body["data"]
            try:
                url = data["url"]
                dtype = data["datasetType"]
                model = data["model"]
                params = data["params"]
                names = data["names"]
                if type(names) == str:
                    names = names.split(",")

                # Delete message early to avoid over run model training
                sqs.delete_message(
                    QueueUrl=GeneralKeys.SQS_QUEUE_URL, ReceiptHandle=receipt_handle
                )
                logger.info("Processed and deleted message from SQS.")

                # Process the dataset
                process_and_upload_dataset(
                    url=url,
                    dtype=dtype,
                    names=names,
                )
                # instance_id = trigger_training(model, params)
                #
                # # make sure instance id is available on api
                # time.sleep(60)
                #
                # while not check_instance_terminated(instance_id):
                #     logger.info("Currently training...")
                #     time.sleep(30)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                sns.send(
                    "Error | reading request",
                    f"""Project: pipeline
                         Error: {e}
                         Traceback: {traceback.format_exc()}""",
                )
                sqs.delete_message(
                    QueueUrl=GeneralKeys.SQS_QUEUE_URL, ReceiptHandle=receipt_handle
                )
                logger.info("Deleted message from SQS with errors")
        else:
            if _counter == 360:
                _counter = 0
                logger.info("No messages in queue. Waiting...")
        time.sleep(5)  # Poll every 5 seconds
        _counter += 1


def run():
    sns.send("Preprocessor", "Preprocessor started/restarted")
    logger.info("Preprocessor started/restarted")

    if GeneralKeys.DEPLOYMENT == "dev":
        # process_and_upload_dataset(
        #     "file:///mnt/d/datasets/VisDroneSmall.zip",
        #     dtype=PreProcessorKeys.TYPE_VISDRONE,
        #     names=[
        #         "pedestrian",
        #         "people",
        #         "bicycle",
        #         "car",
        #         "van",
        #         "truck",
        #         "tricycle",
        #         "awning-tricycle",
        #         "bus",
        #         "motor",
        #     ],
        # )
        process_and_upload_dataset(
            "https://universe.roboflow.com/drone-obstacle-detection/drone-object-detection-yhpn6/dataset/15",
            dtype=PreProcessorKeys.TYPE_ROBOFLOW,
        )
        # trigger_training(TrainerKeys.MODEL_YOLO, {})
    else:
        listen_to_sqs()
