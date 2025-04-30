import hashlib
import json
import logging
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import boto3
import wget

from aws_handler import S3Handler, SNSHandler
from db import db_manager
from db.models import Dataset
from db.queries import queries
from db.writer import DatabaseWriter
from keys import DatasetKeys, GeneralKeys, PreProcessorKeys, TrainerKeys
from preprocessor.dataset import (
    convert_roboflow_yolo_to_bbox,
    download_dataset_from_roboflow,
    get_roboflow_classnames_and_fix,
    perform_combine,
    visdrone2yolo,
    yolo_to_coco,
)

logger = logging.getLogger("sfdt_preprocessor")
logging.basicConfig(filename="sfdt_preprocessor.log", encoding="utf-8", level=logging.INFO)

sns = SNSHandler(logger=logger)
s3 = S3Handler(bucket=GeneralKeys.S3_BUCKET_NAME, logger=logger)


def verify_dataset_info_for_processing(data) -> Tuple[bool, str]:
    dataset_name = data.get("name", None)
    url = data.get("url", None)
    dtype = data.get("datasetType", None)
    if dataset_name is None or url is None or dtype is None:
        return (
            False,
            f"Something went wrong trying process_and_upload_dataset with the following:\n\n{json.dumps(data)}",
        )

    should_combine = data.get("shouldCombine", False)
    combine_id = data.get("combineID", None)
    if should_combine and combine_id is None:
        return False, f"process_and_upload_dataset tried to combine without combine id:\n\n{json.dumps(data)}"

    if dtype not in PreProcessorKeys.SUPPORTED_TYPES:
        return False, f"{dtype} download type not supported"

    return True, ""


def check_valid_combination_id(combine_id) -> Tuple[bool, str, Optional[Dataset]]:
    is_valid = True
    logger.info("Finding dataset to combine with")
    dataset_obj = queries().get_dataset_by_id(combine_id)
    dataset_obj_old = None
    combine_id_old = None

    first_fail_msg = "Could not find provided dataset, cannot combine!"
    if dataset_obj is None:
        is_valid = False

    if dataset_obj is not None and dataset_obj.datasetType.value == "coco":
        # Assumption is that yolo will always be defined before coco
        # The rest of the code assumes a yolo format will be combined with before continuing
        logger.info("A coco dataset was provided to combine, attempting to find yolo format")
        first_fail_msg = "Could not find yolo format of provided dataset, cannot combine!"
        dataset_obj_old = dataset_obj
        combine_id_old = combine_id
        combine_id -= 1
        dataset_obj = queries().get_dataset_by_id(combine_id)
        if dataset_obj is None:
            is_valid = False
        if dataset_obj is not None and dataset_obj.datasetType.value != "yolo":
            is_valid = False

    if not is_valid:
        msg = [
            first_fail_msg,
            "Instead, the dataset will be uploaded and converted on its own",
            f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "Tried combining with:",
        ]

        if dataset_obj_old is not None:
            msg.append(f"Database entry for ID {combine_id_old}:\n")
            msg.append(f"{dataset_obj_old}")

        if dataset_obj is not None:
            msg.append(f"Database entry for ID {combine_id}:\n")
            msg.append(f"{dataset_obj}")

        msg = "\n".join(msg)
        return is_valid, msg, None
    return is_valid, "", dataset_obj


def upload_dataset_and_make_entry(
    name, tags, links, dir_name, checksum_blob, key
) -> Tuple[bool, int, Tuple[str, str], str]:
    current_ts = str(int(time.time()))
    checksum_fn = f"checksum_{current_ts}.txt"
    zip_name = f"{key}_{current_ts}.zip"

    with open(checksum_fn, "w") as fd:
        fd.write("\n".join(checksum_blob))

    session = next(db_manager.get_db())
    writer = DatabaseWriter(session)

    dataset_type = queries().get_dataset_type_by_value(key)
    if dataset_type is None:
        return False, -1, ("", ""), f"{key} dataset type does not exist? Check database"

    ### upload to s3 and make db entry
    _, checksum_key = s3.upload_file_to_s3(checksum_fn, "dataset", checksum_fn)
    _, dataset_key = s3.upload_zip_to_s3(dir_name, "dataset", zip_name=zip_name)
    assert isinstance(dataset_type.id, int)
    new_entry = writer.create_dataset(
        datasetTypeId=dataset_type.id,
        name=name,
        tags=tags,
        links=links,
        s3Key=f"dataset/{zip_name}",
        checksumBlobS3Key=f"dataset/{checksum_fn}",
    )
    assert isinstance(new_entry.id, int)
    return True, new_entry.id, (checksum_key, dataset_key), ""


def process_and_upload_dataset(data):
    success, msg = verify_dataset_info_for_processing(data)
    if not success:
        logger.warning(msg)
        sns.send("Something went wrong", msg)
        return

    dataset_name = data.get("name", None)
    url = data.get("url", None)
    dtype = data.get("datasetType", None)
    should_combine = data.get("shouldCombine", False)
    combine_id = data.get("combineID", None)
    tags = data.get("tags", [dataset_name.lower().replace(" ", "-").strip()])

    sns.send(
        f"Converting {dtype} dataset | {dataset_name} | start",
        "\n".join(
            [
                f"Converting {dtype} dataset | Start",
                f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"tags: {tags}\n\n",
                "Data dump:",
                json.dumps(data),
            ]
        ),
    )

    dataset_obj = None
    links = [url]
    if should_combine:
        valid_combination_id, msg, dataset_obj = check_valid_combination_id(combine_id)
        if valid_combination_id:
            assert isinstance(dataset_obj, Dataset)
            assert isinstance(dataset_obj.links, list)
            assert isinstance(dataset_obj.tags, list)
            links.extend(dataset_obj.links)
            links = list(set(links))
            tags.extend(dataset_obj.tags)
            tags = list(set(tags))
            s3.download_file(dataset_obj.s3Key, "yolo_old.zip")
        else:
            should_combine = False
            logger.warning(msg)
            sns.send(f"Converting {dtype} dataset | {dataset_name} | IMPORTANT", msg)

    dir_name = None
    names = None

    ### Create YOLO format of datasets
    if dtype == PreProcessorKeys.TYPE_ROBOFLOW:
        logger.info(f"Started download for roboflow dataset at {url}")
        dir_name = download_dataset_from_roboflow(url, PreProcessorKeys.ROBOFLOW_YOLOV11, PreProcessorKeys.ROBOFLOW_KEY)
        dir_name = Path(dir_name)
        logger.info(f"Finished download of {url} to {dir_name}")
        names = get_roboflow_classnames_and_fix(dir_name)
        convert_roboflow_yolo_to_bbox(dir_name)

    elif dtype == PreProcessorKeys.TYPE_VISDRONE:
        logger.info(f"Started download for dataset at {url}")
        wget.download(url=url, out="visdrone.zip", bar=None)
        with zipfile.ZipFile("visdrone.zip", "r") as zipf:
            dir_name = Path("./" + zipf.namelist()[0])
            logger.info(f"Unzipped dataset to {dir_name}")
            zipf.extractall()

        logger.info("Converting dataset to YOLO format")
        names = PreProcessorKeys.VISDRONE_CLASSES
        visdrone2yolo(dir_name, names)

    try:
        assert isinstance(dir_name, Path)
        assert isinstance(names, list)
    except AssertionError:
        logger.error("Something went wrong converting to YOLO")
        sns.send(
            f"Converting {dtype} dataset | {dataset_name} | IMPORTANT",
            "Something went wrong converting to YOLO, process aborted.",
        )
        return

    checksum_blob = []
    if should_combine:
        dir_name, checksum_blob, skipped = perform_combine(dataset_obj, dir_name, s3)
        logger.info(f"Finished combining datasets with {skipped} images skipped")
    else:
        for split in ["test", "train", "valid"]:
            for img_path in (dir_name / split / "images").glob("*"):
                img_checksum = hashlib.md5(img_path.read_bytes()).hexdigest()
                checksum_blob.append(img_checksum)

    upload_yolo, new_yolo_id, (checksum_key, yolo_key), msg = upload_dataset_and_make_entry(
        dataset_name, tags, links, dir_name, checksum_blob, DatasetKeys.YOLO_FORMAT
    )
    if not upload_yolo:
        logger.warning(msg)
        return
    yolo_to_coco(dir_name, names)
    upload_coco, new_coco_id, (_, coco_key), msg = upload_dataset_and_make_entry(
        dataset_name, tags, links, dir_name, checksum_blob, DatasetKeys.COCO_FORMAT
    )
    if not upload_coco:
        logger.warning(msg)
        return

    logger.info("Dataset conversions complete")
    sns.send(
        f"Converting {dtype} dataset",
        f"Converted dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\ndatasets location: {GeneralKeys.S3_BUCKET_NAME}/datasets/",
    )

    sns.send(
        f"Converting {dtype} dataset | {dataset_name} | complete",
        "\n".join(
            [
                f"Dataset: {dataset_name}",
                f"Converting {dtype} dataset | Complete",
                f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"uploaded yolo dataset id: {new_yolo_id}",
                f"uploaded yolo s3 key: {yolo_key}",
                f"uploaded coco dataset id: {new_coco_id}",
                f"uploaded coco s3 key: {coco_key}",
                f"uploaded checksum blob file s3 key: {checksum_key}",
                f"tags: {tags}\n\n",
                "Data dump:",
                json.dumps(data),
            ]
        ),
    )
    cleanup()
    return


def trigger_training(model, params, data):
    model_id = queries().get_model_by_key(model)
    if model_id == -1:
        return None

    ec2 = boto3.client("ec2", region_name="us-east-1")

    extra_commands = []
    if model in TrainerKeys.REQUIRE_YAML:
        if model in TrainerKeys.USE_YOLO:
            extra_commands.append(f"wget -O yolov8s-custom.yaml {data['yaml_utkey']}")
        elif model in TrainerKeys.USE_RTDETR:
            extra_commands.append(f"wget -O rtdetr-custom.yaml {data['yaml_utkey']}")

        extra_commands.append(f"echo 'YAML_URL={data['yaml_utkey']}' >> .extra")

    if "tags" in data.keys() and len(data["tags"]) > 0:
        for tag in data["tags"]:
            extra_commands.append(f'echo "{tag}" >> tags.txt')

    if "datasetId" in data.keys() and data["datasetId"] is not None:
        extra_commands.append(f"echo 'DATASET_ID={data['datasetId']}' >> .extra")

    extra_commands.append(f"echo 'MODEL_ID={model_id}' >> .extra")

    extra_commands = "\n".join(extra_commands)

    # fmt: off
    # Define User Data script
    script = []
    script.append("#!/bin/bash")

    # update and install required packages
    script.append("sudo apt update -y")
    script.append("sudo apt upgrade -y")
    script.append("sudo apt install python3-full python3-pip git amazon-ec2-utils libgl1 -y")
    script.append("wget https://amazoncloudwatch-agent.s3.amazonaws.com/debian/amd64/latest/amazon-cloudwatch-agent.deb")
    script.append("sudo dpkg -i -E ./amazon-cloudwatch-agent.deb")

    # configure and start cloudwatch agent
    script.append(f"""echo "{{\\"logs\\":{{\\"logs_collected\\":{{\\"files\\":{{\\"collect_list\\":[{{\\"file_path\\":\\"/home/ubuntu/trainer/sfdt_trainer.log\\",\\"log_group_name\\":\\"sfdt-log-group\\",\\"log_stream_name\\":\\"trainer/{model}/instance-$(ec2-metadata -i | awk '{{print $2}}')\\"}}]}}}}}}}}" | sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null""")
    script.append("sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s")

    # get code
    script.append(f"git clone https://ibrahimmkhalid:{GeneralKeys.GITHUB_ACCESS_TOKEN}@github.com/sjsu2024-data298-team6/monorepo /home/ubuntu/trainer")

    # setup environment
    script.append("cd /home/ubuntu/trainer")
    script.append(f'echo "DEPLOYMENT=prod\nS3_BUCKET_NAME={GeneralKeys.S3_BUCKET_NAME}\nSNS_ARN={GeneralKeys.SNS_ARN}\nMODEL_TO_TRAIN={model}\nRUNNER=train\nWANDB_KEY={GeneralKeys.WANDB_KEY}\nWANDB_ENTITY={GeneralKeys.WANDB_ENTITY}\nDB_URI={GeneralKeys.DB_URI}" >> .env')
    script.append(f"echo '{params}' >> params.json")
    script.append("python3 -m venv venv")
    script.append("source venv/bin/activate")

    script.append(f"{extra_commands}")

    # install python packages
    script.append("pip install git+https://github.com/sjsu2024-data298-team6/ultralytics.git")
    script.append("pip install -r requirements.txt")

    # run script
    script.append("python3 main.py >> sfdt_trainer.log 2>&1")
    script.append("if [ $? -ne 0 ]; then")
    script.append('    echo "Training script failed with exit code $?" >> sfdt_trainer.log')
    script.append("else")
    script.append('    echo "Training script completed successfully" >> sfdt_trainer.log')
    script.append("    sudo shutdown -h now")
    script.append("fi")

    script = "\n".join(script)
    # fmt: on

    # Launch EC2 instance
    response = ec2.run_instances(
        ImageId="ami-093fcc54e22f8fcd4",
        InstanceType="g5.2xlarge",
        InstanceInitiatedShutdownBehavior="terminate",
        KeyName="sjsu-fall24-data298-team6-key-pair",
        MinCount=1,
        MaxCount=1,
        UserData=script,
        IamInstanceProfile={"Arn": os.getenv("EC2_INSTANCE_IAM_ARN")},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "Encrypted": False,
                    "DeleteOnTermination": True,
                    "Iops": 3000,
                    "SnapshotId": "snap-0186871616ff60f9f",
                    "VolumeSize": 200,
                    "VolumeType": "gp3",
                    "Throughput": 125,
                },
            }
        ],
        NetworkInterfaces=[
            {
                "AssociatePublicIpAddress": True,
                "DeviceIndex": 0,
                "Groups": ["sg-0ae6a08ce3772678c"],
            },
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"sfdt-trainer-{model}"},
                    {"Key": "d298_task_type", "Value": "training"},
                ],
            },
        ],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    logger.info(f"Trainer EC2 instance launched: {instance_id}")
    sns.send(f"Training {model}", f"Trainer EC2 instance launched: {instance_id}")
    return instance_id


def check_instance_terminated(instance_id):
    ec2 = boto3.client("ec2", region_name="us-east-1")
    response = ec2.describe_instance_status(InstanceIds=[instance_id], IncludeAllInstances=True)
    return response["InstanceStatuses"][0]["InstanceState"]["Name"] == "terminated"


def cleanup():
    keywords = ["yolo", "coco", "visdrone", "checksum", "old"]
    for item in os.listdir("."):
        if any(keyword in item for keyword in keywords):
            print(item)
            path = os.path.join(".", item)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


def listen_to_sqs():
    sqs = boto3.client("sqs", region_name="us-east-1")
    instance_id = None
    _counter = 0
    while True:
        response = sqs.receive_message(QueueUrl=GeneralKeys.SQS_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=10)

        if "Messages" in response:
            message = response["Messages"][0]
            receipt_handle = message["ReceiptHandle"]
            body = json.loads(message["Body"])
            data = body["data"]
            task = body["task"]
            try:
                # Delete message early to avoid over run model training
                sqs.delete_message(QueueUrl=GeneralKeys.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                logger.info("Processed and deleted message from SQS.")
                ########################################################
                if task == "model":
                    model = data["model"]
                    params = data["params"]

                    if model not in TrainerKeys.SUPPORTED_MODELS:
                        logger.info(f"Model {model} not supported currently")
                        continue

                    instance_id = trigger_training(model, params, data)
                    if instance_id is None:
                        logger.info("Model ID not found. Aborting")
                        sns.send(
                            "Error | Model ID not found",
                            f"""Project: pipeline
                            request body: {body}
                            """,
                        )
                        continue

                    # make sure instance id is available on api
                    time.sleep(60)

                    while not check_instance_terminated(instance_id):
                        if _counter == 360:
                            _counter = 0
                            logger.info("Currently training...")
                        time.sleep(30)
                        _counter += 1
                    continue
                ########################################################
                elif task == "dataset":
                    process_and_upload_dataset(data)
                    continue
                ########################################################
                else:
                    logger.warning(f"Task type {task} not supported")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                sns.send(
                    "Error | reading request",
                    f"""Project: pipeline
                         Error: {e}
                         Traceback: {traceback.format_exc()}""",
                )
                sqs.delete_message(QueueUrl=GeneralKeys.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
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
    listen_to_sqs()
