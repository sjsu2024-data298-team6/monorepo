import hashlib
import json
import logging
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path

import boto3
import wget
import yaml

from aws_handler import S3Handler, SNSHandler
from db import db_manager
from db.queries import queries
from db.writer import DatabaseWriter
from keys import DatasetKeys, GeneralKeys, PreProcessorKeys, TrainerKeys
from preprocessor.dataset import (
    convert_mask_to_bbox,
    download_dataset_from_roboflow,
    visdrone2yolo,
    yolo_to_coco,
)

logger = logging.getLogger("sfdt_preprocessor")
logging.basicConfig(filename="sfdt_preprocessor.log", encoding="utf-8", level=logging.INFO)

sns = SNSHandler(logger=logger)
s3 = S3Handler(bucket=GeneralKeys.S3_BUCKET_NAME, logger=logger)


def process_and_upload_dataset(data):
    dataset_name = data.get("name", None)
    url = data.get("url", None)
    dtype = data.get("datasetType", None)
    if dataset_name is None or url is None or dtype is None:
        logger.warning(f"Something went wrong trying to process {json.dumps(data)}")
        sns.send(
            "Something went wrong",
            f"Something went wrong trying process_and_upload_dataset with the following:\n\n{json.dumps(data)}",
        )
        return

    should_combine = data.get("shouldCombine", False)
    combine_id = data.get("combineID", None)
    if should_combine and combine_id is None:
        logger.warning(f"combine_id is required when should_combine is True {json.dumps(data)}")
        sns.send(
            "Something went wrong",
            f"process_and_upload_dataset tried to combine without combine id:\n\n{json.dumps(data)}",
        )
        return

    if dtype not in PreProcessorKeys.SUPPORTED_TYPES:
        logger.warning(f"{dtype} download type not supported")
        sns.send(
            "Something went wrong",
            f"{dtype} download type not supported",
        )
        return

    tags = data.get("tags", [dataset_name.lower().replace(" ", "-").strip()])

    sns.send(
        f"Converting {dtype} dataset | {dataset_name}",
        "\n".join(
            [
                f"Converting {dtype} dataset",
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
        logger.info("Finding dataset to combine with")
        dataset_obj = queries().get_dataset_by_id(combine_id)
        if dataset_obj is None:
            msg = f"Dataset at id {combine_id} not found"
            logger.info(msg)
            sns.send("Error", msg)
            return
        if dataset_obj.datasetType.value == "coco":
            # Assumption is that yolo will always be defined before coco
            # The rest of the code assumes a yolo format will be combined with before continuing
            logger.info("A coco dataset was provided to combine, attempting to find yolo format")
            combine_id -= 1
            dataset_obj_old = dataset_obj
            dataset_obj = queries().get_dataset_by_id(combine_id)
            if dataset_obj is None:
                msg = f"Dataset at id {combine_id} not found"
                logger.info(msg)
                sns.send("Error", msg)
                return
            if dataset_obj.datasetType.value != "yolo":
                logger.warning("Could not find yolo format for combining, creating new dataset without combination")
                sns.send(
                    f"Converting {dtype} dataset | {dataset_name} | IMPORTANT",
                    "\n".join(
                        [
                            "Could not find yolo format of provided dataset, cannot combine!",
                            "Instead, the dataset will be uploaded and converted on its own",
                            f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            "Tried combining with",
                            f"Database entry for ID {combine_id}:\n",
                            f"{dataset_obj_old}",
                            f"Database entry fix attempt ID {combine_id - 1}:\n",
                            f"{dataset_obj}",
                        ]
                    ),
                )
                should_combine = False
        if should_combine:
            assert isinstance(dataset_obj.links, list)
            assert isinstance(dataset_obj.tags, list)
            links.extend(dataset_obj.links)
            links = list(set(links))
            tags.extend(dataset_obj.tags)
            tags = list(set(tags))
            s3.download_file(dataset_obj.s3Key, "yolo_old.zip")

    dir_name = None

    ### Create YOLO format of datasets
    if dtype == PreProcessorKeys.TYPE_ROBOFLOW:
        logger.info(f"Started download for roboflow dataset at {url}")
        dir_name = download_dataset_from_roboflow(url, PreProcessorKeys.ROBOFLOW_YOLOV11, PreProcessorKeys.ROBOFLOW_KEY)
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
        logger.info(f"Started download for dataset at {url}")
        wget.download(url=url, out="visdrone.zip", bar=None)
        with zipfile.ZipFile("visdrone.zip", "r") as zipf:
            dir_name = Path("./" + zipf.namelist()[0])
            logger.info(f"Unzipped dataset to {dir_name}")
            zipf.extractall()

        logger.info("Converting dataset to YOLO format")
        names = PreProcessorKeys.VISDRONE_CLASSES
        visdrone2yolo(dir_name, names)

        os.remove("visdrone.zip")

    else:
        logger.info("Should not have gotten here")
        return

    try:
        assert isinstance(dir_name, Path)
        assert isinstance(names, list)
    except AssertionError:
        logger.error("Something went wrong converting to YOLO")
        return

    checksum_blob = []
    if should_combine:
        try:
            assert dataset_obj is not None
        except AssertionError:
            logger.error("Something went wrong when asserting dataset_obj during combining step")
            return

        if s3.check_file_exists(dataset_obj.checksumBlobS3Key, logger=logger):
            s3.download_file(dataset_obj.checksumBlobS3Key, "checksum_blob.txt")
            with open("checksum_blob.txt", "r") as fd:
                checksum_blob = fd.read().splitlines()
            os.remove("checksum_blob.txt")
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
                shutil.move(str(img_path), str(old_dir / split / "images" / f"{current_ts}_{img_path.name}"))
                shutil.move(str(txt_path), str(old_dir / split / "labels" / f"{current_ts}_{txt_path.name}"))

        shutil.rmtree(dir_name)
        os.remove("yolo_old.zip")
        dir_name = old_dir
        logger.info(f"Finished combining datasets with {skipped} images skipped")

    else:
        for split in ["test", "train", "valid"]:
            for img_path in (dir_name / split / "images").glob("*"):
                img_checksum = hashlib.md5(img_path.read_bytes()).hexdigest()
                checksum_blob.append(img_checksum)

    current_ts = str(int(time.time()))
    checksum_fn = f"checksum_{current_ts}.txt"
    zip_name = f"{DatasetKeys.YOLO_FORMAT}_{current_ts}.zip"

    with open(checksum_fn, "w") as fd:
        fd.write("\n".join(checksum_blob))

    session = next(db_manager.get_db())
    writer = DatabaseWriter(session)

    yolo_dataset_type = queries().get_dataset_type_by_value("yolo")
    if yolo_dataset_type is None:
        logger.warning("YOLO dataset type does not exist? Check database")
        return

    ### upload to s3 and make db entry
    s3.upload_file_to_s3(checksum_fn, "dataset", checksum_fn)
    s3.upload_zip_to_s3(dir_name, "dataset", zip_name=zip_name)
    assert isinstance(yolo_dataset_type.id, int)
    writer.create_dataset(
        datasetTypeId=yolo_dataset_type.id,
        name=dataset_name,
        tags=tags,
        links=links,
        s3Key=f"dataset/{zip_name}",
        checksumBlobS3Key=f"dataset/{checksum_fn}",
    )

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

    zip_name = f"{DatasetKeys.COCO_FORMAT}_{current_ts}.zip"

    coco_dataset_type = queries().get_dataset_type_by_value("coco")
    if coco_dataset_type is None:
        logger.warning("coco dataset type does not exist? Check database")
        return

    ### upload to s3 and make db entry
    s3.upload_file_to_s3(checksum_fn, "dataset", checksum_fn)
    s3.upload_zip_to_s3(dir_name, "dataset", zip_name=zip_name)
    assert isinstance(coco_dataset_type.id, int)
    writer.create_dataset(
        datasetTypeId=coco_dataset_type.id,
        name=dataset_name,
        tags=tags,
        links=links,
        s3Key=f"dataset/{zip_name}",
        checksumBlobS3Key=f"dataset/{checksum_fn}",
    )

    shutil.rmtree(dir_name)

    logger.info("Dataset conversions complete")
    sns.send(
        f"Converting {dtype} dataset",
        f"Converted dataset from {url}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\ndatasets location: {GeneralKeys.S3_BUCKET_NAME}/datasets/",
    )
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
