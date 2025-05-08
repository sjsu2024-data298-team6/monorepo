import os
from dotenv import load_dotenv

load_dotenv()


class GeneralKeys:
    SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    SNS_ARN = os.getenv("SNS_ARN")
    DEPLOYMENT = os.getenv("DEPLOYMENT")
    GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
    RUNNER = os.getenv("RUNNER")
    WANDB_KEY = os.getenv("WANDB_KEY")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    DB_URI = os.getenv("DB_URI")


class PreProcessorKeys:
    ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")

    TYPE_ROBOFLOW = "roboflow"
    TYPE_ZIPFILE = "zipfile"

    TYPE_VISDRONE = "visdrone"
    VISDRONE_CLASSES = [
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
    ]

    SUPPORTED_TYPES = {
        TYPE_ROBOFLOW,
        TYPE_VISDRONE,
    }

    ROBOFLOW_YOLOV11 = "yolov11"
    ROBOFLOW_DETECTRON = "coco"

    ROBOFLOW_SUPPORTED_DATASETS = {
        ROBOFLOW_YOLOV11,
        ROBOFLOW_DETECTRON,
    }


class TrainerKeys:
    YOLO_CUSTOM = "custom_yolo"
    RTDETR_CUSTOM = "custom_rtdetr"
    YOLOV8_BASE = "yolov8_base"
    YOLOV11_BASE = "yolov11_base"
    YOLOV8_DCE = "yolov8_dce"
    YOLOV8_INVO = "yolov8_invo"
    YOLOV8_MHSA = "yolov8_mhsa"
    YOLOV8_MHSA_RTDETR = "yolov8_mhsa_rtdetr"
    YOLOV8_RESNET_SINGLE = "yolov8_resnet_single"
    YOLOV8_RESNET_DOUBLE = "yolov8_resnet_double"
    YOLOV8_RESNET_ALT = "yolov8_resnet_alt"
    YOLOV8_VIT_INVO = "yolov8_vit_invo"
    RTDETR_BASE = "rtdetr_base"

    SUPPORTED_MODELS = {
        YOLOV8_BASE,
        YOLOV11_BASE,
        YOLOV8_DCE,
        YOLOV8_MHSA,
        YOLOV8_MHSA_RTDETR,
        YOLOV8_RESNET_SINGLE,
        YOLOV8_RESNET_DOUBLE,
        RTDETR_BASE,
        YOLO_CUSTOM,
        RTDETR_CUSTOM,
    }

    # TODO: Update in the future if this changes, might require code changes as well
    ULTRALYTICS_TRAINER = SUPPORTED_MODELS.copy()
    REQUIRE_YAML = SUPPORTED_MODELS.copy()

    USE_RTDETR = {
        RTDETR_BASE,
        YOLOV8_MHSA_RTDETR,
        RTDETR_CUSTOM,
    }

    USE_YOLO = {
        YOLOV8_MHSA,
        YOLOV8_BASE,
        YOLOV11_BASE,
        YOLOV8_DCE,
        YOLOV8_INVO,
        YOLOV8_RESNET_SINGLE,
        YOLOV8_RESNET_DOUBLE,
        YOLOV8_RESNET_ALT,
        YOLOV8_VIT_INVO,
        YOLO_CUSTOM,
    }

    TFJS_SUPPORTED_YOLO_MODELS = {
        YOLOV8_BASE,
        YOLOV8_DCE,
        YOLOV8_INVO,
        YOLOV8_RESNET_SINGLE,
        YOLOV8_RESNET_DOUBLE,
        YOLOV8_RESNET_ALT,
        YOLOV8_VIT_INVO,
        YOLO_CUSTOM,
    }


class DatasetKeys:
    YOLO_FORMAT = "yolo"
    COCO_FORMAT = "coco"
