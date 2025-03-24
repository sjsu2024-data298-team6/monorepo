import os


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
    MODEL_YOLO = "yolo"
    MODEL_RTDETR = "rtdetr"
    MODEL_YOLO_CUSTOM = "custom_yolo"

    SUPPORTED_MODELS = {
        MODEL_YOLO,
        MODEL_RTDETR,
        MODEL_YOLO_CUSTOM,
    }


class DatasetKeys:
    YOLO_FORMAT = "yolo"
    COCO_FORMAT = "coco"
