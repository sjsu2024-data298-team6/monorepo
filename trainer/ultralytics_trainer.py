from typing import Dict, Tuple
import torch
import yaml
from ultralytics import YOLO, RTDETR
import matplotlib.pyplot as plt
import os
from pathlib import Path
from trainer.params import *
import logging
import wandb
from aws_handler import SNSHandler
from keys import GeneralKeys, TrainerKeys

logger = None
wandb.login(key=GeneralKeys.WANDB_KEY)
sns = SNSHandler()

params_ = {
    TrainerKeys.MODEL_YOLO: yolo_params,
    TrainerKeys.MODEL_YOLO_CUSTOM: yolo_params,
    TrainerKeys.MODEL_RTDETR: rtdetr_params,
    TrainerKeys.MODEL_RTDETR_CUSTOM: rtdetr_params,
}


def train_main(logger_, model_, extra_keys_) -> Tuple[str, Path, Dict]:
    logger = logger_
    assert isinstance(logger, logging.Logger)

    project = "MSDA_Capstone_Project"
    params = params_[model_]()
    logger.info(f"Params: {params}")
    run = wandb.init(project=project, tags=extra_keys_["tags"], entity=GeneralKeys.WANDB_ENTITY, config=params.__dict__)

    logger.info(f"Detailed logs at: {run.url}")
    sns.send(f"Training {model_}", f"Detailed logs at: {run.url}")

    if model_ == TrainerKeys.MODEL_YOLO:
        model = YOLO("yolo11s.yaml")
    elif model_ == TrainerKeys.MODEL_RTDETR:
        model = RTDETR("rtdetr-l.yaml")
    elif model_ == TrainerKeys.MODEL_YOLO_CUSTOM:
        model = YOLO("./yolov8s-custom.yaml")
    elif model_ == TrainerKeys.MODEL_RTDETR_CUSTOM:
        model = RTDETR("./yolov8s-custom.yaml")
    else:
        raise Exception(f"Unsupported ultralytics model: {model_}")

    cwd = Path(os.getcwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.watch(model, log_freq=100, log_graph=True)
    logger.info("Loaded baseline model")

    model.train(
        data=cwd / "data/data.yaml",
        epochs=params.epochs,
        imgsz=params.imgsz,
        batch=params.batch,
        device=device,
        project=project,
    )
    logger.info("finished training")

    runs_dir = Path(project)
    with open(cwd / "data/data.yaml", "r") as file:
        yaml_content = yaml.safe_load(file)
    test_base = yaml_content["test"].split(".")[-1][1:]
    test_base = cwd / "data" / test_base

    logger.info("Started inference")
    inference_info = get_inference(model, f"{cwd}/data/test", runs_dir)
    wandb.finish()

    # hacky solution, fix later
    content = None

    for encoding in ["utf-8", "latin-1", "ascii"]:
        try:
            with open("./wandb/latest-run/files/output.log", "r", encoding=encoding) as fd:
                content = fd.readlines()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        logger.error("Failed to read wandb output log with any encoding")
        content = []

    idx = 0
    for i, line in enumerate(content):
        if "mAP50  mAP50-95" in line:
            idx = i + 1
    mAPScores = content[idx].split()

    best_wt = runs_dir / "train/weights/best.pt"
    tfjs_path = ""
    if model_ in [TrainerKeys.MODEL_YOLO, TrainerKeys.MODEL_YOLO_CUSTOM]:
        model = YOLO(best_wt)
        logger.info("Starting model conversion to tfjs format")
        try:
            model.export(format="tfjs", half=True)
            tfjs_path = f"{project}/train/weights/best_web_model"
        except Exception as e:
            logger.info(f"Failed to convert model to tfjs format: {e}")

    ### RT-DETR is not supported by TFJS
    # else:  # model_ in [TrainerKeys.MODEL_RTDETR, TrainerKeys.MODEL_RTDETR_CUSTOM]:
    #     model = RTDETR(best_wt)
    ###

    return (
        inference_info,
        runs_dir,
        {
            "params": params.__dict__,
            "extras": {"wandb_logs": run.url},
            "best_wt": best_wt,
            "tfjs_path": tfjs_path,
            "map50": mAPScores[-2],
            "map5095": mAPScores[-1],
        },
    )


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def get_inference(model, test_base, runs_dir) -> str:
    label_path = f"{test_base}/labels"
    test_path = f"{test_base}/images"

    pred = model.predict(source=test_path, save=True)
    names = model.names

    num_per_class = {name: 0 for _, name in names.items()}
    avg_iou_per_class = {name: 0 for _, name in names.items()}

    for idx, result in enumerate(pred):
        gt_boxes = []
        image_name = os.path.basename(result.path)
        img = plt.imread(result.path)  # Read the image to get its dimensions
        img_height, img_width = img.shape[:2]

        gt_label_path = os.path.join(label_path, image_name.replace(".jpg", ".txt"))

        with open(gt_label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                _, x_center, y_center, width, height = map(float, parts)
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height

                gt_boxes.append([x_min, y_min, x_max, y_max])

        for r in result:
            boxes = r.boxes
            for box in boxes:
                c = box.cls
                pred_box = box.xyxy[0].tolist()
                num_per_class[names[int(c)]] += 1
                best_iou = 0
                for gt_box in gt_boxes:
                    curr_iou = iou(pred_box, gt_box)
                    best_iou = max(best_iou, curr_iou)
                avg_iou_per_class[names[int(c)]] += best_iou

    results = []
    for key, value in avg_iou_per_class.items():
        try:
            avg_iou = value / num_per_class[key]
        except ZeroDivisionError:
            avg_iou = -1

        results.append(f"{key} iou: {avg_iou}")

    try:
        results.append(f"Average IoU: {sum(avg_iou_per_class.values()) / sum(num_per_class.values())}")
    except ZeroDivisionError:
        results.append("Average IoU: -1")

    inference = 0
    for idx, result in enumerate(pred):
        inference += result.speed["inference"]

    results.append(f"Average inference time: {inference / len(pred)}")
    results = "\n".join(results)

    with open(runs_dir / "summary_results.txt", "w") as f:
        f.write(results)
    return results
