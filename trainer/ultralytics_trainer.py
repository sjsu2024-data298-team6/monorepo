import logging
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import RTDETR, YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou

import trainer.params as tparams
import wandb
from aws_handler import SNSHandler
from keys import GeneralKeys, TrainerKeys

from trainer.trainer import TrainingResultClass, AllMetrics, ClassMetrcis

logger = None
wandb.login(key=GeneralKeys.WANDB_KEY)
sns = SNSHandler()


def train_main(logger_, model_, extra_keys_) -> TrainingResultClass:
    logger = logger_
    assert isinstance(logger, logging.Logger)

    project = "MSDA_Capstone_Project"
    params = None

    if model_ in TrainerKeys.USE_YOLO:
        params = tparams.yolo_params()
    elif model_ in TrainerKeys.USE_RTDETR:
        params = tparams.rtdetr_params()
    else:
        logger.warning("Something went wrong trying to check for model parameters")
        sns.send("Training Failed", "Something went wrong trying to check for model parameters")
        exit()

    logger.info(f"Params: {params}")
    run = wandb.init(project=project, tags=extra_keys_["tags"], entity=GeneralKeys.WANDB_ENTITY, config=params.__dict__)

    logger.info(f"Detailed logs at: {run.url}")
    sns.send(f"Training {model_} | {extra_keys_['MODEL_NAME']}", f"Detailed logs at: {run.url}")

    if model_ in TrainerKeys.USE_YOLO:
        model = YOLO("./yolov8s-custom.yaml")
    elif model_ in TrainerKeys.USE_RTDETR:
        model = RTDETR("./rtdetr-custom.yaml")
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
    metrics_str, metrics = get_inference(model, f"{cwd}/data/test", runs_dir)
    wandb.finish()

    with open(runs_dir / "summary_results.txt", "w") as f:
        f.write(metrics_str)

    best_wt = runs_dir / "train/weights/best.pt"
    tfjs_path = None
    if model_ in TrainerKeys.TFJS_SUPPORTED_YOLO_MODELS:
        model = YOLO(best_wt)
        logger.info("Starting model conversion to tfjs format")
        try:
            model.export(format="tfjs", half=True)
            tfjs_path = runs_dir / "/train/weights/best_web_model"
        except Exception as e:
            logger.info(f"Failed to convert model to tfjs format: {e}")
    else:
        logger.info(f"{model_} does not support conversion to tfjs format")

    return TrainingResultClass(
        runs_dir=runs_dir,
        params=params.__dict__,
        wandb_logs=run.url,
        best_wt=best_wt,
        tfjs_path=tfjs_path,
        metrics_str=metrics_str,
        metrics=metrics,
    )


def get_inference(model, test_base, runs_dir) -> Tuple[str, AllMetrics | None]:
    label_path = Path(test_base) / "labels"
    test_path = Path(test_base) / "images"

    pred = model.predict(source=test_path, save=True, save_txt=True)
    names = model.names

    metrics = DetMetrics(save_dir=runs_dir, plot=True, names=names)

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []

    num_per_class = {name: 0 for name in names.values()}
    avg_iou_per_class = {name: 0.0 for name in names.values()}

    for result in pred:
        img = plt.imread(result.path)
        img_height, img_width = img.shape[:2]

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_scores = result.boxes.conf.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy()

        img_name = Path(result.path).stem
        gt_path = label_path / f"{img_name}.txt"

        if gt_path.exists():
            with open(gt_path, "r") as f:
                gt_data = np.array([line.strip().split() for line in f.readlines()], dtype=np.float32)
                if len(gt_data):
                    gt_boxes = gt_data[:, 1:5]
                    gt_boxes[:, [0, 2]] *= img_width
                    gt_boxes[:, [1, 3]] *= img_height
                    gt_boxes = np.concatenate(
                        (gt_boxes[:, [0, 1]] - gt_boxes[:, [2, 3]] / 2, gt_boxes[:, [0, 1]] + gt_boxes[:, [2, 3]] / 2),
                        axis=1,
                    )
                    gt_classes = gt_data[:, 0]
                else:
                    gt_boxes = np.zeros((0, 4))
                    gt_classes = np.zeros(0)
        else:
            gt_boxes = np.zeros((0, 4))
            gt_classes = np.zeros(0)

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = box_iou(torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)).numpy()
        else:
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

        for i, _ in enumerate(pred_boxes):
            class_name = names[int(pred_classes[i])]
            num_per_class[class_name] += 1

            if len(gt_boxes) > 0:
                best_iou = iou_matrix[i].max()
                avg_iou_per_class[class_name] += best_iou

        iou_thresholds = np.linspace(0.5, 0.95, 10)
        tp = np.zeros((len(pred_boxes), len(iou_thresholds)))

        for pred_idx in range(len(pred_boxes)):
            pred_class = pred_classes[pred_idx]
            matching_class_mask = gt_classes == pred_class

            if not matching_class_mask.any():
                continue

            ious = iou_matrix[pred_idx][matching_class_mask]
            if len(ious) == 0:
                continue

            for iou_idx, iou_thresh in enumerate(iou_thresholds):
                tp[pred_idx, iou_idx] = ious.max() >= iou_thresh

        all_tp.append(tp)
        all_conf.append(pred_scores)
        all_pred_cls.append(pred_classes)
        all_target_cls.append(gt_classes)

    if not all_tp:
        return "No predictions found in the test set", None

    tp = np.concatenate(all_tp, axis=0)
    conf = np.concatenate(all_conf, axis=0)
    pred_cls = np.concatenate(all_pred_cls, axis=0)
    target_cls = np.concatenate(all_target_cls, axis=0)

    metrics.process(tp, conf, pred_cls, target_cls)

    results_str = format_results_str(metrics, names, num_per_class, avg_iou_per_class, pred)
    results = format_results(metrics, names, num_per_class, avg_iou_per_class, pred)

    return results_str, results


def format_results(metrics, names, num_per_class, avg_iou_per_class, pred) -> AllMetrics:
    inference_time = sum(result.speed["inference"] for result in pred)

    try:
        overall_iou = sum(avg_iou_per_class.values()) / sum(num_per_class.values())
    except ZeroDivisionError:
        overall_iou = None

    class_metrics_list = []
    ap_class_index = metrics.box.ap_class_index
    for i, class_idx in enumerate(ap_class_index):
        class_name = names[class_idx]
        try:
            class_iou = avg_iou_per_class[class_name] / num_per_class[class_name]
        except ZeroDivisionError:
            class_iou = None

        class_metrics_list.append(
            ClassMetrcis(
                name=class_name,
                precision=metrics.box.p[i],
                recall=metrics.box.r[i],
                map50=metrics.box.ap50[i],
                map5095=metrics.box.ap[i],
                iou=class_iou,
            )
        )

    return AllMetrics(
        precision=metrics.box.mp,
        recall=metrics.box.mr,
        map50=metrics.box.map50,
        map5095=metrics.box.map,
        iou=overall_iou,
        inference_time=inference_time / len(pred) if len(pred) > 0 else 0,
        class_metrics=class_metrics_list,
    )


def format_results_str(metrics, names, num_per_class, avg_iou_per_class, pred) -> str:
    inference_time = sum(result.speed["inference"] for result in pred)
    results = []

    results.append("Overall Metrics:")
    results.append(f"Precision: {metrics.box.mp:.4f}")
    results.append(f"Recall: {metrics.box.mr:.4f}")
    results.append(f"mAP@0.5: {metrics.box.map50:.4f}")
    results.append(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    try:
        results.append(f"Average IoU: {sum(avg_iou_per_class.values()) / sum(num_per_class.values()):.4f}")
    except ZeroDivisionError:
        results.append("Average IoU: N/A")
    results.append(f"Average inference time: {inference_time / len(pred):.2f} ms")

    results.append("")
    results.append("Per Class Metrics:")
    ap_class_index = metrics.box.ap_class_index

    max_class_name_len = max(len(names[class_idx]) for class_idx in ap_class_index) + 1

    for i, class_idx in enumerate(ap_class_index):
        class_name = names[class_idx]
        try:
            avg_iou = avg_iou_per_class[class_name] / num_per_class[class_name]
            avg_iou_str = f"{avg_iou:.4f}"
        except ZeroDivisionError:
            avg_iou_str = "N/A"

        padded_class_name = class_name.ljust(max_class_name_len)

        results.append(
            " | ".join(
                [
                    f"{padded_class_name}",
                    f"Precision: {metrics.box.p[i]:.4f}",
                    f"Recall: {metrics.box.r[i]:.4f}",
                    f"mAP@0.5: {metrics.box.ap50[i]:.4f}",
                    f"mAP@0.5:0.95: {metrics.box.ap[i]:.4f}",
                    f"IoU: {avg_iou_str}",
                ]
            )
        )

    return "\n".join(results)
