import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from pathlib import Path
from trainer.params import yolo_params


def train_main() -> str:
    model_params = yolo_params()
    cwd = Path(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolo11n.pt")
    model.train(
        data=cwd / "data/data.yaml",
        epochs=model_params.epochs,
        imgsz=model_params.imgsz,
        batch=model_params.batch,
        device=device,
    )

    with open(cwd / "data/data.yaml", "r") as file:
        yaml_content = yaml.safe_load(file)
    test_base = yaml_content["test"].split(".")[-1][1:]
    test_base = cwd / "data" / test_base

    return get_inference(model, f"{cwd}/data/test")


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


def get_inference(model, test_base) -> str:
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
        results.append(
            f"Average IoU: {sum(avg_iou_per_class.values())/sum(num_per_class.values())}"
        )
    except ZeroDivisionError:
        results.append("Average IoU: -1")

    inference = 0
    for idx, result in enumerate(pred):
        inference += result.speed["inference"]

    results.append(f"Average inference time: {inference/len(pred)}")
    results = "\n".join(results)

    with open("./runs/summary_results.txt", "w") as f:
        f.write(results)
    return results


if __name__ == "__main__":
    train_main()
