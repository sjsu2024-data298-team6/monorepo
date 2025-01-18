from PIL import Image
from pathlib import Path
from roboflow import Roboflow
import json
import os
import random
import shutil
import yaml


def convert_box_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    return (
        (box[0] + box[2] / 2) * dw,
        (box[1] + box[3] / 2) * dh,
        box[2] * dw,
        box[3] * dh,
    )


def visdrone2yolo(dir: Path, names):
    (dir / "labels").mkdir(parents=True, exist_ok=True)  # make labels directory
    for f in (dir / "annotations").glob("*.txt"):
        img_size = Image.open((dir / "images" / f.name).with_suffix(".jpg")).size
        lines = []
        with open(f, "r") as file:  # read annotation.txt
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] == "0":  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box_to_yolo(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(
                    str(f).replace(
                        f"{os.sep}annotations{os.sep}", f"{os.sep}labels{os.sep}"
                    ),
                    "w",
                ) as fl:
                    fl.writelines(lines)  # write label.txt

    splits = ["test", "train", "valid"]
    for split in splits:
        (dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dir / split / "labels").mkdir(parents=True, exist_ok=True)

    dataset_len = len(list((dir / "labels").glob("*.txt")))
    train_len = int(0.8 * dataset_len)
    valtest_len = dataset_len - train_len
    val_len = int(valtest_len / 2)
    test_len = valtest_len - val_len

    counts = {"test": 0, "train": 0, "valid": 0}
    max_counts = {"test": test_len, "train": train_len, "valid": val_len}
    for f in (dir / "labels").glob("*.txt"):
        i = (dir / "images" / f.name).with_suffix(".jpg")
        split = random.choice(splits)
        shutil.move(f, dir / split / "labels")
        shutil.move(i, dir / split / "images")
        counts[split] += 1
        if counts[split] == max_counts[split]:
            splits.remove(split)

    yaml_file = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(names),
        "names": names,
    }

    with open(dir / "data.yaml", "w") as file:
        yaml.dump(yaml_file, file)

    shutil.rmtree(dir / "annotations")
    shutil.rmtree(dir / "labels")
    shutil.rmtree(dir / "images")


def convert_mask_to_bbox(line):
    cls = float(line[0])
    xarr = []
    yarr = []
    for i, p in enumerate(line[1:]):
        if i % 2 == 0:
            xarr.append(float(p))
        else:
            yarr.append(float(p))
    xcenter = (max(xarr) + min(xarr)) / 2
    ycenter = (max(yarr) + min(yarr)) / 2
    bboxw = max(xarr) - min(xarr)
    bboxh = max(yarr) - min(yarr)
    return cls, xcenter, ycenter, bboxw, bboxh


def yolo_to_coco(image_dir, label_dir, output_path, categories):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)],
    }

    # Initialize annotation id
    ann_id = 0

    # Loop through all images
    for img_id, img_name in enumerate(os.listdir(image_dir)):
        if not img_name.endswith((".jpg", ".jpeg", ".png")):
            continue

        # Get image path
        img_path = os.path.join(image_dir, img_name)

        # Open image to get dimensions
        img = Image.open(img_path)
        width, height = img.size

        # Add image info to COCO format
        coco_format["images"].append(
            {"id": img_id, "file_name": img_name, "width": width, "height": height}
        )

        # Get corresponding label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        # Read YOLO annotations
        with open(label_path, "r") as f:
            label_lines = f.readlines()

        # Convert YOLO annotations to COCO format
        for line in label_lines:
            try:
                class_id, x_center, y_center, bbox_width, bbox_height = map(
                    float, line.strip().split()
                )
            except ValueError:
                class_id, x_center, y_center, bbox_width, bbox_height = (
                    convert_mask_to_bbox(line.strip().split())
                )

            # Convert YOLO coordinates to COCO coordinates
            x = (x_center - bbox_width / 2) * width
            y = (y_center - bbox_height / 2) * height
            w = bbox_width * width
            h = bbox_height * height

            # Add annotation to COCO format
            coco_format["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=2)


def download_dataset_from_roboflow(url, dl_format, key):
    parts = url.split("/")
    ds_version = parts[-1]
    ds_project = parts[-3]
    ds_workspace = parts[-4]
    rf = Roboflow(api_key=key)
    project = rf.workspace(ds_workspace).project(ds_project)
    version = project.version(ds_version)
    dataset = version.download(dl_format, location=f"./{dl_format}", overwrite=True)
    return Path(dataset.location)
