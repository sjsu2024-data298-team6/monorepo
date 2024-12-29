import json

with open("params.json", "r") as f:
    raw_params = json.loads(f.read())


class yolo_params:
    def __init__(self):
        self.epochs = raw_params.get("epochs", 10)
        self.imgsz = raw_params.get("imgsz", 640)
        self.batch = raw_params.get("batch", 16)


class rtdetr_params:
    def __init__(self):
        self.epochs = raw_params.get("epochs", 10)
        self.imgsz = raw_params.get("imgsz", 640)
        self.batch = raw_params.get("batch", 8)
