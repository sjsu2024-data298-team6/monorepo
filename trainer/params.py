import json

try:
    with open("params.json", "r") as f:
        content = f.read()
        raw_params = json.loads(content)
except FileNotFoundError:
    raw_params = {}


class yolo_params(object):
    def __init__(self):
        self.epochs = raw_params.get("epochs", 10)
        self.imgsz = raw_params.get("imgsz", 640)
        self.batch = raw_params.get("batch", 16)


class rtdetr_params(object):
    def __init__(self):
        self.epochs = raw_params.get("epochs", 10)
        self.imgsz = raw_params.get("imgsz", 640)
        self.batch = raw_params.get("batch", 8)
