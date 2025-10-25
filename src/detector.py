from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)

