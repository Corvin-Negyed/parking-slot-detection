import cv2
from src.detector import ParkingDetector


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detector = ParkingDetector()
    
    def process_frame(self, frame):
        pass

