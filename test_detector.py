#!/usr/bin/env python3

import cv2
import numpy as np
from src.detector import ParkingDetector

print("="*50)
print("Testing Detector")
print("="*50)

# Create test frame
test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
test_frame[:] = (100, 100, 100)  # Gray background

# Draw a fake car (white rectangle)
cv2.rectangle(test_frame, (300, 200), (400, 350), (255, 255, 255), -1)
cv2.rectangle(test_frame, (500, 200), (600, 350), (255, 255, 255), -1)

print("\n1. Creating detector...")
detector = ParkingDetector()
print("✓ Detector created")

print("\n2. Detecting vehicles...")
results = detector.detect_vehicles(test_frame)
print(f"✓ YOLO results: {results}")

print("\n3. Getting vehicle boxes...")
vehicles = detector.get_vehicle_bboxes(results)
print(f"✓ Found {len(vehicles)} vehicles: {vehicles}")

print("\n4. Drawing detections...")
output_frame, stats = detector.draw_detections(test_frame, vehicles)
print(f"✓ Stats: {stats}")

print("\n" + "="*50)
if len(vehicles) > 0 or stats['total'] > 0:
    print("✓✓✓ DETECTOR WORKING!")
else:
    print("⚠ No detections - model may need tuning")
print("="*50)

