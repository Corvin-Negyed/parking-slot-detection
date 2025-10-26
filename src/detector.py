"""
Simple parking detection
"""

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """Initialize"""
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)
        self.available_color = (0, 255, 0)
        self.polygon_data = self.load_polygons()
        
    def load_polygons(self):
        """Load parking polygons"""
        try:
            with open("object/poligon.obj", "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} parking spots")
            return data
        except:
            print("No polygons - will use detected vehicles")
            return []
    
    def find_polygon_center(self, polygon):
        """Find polygon center"""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
    
    def is_point_in_polygon(self, point, polygon):
        """Point in polygon check"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def detect_vehicles(self, frame):
        """Detect all objects"""
        return self.model(frame, conf=0.2, verbose=False)
    
    def get_vehicle_bboxes(self, results):
        """Get ALL detections first"""
        all_boxes = []
        
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"Found {len(result.boxes)} total detections")
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        all_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                        
                        if cls in [2, 3, 5, 7]:  # vehicle classes
                            print(f"Vehicle detected: cls={cls}, conf={conf:.2f}")
        
        return all_boxes
    
    def draw_detections(self, frame, vehicle_boxes):
        """Draw parking areas"""
        
        if not self.polygon_data:
            # No polygons: show all detected objects
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            return frame, {
                'total': len(vehicle_boxes),
                'occupied': len(vehicle_boxes),
                'available': 0
            }
        
        # Use polygons
        mask_occupied = np.zeros_like(frame)
        mask_available = np.zeros_like(frame)
        
        occupied_count = 0
        total_spots = len(self.polygon_data)
        
        polygon_copy = self.polygon_data.copy()
        
        # Check each vehicle against each polygon
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            car_polygon = [(vx1, vy1), (vx1, vy2), (vx2, vy2), (vx2, vy1)]
            
            for park_poly in self.polygon_data:
                if park_poly in polygon_copy:
                    poly_center = self.find_polygon_center(park_poly)
                    
                    if self.is_point_in_polygon(poly_center, car_polygon):
                        # DOLU
                        cv2.fillPoly(mask_occupied, [np.array(park_poly)], self.occupied_color)
                        polygon_copy.remove(park_poly)
                        occupied_count += 1
        
        # Remaining are BOŞ
        for park_poly in polygon_copy:
            cv2.fillPoly(mask_available, [np.array(park_poly)], self.available_color)
        
        # Blend
        frame = cv2.addWeighted(mask_occupied, 0.2, frame, 1, 0)
        frame = cv2.addWeighted(mask_available, 0.2, frame, 1, 0)
        
        stats = {
            'total': total_spots,
            'occupied': occupied_count,
            'available': len(polygon_copy)
        }
        
        print(f"PARK: Total={total_spots}, Occupied={occupied_count}, Available={len(polygon_copy)}")
        
        return frame, stats
