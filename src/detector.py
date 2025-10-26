"""
Parking detection using polygons and YOLOv8
"""

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """Initialize detector"""
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)  # Red
        self.available_color = (0, 255, 255)  # Yellow-green
        self.polygon_data = []
        self.load_polygons()
        
    def load_polygons(self):
        """Load parking spot polygons from object/poligon.obj"""
        try:
            with open("object/poligon.obj", "rb") as f:
                self.polygon_data = pickle.load(f)
            print(f"Loaded {len(self.polygon_data)} parking polygons")
        except:
            self.polygon_data = []
            print("No polygon file found")
    
    def find_polygon_center(self, points):
        """Find center of polygon"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        center_x = int(sum(x_coords) / len(points))
        center_y = int(sum(y_coords) / len(points))
        return (center_x, center_y)
    
    def is_point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        x, y = point
        poly_points = [(px, py) for px, py in polygon]
        n = len(poly_points)
        inside = False
        
        p1x, p1y = poly_points[0]
        for i in range(n + 1):
            p2x, p2y = poly_points[i % n]
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
        """Detect vehicles using YOLO"""
        results = self.model(frame, verbose=False)
        return results
    
    def get_vehicle_bboxes(self, results):
        """Extract vehicle bounding boxes"""
        vehicle_boxes = []
        
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Filter vehicles
                        if cls in [2, 3, 5, 7] and conf > 0.4:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return vehicle_boxes
    
    def draw_detections(self, frame, vehicle_boxes):
        """Draw parking spots and check occupancy"""
        
        # If no polygons loaded, show vehicles only
        if not self.polygon_data:
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.occupied_color, 2)
            
            return frame, {
                'total': len(vehicle_boxes),
                'occupied': len(vehicle_boxes),
                'available': 0
            }
        
        # Create masks for occupied and available spots
        mask_occupied = np.zeros_like(frame)
        mask_available = np.zeros_like(frame)
        
        # Make a copy to track which polygons are free
        polygon_data_copy = self.polygon_data.copy()
        
        # Check each detected vehicle
        for detection_bbox in vehicle_boxes:
            x1, y1, x2, y2 = detection_bbox
            
            # Create car polygon (bounding box as polygon)
            car_polygon = [
                (int(x1), int(y1)), 
                (int(x1), int(y2)), 
                (int(x2), int(y2)), 
                (int(x2), int(y1))
            ]
            
            # Check each parking polygon
            for parking_polygon in self.polygon_data:
                if parking_polygon in polygon_data_copy:
                    # Find center of parking polygon
                    polygon_center = self.find_polygon_center(parking_polygon)
                    
                    # Check if polygon center is inside car bounding box
                    is_present = self.is_point_in_polygon(polygon_center, car_polygon)
                    
                    if is_present:
                        # Mark as occupied (red)
                        cv2.fillPoly(mask_occupied, [np.array(parking_polygon)], self.occupied_color)
                        polygon_data_copy.remove(parking_polygon)
        
        # Draw remaining polygons as available (yellow-green)
        for parking_polygon in polygon_data_copy:
            cv2.fillPoly(mask_available, [np.array(parking_polygon)], self.available_color)
        
        # Blend masks with frame
        frame = cv2.addWeighted(mask_occupied, 0.2, frame, 1, 0)
        frame = cv2.addWeighted(mask_available, 0.2, frame, 1, 0)
        
        # Calculate statistics
        total_spots = len(self.polygon_data)
        occupied_spots = total_spots - len(polygon_data_copy)
        available_spots = len(polygon_data_copy)
        
        stats = {
            'total': total_spots,
            'occupied': occupied_spots,
            'available': available_spots
        }
        
        return frame, stats

