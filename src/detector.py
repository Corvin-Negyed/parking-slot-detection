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
        self.learned_spots = {}  # Learn spots from detections
        self.learning_phase = True
        self.learning_frames = 0
        
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
        
        # Learning phase: collect all vehicle positions from first frames
        if self.learning_phase and self.learning_frames < 30:
            self.learning_frames += 1
            
            # Collect vehicle positions
            for x1, y1, x2, y2 in vehicle_boxes:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check if this is a new parking spot location
                found = False
                for spot_id, spot_data in self.learned_spots.items():
                    spot_cx, spot_cy = spot_data['center']
                    dist = np.sqrt((center_x - spot_cx)**2 + (center_y - spot_cy)**2)
                    
                    if dist < 60:  # Same parking spot
                        found = True
                        # Keep largest bounding box seen
                        spot_data['x1'] = min(spot_data['x1'], x1)
                        spot_data['y1'] = min(spot_data['y1'], y1)
                        spot_data['x2'] = max(spot_data['x2'], x2)
                        spot_data['y2'] = max(spot_data['y2'], y2)
                        spot_data['center'] = ((spot_data['x1'] + spot_data['x2']) // 2,
                                              (spot_data['y1'] + spot_data['y2']) // 2)
                        spot_data['seen_count'] += 1
                        break
                
                if not found:
                    # Register new parking spot
                    spot_id = len(self.learned_spots)
                    self.learned_spots[spot_id] = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'center': (center_x, center_y),
                        'seen_count': 1
                    }
            
            # Show learning
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            cv2.putText(frame, f"Learning parking spots... {self.learning_frames}/30", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Found {len(self.learned_spots)} spots so far", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame, {
                'total': len(self.learned_spots),
                'occupied': len(vehicle_boxes),
                'available': max(0, len(self.learned_spots) - len(vehicle_boxes))
            }
        
        # Finish learning
        if self.learning_phase:
            self.learning_phase = False
            # Keep only spots seen multiple times (reduce noise)
            reliable_spots = {k: v for k, v in self.learned_spots.items() if v['seen_count'] >= 2}
            self.learned_spots = reliable_spots
            print(f"✓ Learned {len(self.learned_spots)} reliable parking spots")
        
        # Use learned spots
        if not self.learned_spots:
            # No spots learned, show vehicles
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.occupied_color, 2)
            
            return frame, {
                'total': len(vehicle_boxes),
                'occupied': len(vehicle_boxes),
                'available': 0
            }
        
        # Check occupancy
        mask_occupied = np.zeros_like(frame)
        mask_available = np.zeros_like(frame)
        
        occupied_count = 0
        
        for spot_id, spot_data in self.learned_spots.items():
            is_occupied = False
            
            # Create parking spot polygon
            spot_polygon = [
                (spot_data['x1'], spot_data['y1']),
                (spot_data['x1'], spot_data['y2']),
                (spot_data['x2'], spot_data['y2']),
                (spot_data['x2'], spot_data['y1'])
            ]
            
            # Check if any vehicle overlaps
            for vx1, vy1, vx2, vy2 in vehicle_boxes:
                # Simple overlap check
                if not (vx2 < spot_data['x1'] or vx1 > spot_data['x2'] or
                       vy2 < spot_data['y1'] or vy1 > spot_data['y2']):
                    is_occupied = True
                    break
            
            poly_arr = np.array(spot_polygon, np.int32)
            
            if is_occupied:
                occupied_count += 1
                cv2.fillPoly(mask_occupied, [poly_arr], self.occupied_color)
            else:
                cv2.fillPoly(mask_available, [poly_arr], self.available_color)
        
        # Blend
        frame = cv2.addWeighted(mask_occupied, 0.25, frame, 1, 0)
        frame = cv2.addWeighted(mask_available, 0.25, frame, 1, 0)
        
        stats = {
            'total': len(self.learned_spots),
            'occupied': occupied_count,
            'available': len(self.learned_spots) - occupied_count
        }
        
        return frame, stats

