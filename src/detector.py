"""
Parking detection module using YOLOv8.
Detects vehicles and parking spot occupancy with visual feedback.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """
        Initialize parking detector with YOLOv8 model
        
        Args:
            parking_spots: List of parking spot coordinates (optional)
        """
        self.model = YOLO(Config.MODEL_PATH)
        self.parking_spots = parking_spots or []
        self.occupied_color = (0, 0, 255)  # Red for occupied
        self.available_color = (0, 255, 0)  # Green for available
        
    def detect_vehicles(self, frame):
        """
        Detect vehicles in the frame using YOLOv8
        
        Args:
            frame: Input video frame
            
        Returns:
            Detection results from YOLO
        """
        results = self.model(frame, verbose=False)
        return results
    
    def get_vehicle_bboxes(self, results):
        """
        Extract vehicle bounding boxes from YOLO results
        
        Args:
            results: YOLO detection results
            
        Returns:
            List of vehicle bounding boxes [(x1, y1, x2, y2), ...]
        """
        vehicle_boxes = []
        
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Filter for vehicles (car, truck, bus, etc.)
                        # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                        if cls in [2, 3, 5, 7] and conf > 0.3:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return vehicle_boxes
    
    def set_parking_spots(self, spots):
        """
        Set parking spot coordinates
        
        Args:
            spots: List of parking spot polygons or rectangles
        """
        self.parking_spots = spots
    
    def check_spot_occupancy(self, spot, vehicle_boxes):
        """
        Check if a parking spot is occupied by any vehicle
        
        Args:
            spot: Parking spot coordinates (polygon or rectangle)
            vehicle_boxes: List of vehicle bounding boxes
            
        Returns:
            True if occupied, False if available
        """
        if not vehicle_boxes:
            return False
        
        # Convert spot to rectangle if it's a polygon
        spot_rect = self._polygon_to_rect(spot)
        
        # Check intersection with each vehicle
        for veh_box in vehicle_boxes:
            if self._boxes_intersect(spot_rect, veh_box):
                return True
        
        return False
    
    def _polygon_to_rect(self, polygon):
        """Convert polygon to bounding rectangle"""
        if len(polygon) == 4 and isinstance(polygon[0], (int, float)):
            # Already a rectangle (x1, y1, x2, y2)
            return polygon
        else:
            # Polygon points - get bounding box
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            return (min(xs), min(ys), max(xs), max(ys))
    
    def _boxes_intersect(self, box1, box2, threshold=0.1):
        """
        Check if two bounding boxes intersect
        
        Args:
            box1, box2: Bounding boxes (x1, y1, x2, y2)
            threshold: Minimum intersection ratio to consider as occupied
            
        Returns:
            True if boxes intersect significantly
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate spot area
        spot_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        # Check if intersection is significant
        intersection_ratio = intersection_area / spot_area if spot_area > 0 else 0
        
        return intersection_ratio > threshold
    
    def draw_parking_spots(self, frame, vehicle_boxes):
        """
        Draw parking spots on frame with color coding
        
        Args:
            frame: Video frame to draw on
            vehicle_boxes: List of detected vehicle bounding boxes
            
        Returns:
            Frame with drawn parking spots and statistics
        """
        total_spots = len(self.parking_spots)
        occupied_spots = 0
        
        for i, spot in enumerate(self.parking_spots):
            is_occupied = self.check_spot_occupancy(spot, vehicle_boxes)
            
            if is_occupied:
                occupied_spots += 1
                color = self.occupied_color
            else:
                color = self.available_color
            
            # Draw spot
            if len(spot) == 4 and isinstance(spot[0], (int, float)):
                # Rectangle
                x1, y1, x2, y2 = [int(v) for v in spot]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                # Polygon
                pts = np.array(spot, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, color, 2)
            
            # Draw spot number
            center = self._get_spot_center(spot)
            cv2.putText(frame, str(i+1), center, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        available_spots = total_spots - occupied_spots
        
        stats = {
            'total': total_spots,
            'occupied': occupied_spots,
            'available': available_spots
        }
        
        return frame, stats
    
    def _get_spot_center(self, spot):
        """Get center point of a parking spot"""
        if len(spot) == 4 and isinstance(spot[0], (int, float)):
            # Rectangle
            x1, y1, x2, y2 = spot
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        else:
            # Polygon
            xs = [p[0] for p in spot]
            ys = [p[1] for p in spot]
            return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
    
    def generate_default_spots(self, frame_width, frame_height, rows=2, cols=10):
        """
        Generate default parking spot grid
        
        Args:
            frame_width: Frame width
            frame_height: Frame height
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            List of parking spot rectangles
        """
        spots = []
        
        # Calculate spot dimensions
        margin = 50
        spacing = 10
        
        spot_width = (frame_width - 2 * margin - (cols - 1) * spacing) // cols
        spot_height = (frame_height - 2 * margin - (rows - 1) * spacing) // rows
        
        for row in range(rows):
            for col in range(cols):
                x1 = margin + col * (spot_width + spacing)
                y1 = margin + row * (spot_height + spacing)
                x2 = x1 + spot_width
                y2 = y1 + spot_height
                
                spots.append((x1, y1, x2, y2))
        
        return spots
