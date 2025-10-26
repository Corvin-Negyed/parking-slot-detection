"""
Parking detection module using YOLOv8.
Detects white parking lines and determines occupancy.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon as ShapelyPolygon
from src.config import Config


class ParkingDetector:
    def __init__(self):
        """Initialize parking detector with YOLOv8 model"""
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)  # Red
        self.available_color = (0, 255, 0)  # Green
        self.parking_spots = []
        
    def detect_white_lines(self, frame):
        """Detect small white parking lines"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Strong threshold for white lines only
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Detect edges
        edges = cv2.Canny(binary, 100, 200)
        
        # Detect lines - focus on short white lines (parking dividers)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=20,
            maxLineGap=5
        )
        
        return lines if lines is not None else []
    
    def create_parking_spots(self, lines, frame_width, frame_height):
        """Create parking spots from white line pairs"""
        if len(lines) < 2:
            return []
        
        # Extract line segments
        segments = [line[0] for line in lines]
        
        # Calculate line properties
        line_data = []
        for x1, y1, x2, y2 in segments:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Keep short white lines (typical parking dividers)
            if 15 < length < 150:
                line_data.append({
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'center': (center_x, center_y),
                    'length': length
                })
        
        if len(line_data) < 2:
            return []
        
        # Find dominant angle (parking line orientation)
        angles = [ld['angle'] for ld in line_data]
        angle_hist, angle_bins = np.histogram(angles, bins=18)
        dominant_bin = np.argmax(angle_hist)
        dominant_angle = (angle_bins[dominant_bin] + angle_bins[dominant_bin + 1]) / 2
        
        # Filter lines with similar angle
        parallel_lines = []
        for ld in line_data:
            if abs(ld['angle'] - dominant_angle) < 15:
                parallel_lines.append(ld)
        
        if len(parallel_lines) < 2:
            return []
        
        # Sort by center position (perpendicular to line angle)
        # For angled lines, sort by projected position
        if abs(dominant_angle) < 45:  # More horizontal
            parallel_lines.sort(key=lambda ld: ld['center'][1])
        else:  # More vertical
            parallel_lines.sort(key=lambda ld: ld['center'][0])
        
        # Create parking spots between consecutive line pairs
        parking_spots = []
        for i in range(len(parallel_lines) - 1):
            line1 = parallel_lines[i]['coords']
            line2 = parallel_lines[i + 1]['coords']
            
            x1_a, y1_a, x2_a, y2_a = line1
            x1_b, y1_b, x2_b, y2_b = line2
            
            # Calculate distance between lines
            dist = np.sqrt((parallel_lines[i]['center'][0] - parallel_lines[i + 1]['center'][0])**2 +
                          (parallel_lines[i]['center'][1] - parallel_lines[i + 1]['center'][1])**2)
            
            # Typical parking spot width: 40-180 pixels
            if 40 < dist < 180:
                # Create polygon (quadrilateral) from the two lines
                polygon = [
                    (int(x1_a), int(y1_a)),
                    (int(x2_a), int(y2_a)),
                    (int(x2_b), int(y2_b)),
                    (int(x1_b), int(y1_b))
                ]
                
                # Verify polygon is within frame
                valid = all(0 <= x < frame_width and 0 <= y < frame_height 
                          for x, y in polygon)
                
                if valid:
                    parking_spots.append(polygon)
        
        return parking_spots
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv8"""
        results = self.model(frame, verbose=False)
        return results
    
    def get_vehicle_bboxes(self, results):
        """Extract stationary vehicle bounding boxes"""
        vehicle_boxes = []
        
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Cars, motorcycles, buses, trucks
                        if cls in [2, 3, 5, 7] and conf > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return vehicle_boxes
    
    def check_occupancy_iou(self, polygon, vehicle_bbox):
        """Check if vehicle overlaps with parking spot using IoU"""
        try:
            poly = ShapelyPolygon(polygon)
            x1, y1, x2, y2 = vehicle_bbox
            bbox_poly = ShapelyPolygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            
            intersection = poly.intersection(bbox_poly).area
            union = poly.union(bbox_poly).area
            
            iou = intersection / union if union > 0 else 0
            return iou >= Config.POLYGON_IOU_THRESHOLD
            
        except Exception:
            return False
    
    def draw_detections(self, frame, vehicle_boxes):
        """Draw parking spots and check occupancy"""
        h, w = frame.shape[:2]
        
        # Detect white lines and create parking spots
        if not self.parking_spots:
            lines = self.detect_white_lines(frame)
            spots = self.create_parking_spots(lines, w, h)
            
            if spots:
                self.parking_spots = spots
                print(f"Detected {len(spots)} parking spots from white lines")
        
        # If no spots detected, fallback to showing vehicles only
        if not self.parking_spots:
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.occupied_color, 2)
            
            return frame, {
                'total': len(vehicle_boxes),
                'occupied': len(vehicle_boxes),
                'available': 0
            }
        
        # Check each parking spot for occupancy
        total_spots = len(self.parking_spots)
        occupied_count = 0
        
        mask_occupied = np.zeros_like(frame)
        mask_available = np.zeros_like(frame)
        
        for polygon in self.parking_spots:
            is_occupied = False
            
            # Check if any vehicle overlaps with this spot
            for vehicle_bbox in vehicle_boxes:
                if self.check_occupancy_iou(polygon, vehicle_bbox):
                    is_occupied = True
                    break
            
            # Draw polygon
            poly_array = np.array(polygon, dtype=np.int32)
            
            if is_occupied:
                occupied_count += 1
                cv2.fillPoly(mask_occupied, [poly_array], self.occupied_color)
            else:
                cv2.fillPoly(mask_available, [poly_array], self.available_color)
        
        # Blend masks with original frame
        frame = cv2.addWeighted(mask_occupied, 0.3, frame, 1, 0)
        frame = cv2.addWeighted(mask_available, 0.3, frame, 1, 0)
        
        available_spots = total_spots - occupied_count
        
        stats = {
            'total': total_spots,
            'occupied': occupied_count,
            'available': available_spots
        }
        
        return frame, stats

