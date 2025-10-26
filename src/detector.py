"""
Parking detection module using YOLOv8.
Detects vehicles and parking spot occupancy with visual feedback.
"""

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from shapely.geometry import Polygon
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
        self.lines_detected = False  # Track if parking lines detected
        self.polygon_spots = []  # Store polygon-based parking spots
        
        # Try to load predefined polygons
        self.load_polygon_spots()
    
    def load_polygon_spots(self, obj_path='object/poligon.obj'):
        """Load parking spot polygons from pickle file"""
        try:
            with open(obj_path, 'rb') as f:
                self.polygon_spots = pickle.load(f)
            
            if self.polygon_spots and len(self.polygon_spots) > 0:
                print(f"Loaded {len(self.polygon_spots)} parking spots from {obj_path}")
                self.lines_detected = True
        except Exception as e:
            print(f"No polygon file found: {e}")
            self.polygon_spots = []
    
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
        
    def detect_parking_lines(self, frame):
        """
        Detect parking lines (white stripes) in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get white lines
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection
        edges = cv2.Canny(thresh, 50, 150)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=40, maxLineGap=20)
        
        return lines if lines is not None else []
    
    def create_spots_from_lines(self, lines, frame_width, frame_height):
        """
        Create parking spots from detected lines
        Two parallel lines define a parking spot
        
        Args:
            lines: Detected lines from HoughLinesP
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            List of parking spot polygons
        """
        if len(lines) < 2:
            return []
        
        parking_spots = []
        
        # Group lines by orientation (vertical vs horizontal)
        vertical_lines = []
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Vertical lines (parking dividers)
            if 70 < angle < 110:  # Nearly vertical
                vertical_lines.append((x1, y1, x2, y2))
            # Horizontal lines (parking row dividers)
            elif angle < 20 or angle > 160:  # Nearly horizontal
                horizontal_lines.append((x1, y1, x2, y2))
        
        # Sort vertical lines by x coordinate
        vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
        
        # Create spots between consecutive vertical lines
        for i in range(len(vertical_lines) - 1):
            x1_1, y1_1, x2_1, y2_1 = vertical_lines[i]
            x1_2, y1_2, x2_2, y2_2 = vertical_lines[i + 1]
            
            # Average x position for each line
            x_left = int((x1_1 + x2_1) / 2)
            x_right = int((x1_2 + x2_2) / 2)
            
            # Check if lines are close enough to be parking spot dividers (20-200 pixels apart)
            if 20 < (x_right - x_left) < 200:
                # Average y positions
                y_top = int(min(y1_1, y2_1, y1_2, y2_2))
                y_bottom = int(max(y1_1, y2_1, y1_2, y2_2))
                
                # Create parking spot rectangle
                parking_spots.append((x_left, y_top, x_right, y_bottom))
        
        return parking_spots
    
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
        Extract parked vehicle bounding boxes from YOLO results
        Filter out moving vehicles, keep only stationary ones
        
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
                        
                        # Filter for vehicles (car, truck, bus, motorcycle)
                        # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                        # Higher confidence threshold to avoid false positives
                        if cls in [2, 3, 5, 7] and conf > 0.4:
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
    
    def draw_detections(self, frame, vehicle_boxes):
        """
        Detect parking spots from lines and check vehicle occupancy
        
        Args:
            frame: Video frame to draw on
            vehicle_boxes: List of detected vehicle bounding boxes
            
        Returns:
            Frame with drawn parking spots and statistics
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Detect parking lines and create spots (only once or periodically)
        if not self.lines_detected or not self.parking_spots:
            lines = self.detect_parking_lines(frame)
            detected_spots = self.create_spots_from_lines(lines, frame_width, frame_height)
            
            if detected_spots and len(detected_spots) > 0:
                self.parking_spots = detected_spots
                self.lines_detected = True
                print(f"Detected {len(self.parking_spots)} parking spots from lines")
        
        # If no spots detected from lines, use fallback
        if not self.parking_spots:
            # Fallback: Use detected vehicles to estimate spots
            if vehicle_boxes:
                self.parking_spots = [(x1, y1, x2, y2) for x1, y1, x2, y2 in vehicle_boxes]
        
        total_spots = len(self.parking_spots)
        occupied_count = 0
        
        # Check each parking spot for occupancy
        for i, spot in enumerate(self.parking_spots):
            is_occupied = self.check_spot_occupancy(spot, vehicle_boxes)
            
            if is_occupied:
                occupied_count += 1
                color = self.occupied_color
            else:
                color = self.available_color
            
            # Draw parking spot
            x1, y1, x2, y2 = [int(v) for v in spot]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw spot number
            cv2.putText(frame, str(i + 1), (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        available_spots = total_spots - occupied_count
        
        stats = {
            'total': total_spots,
            'occupied': occupied_count,
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
    
    def generate_default_spots(self, frame_width, frame_height, rows=3, cols=8):
        """
        Generate compact parking spot grid
        
        Args:
            frame_width: Frame width
            frame_height: Frame height
            rows: Number of rows (default 3)
            cols: Number of columns (default 8)
            
        Returns:
            List of parking spot rectangles
        """
        spots = []
        
        # Smaller, more realistic parking spots
        spot_width = 80  # Fixed width for consistency
        spot_height = 60  # Fixed height
        spacing = 10
        
        # Center the grid
        total_width = cols * spot_width + (cols - 1) * spacing
        total_height = rows * spot_height + (rows - 1) * spacing
        
        start_x = (frame_width - total_width) // 2
        start_y = (frame_height - total_height) // 2
        
        for row in range(rows):
            for col in range(cols):
                x1 = start_x + col * (spot_width + spacing)
                y1 = start_y + row * (spot_height + spacing)
                x2 = x1 + spot_width
                y2 = y1 + spot_height
                
                spots.append((x1, y1, x2, y2))
        
        return spots
