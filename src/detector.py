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
        # reference size for polygons stored in object/poligon.obj
        self.ref_w = Config.POLYGON_REF_WIDTH
        self.ref_h = Config.POLYGON_REF_HEIGHT
        self.polygons_norm = []
        self.scaled_cache = {'key': None, 'polys': []}

        # If polygon usage is disabled, ensure no preloaded polygons
        if not getattr(Config, 'USE_POLYGON_FILE', False):
            self.polygons_norm = []
            self.polygon_spots = []

    def load_polygon_spots(self, obj_path='object/poligon.obj'):
        """Load parking spot polygons from pickle file"""
        try:
            with open(obj_path, 'rb') as f:
                polys = pickle.load(f) or []
            # normalize to 0..1 if values look like pixels
            self.polygons_norm = []
            for poly in polys:
                if any(px > 1.5 or py > 1.5 for px, py in poly):
                    self.polygons_norm.append([(px / self.ref_w, py / self.ref_h) for px, py in poly])
                else:
                    self.polygons_norm.append(poly)
            if self.polygons_norm:
                print(f"Loaded {len(self.polygons_norm)} polygons from {obj_path}")
                self.lines_detected = True
        except Exception as e:
            print(f"No polygon file found: {e}")
            self.polygons_norm = []
    
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
    
    def get_scaled_polygons(self, frame_w, frame_h):
        key = (frame_w, frame_h)
        if self.scaled_cache['key'] == key:
            return self.scaled_cache['polys']
        scaled = [[(int(px * frame_w), int(py * frame_h)) for px, py in poly] for poly in self.polygons_norm]
        self.scaled_cache = {'key': key, 'polys': scaled}
        return scaled
        
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
        Draw parking spots using polygons and check occupancy
        
        Args:
            frame: Video frame to draw on
            vehicle_boxes: List of detected vehicle bounding boxes
            
        Returns:
            Frame with drawn parking spots and statistics
        """
        # Use polygon spots if available and enabled
        if Config.USE_POLYGON_FILE and self.polygons_norm and len(self.polygons_norm) > 0:
            h, w = frame.shape[:2]
            polys = self.get_scaled_polygons(w, h)
            total_spots = len(polys)
            occupied_count = 0
            
            # Create masks
            mask_occupied = np.zeros_like(frame)
            mask_available = np.zeros_like(frame)
            
            # Check each polygon
            for polygon in polys:
                is_occupied = False
                
                # Check if any vehicle center is inside polygon
                for x1, y1, x2, y2 in vehicle_boxes:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    if self.is_point_in_polygon(center, polygon):
                        is_occupied = True
                        break
                
                # Draw polygon
                poly_array = np.array(polygon, dtype=np.int32)
                
                if is_occupied:
                    occupied_count += 1
                    cv2.fillPoly(mask_occupied, [poly_array], self.occupied_color)
                else:
                    cv2.fillPoly(mask_available, [poly_array], self.available_color)
            
            # Blend
            frame = cv2.addWeighted(mask_occupied, 0.3, frame, 1, 0)
            frame = cv2.addWeighted(mask_available, 0.3, frame, 1, 0)
            
            stats = {
                'total': total_spots,
                'occupied': occupied_count,
                'available': total_spots - occupied_count
            }
            
            return frame, stats
        
        # Else detect from lines dynamically per video
        h, w = frame.shape[:2]
        lines = self.detect_parking_lines(frame)
        spots = self.create_spots_from_lines(lines, w, h)
        if spots:
            self.parking_spots = spots
            total_spots = len(spots)
            occ = 0
            for spot in spots:
                if self.check_spot_occupancy(spot, vehicle_boxes):
                    occ += 1
                    color = self.occupied_color
                else:
                    color = self.available_color
                x1, y1, x2, y2 = [int(v) for v in spot]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            stats = {'total': total_spots, 'occupied': occ, 'available': total_spots - occ}
            return frame, stats
        
        # Final fallback: only vehicles
        for (x1, y1, x2, y2) in vehicle_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.occupied_color, 2)
        n = len(vehicle_boxes)
        return frame, {'total': n, 'occupied': n, 'available': 0}
    
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
