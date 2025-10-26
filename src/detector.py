"""
Parking detection module using YOLOv8.
Detects vehicles and parking spot occupancy with visual feedback.
"""

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from shapely.geometry import Polygon as ShapelyPolygon
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
        # track recent vehicle centers to filter moving vehicles
        self.prev_centers = []  # list of lists of (x,y)
        self.max_history = max(2, Config.STATIONARY_FRAMES)

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

    def iou_polygon_bbox(self, poly_pts, bbox):
        """IoU between polygon and bbox (x1,y1,x2,y2)."""
        try:
            pg = ShapelyPolygon(poly_pts)
            x1, y1, x2, y2 = bbox
            bb = ShapelyPolygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            inter = pg.intersection(bb).area
            uni = pg.union(bb).area
            return (inter / uni) if uni > 0 else 0.0
        except Exception:
            return 0.0
        
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
        Robust divider-only method:
        1) Pick dominant oblique angle via histogram (exclude near-horizontal/vertical).
        2) Keep long lines around that angle.
        3) Project each line onto two horizontal bands y_top/y_bot estimated from endpoints (percentiles).
        4) Adjacent lines -> trapezoid polygon between y_top/y_bot.
        """
        if lines is None or len(lines) < 3:
            return []

        # Collect segments and angles (0..180)
        segs = [l[0] for l in lines]
        angles = []
        lengths = []
        for x1, y1, x2, y2 in segs:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            ang = 180 - ang if ang > 180 else ang
            angles.append(ang)
            lengths.append(np.hypot(x2 - x1, y2 - y1))

        # Build histogram in 2 deg bins to find dominant oblique orientation
        angles_np = np.array(angles)
        # exclude almost horizontal/vertical
        mask_oblique = (angles_np > 20) & (angles_np < 160)
        if mask_oblique.sum() < 3:
            return []
        hist, bin_edges = np.histogram(angles_np[mask_oblique], bins=np.arange(0, 181, 2))
        peak_idx = np.argmax(hist)
        peak_angle = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2

        # Filter lines near peak angle and with sufficient length
        selected = []
        for (x1, y1, x2, y2), ang, ln in zip(segs, angles, lengths):
            if abs(ang - peak_angle) <= 6 and ln > 0.18 * frame_height:
                selected.append((x1, y1, x2, y2))
        if len(selected) < 3:
            return []

        # Estimate vertical band of stalls by endpoint y percentiles
        ys = []
        for x1, y1, x2, y2 in selected:
            ys.extend([y1, y2])
        ys_np = np.array(ys)
        y_top = int(np.percentile(ys_np, 10))
        y_bot = int(np.percentile(ys_np, 90))
        if y_bot - y_top < 0.15 * frame_height:
            # band too thin -> unreliable
            return []

        # Utility: line parameters and x at a y
        def fit_mb(x1, y1, x2, y2):
            if x2 == x1:
                x2 += 1e-6
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return m, b

        def x_at_y(line, y):
            x1, y1, x2, y2 = line
            m, b = fit_mb(x1, y1, x2, y2)
            if abs(m) < 1e-6:
                return (x1 + x2) / 2
            return (y - b) / m

        # Sort by x at mid y
        y_mid = (y_top + y_bot) // 2
        selected.sort(key=lambda ln: x_at_y(ln, y_mid))

        polygons = []
        for i in range(len(selected) - 1):
            l1 = selected[i]
            l2 = selected[i + 1]
            x1_top = x_at_y(l1, y_top); x2_top = x_at_y(l2, y_top)
            x1_bot = x_at_y(l1, y_bot); x2_bot = x_at_y(l2, y_bot)
            # sanity width
            w_top = abs(x2_top - x1_top); w_bot = abs(x2_bot - x1_bot)
            w_avg = (w_top + w_bot) / 2
            if w_avg < max(18, 0.012 * frame_width) or w_avg > 0.18 * frame_width:
                continue
            pts = [(int(x1_top), y_top), (int(x2_top), y_top), (int(x2_bot), y_bot), (int(x1_bot), y_bot)]
            # keep inside
            if any(not (0 <= x < frame_width and 0 <= y < frame_height) for x, y in pts):
                continue
            polygons.append(pts)

        return polygons
    
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
        # filter moving vehicles by center displacement over last frames
        centers = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in vehicle_boxes]
        self.prev_centers.append(centers)
        if len(self.prev_centers) > self.max_history:
            self.prev_centers.pop(0)
        filtered = []
        if len(self.prev_centers) >= self.max_history:
            # keep vehicles whose centers did not move more than threshold
            thresh = Config.STATIONARY_PIXELS
            for idx, c in enumerate(centers):
                stationary = True
                for hist in self.prev_centers:
                    if idx >= len(hist):
                        continue
                    cx, cy = hist[idx]
                    if abs(cx - c[0]) > thresh or abs(cy - c[1]) > thresh:
                        stationary = False
                        break
                if stationary:
                    filtered.append(vehicle_boxes[idx])
        else:
            filtered = vehicle_boxes
        return filtered
    
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
                # Use IoU against each detected vehicle bbox for robust occupancy
                for x1, y1, x2, y2 in vehicle_boxes:
                    iou = self.iou_polygon_bbox(polygon, (x1, y1, x2, y2))
                    if iou >= Config.POLYGON_IOU_THRESHOLD:
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
        polygons = self.create_spots_from_lines(lines, w, h)
        if polygons:
            total_spots = len(polygons)
            occ = 0
            mask_occ = np.zeros_like(frame)
            mask_free = np.zeros_like(frame)
            for poly in polygons:
                is_occ = False
                for (x1, y1, x2, y2) in vehicle_boxes:
                    if self.iou_polygon_bbox(poly, (x1, y1, x2, y2)) >= Config.POLYGON_IOU_THRESHOLD:
                        is_occ = True
                        break
                poly_arr = np.array(poly, np.int32)
                if is_occ:
                    occ += 1
                    cv2.fillPoly(mask_occ, [poly_arr], self.occupied_color)
                else:
                    cv2.fillPoly(mask_free, [poly_arr], self.available_color)
            frame = cv2.addWeighted(mask_occ, 0.3, frame, 1, 0)
            frame = cv2.addWeighted(mask_free, 0.3, frame, 1, 0)
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
