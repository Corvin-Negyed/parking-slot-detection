"""
Automatic parking detection without predefined data
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """Initialize"""
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)  # Red
        self.available_color = (0, 255, 0)  # Green
        self.parking_spots = []
        self.spots_initialized = False
        
    def detect_parking_lines(self, frame):
        """Detect white parking divider lines specifically"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # High threshold for bright white only (parking lines are very white)
        _, white = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        
        # Edge detection
        edges = cv2.Canny(white, 100, 200, apertureSize=3)
        
        # Detect lines - longer lines for parking dividers
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=40,  # Parking lines are long
            maxLineGap=15
        )
        
        return lines if lines is not None else []
    
    def create_spots_from_lines(self, lines, w, h):
        """Create parking spots from parallel divider lines"""
        if len(lines) < 2:
            return []
        
        # Analyze line orientations
        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize angle to 0-180
            if angle < 0:
                angle += 180
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Keep long lines only (parking dividers are prominent)
            if length > h * 0.08:  # At least 8% of frame height
                line_data.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': cx, 'cy': cy,
                    'angle': angle,
                    'length': length
                })
        
        if len(line_data) < 2:
            return []
        
        # Find dominant angle (parking lines are parallel)
        angles = [l['angle'] for l in line_data]
        hist, bins = np.histogram(angles, bins=36)  # 5-degree bins
        dominant_idx = np.argmax(hist)
        dominant_angle = (bins[dominant_idx] + bins[dominant_idx + 1]) / 2
        
        # Keep only parallel lines (within ±10 degrees of dominant)
        parallel_lines = []
        for l in line_data:
            if abs(l['angle'] - dominant_angle) < 10:
                parallel_lines.append(l)
        
        if len(parallel_lines) < 2:
            return []
        
        # Sort lines LEFT to RIGHT (by x coordinate)
        parallel_lines.sort(key=lambda l: l['cx'])
        
        spots = []
        
        # Create spots between adjacent parallel lines (LEFT and RIGHT neighbors)
        for i in range(len(parallel_lines) - 1):
            left_line = parallel_lines[i]
            right_line = parallel_lines[i + 1]
            
            # Check spacing (typical parking spot width)
            spacing = abs(right_line['cx'] - left_line['cx'])
            
            if 35 < spacing < 180:  # Valid parking spot width
                # Create quadrilateral between the two lines
                poly = [
                    (int(left_line['x1']), int(left_line['y1'])),
                    (int(left_line['x2']), int(left_line['y2'])),
                    (int(right_line['x2']), int(right_line['y2'])),
                    (int(right_line['x1']), int(right_line['y1']))
                ]
                
                # Keep only if within frame
                if all(0 <= x < w and 0 <= y < h for x, y in poly):
                    spots.append(poly)
        
        return spots
    
    def detect_vehicles(self, frame):
        """Detect vehicles"""
        return self.model(frame, conf=0.3, verbose=False)
    
    def get_vehicle_bboxes(self, results):
        """Get vehicle boxes"""
        vehicles = []
        
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Vehicle classes
                        if cls in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicles.append((int(x1), int(y1), int(x2), int(y2)))
        
        return vehicles
    
    def polygon_center(self, poly):
        """Get polygon center"""
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs) // len(xs), sum(ys) // len(ys))
    
    def point_in_polygon(self, point, polygon):
        """Check if point in polygon"""
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
    
    def draw_detections(self, frame, vehicle_boxes):
        """Main detection function"""
        h, w = frame.shape[:2]
        
        # Initialize parking spots once
        if not self.spots_initialized:
            lines = self.detect_parking_lines(frame)
            self.parking_spots = self.create_spots_from_lines(lines, w, h)
            self.spots_initialized = True
            
            if self.parking_spots:
                print(f"✓ Found {len(self.parking_spots)} parking spots from lines")
        
        # No spots detected
        if not self.parking_spots:
            # Just show vehicles
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            return frame, {
                'total': len(vehicle_boxes),
                'occupied': len(vehicle_boxes),
                'available': 0
            }
        
        # Check occupancy
        mask_occ = np.zeros_like(frame)
        mask_free = np.zeros_like(frame)
        
        occupied = 0
        
        for spot in self.parking_spots:
            is_occupied = False
            spot_center = self.polygon_center(spot)
            
            # Check if any vehicle covers this spot
            for vx1, vy1, vx2, vy2 in vehicle_boxes:
                car_poly = [(vx1, vy1), (vx1, vy2), (vx2, vy2), (vx2, vy1)]
                
                if self.point_in_polygon(spot_center, car_poly):
                    is_occupied = True
                    break
            
            poly_arr = np.array(spot, np.int32)
            
            if is_occupied:
                occupied += 1
                cv2.fillPoly(mask_occ, [poly_arr], self.occupied_color)
            else:
                cv2.fillPoly(mask_free, [poly_arr], self.available_color)
        
        # Blend
        frame = cv2.addWeighted(mask_occ, 0.3, frame, 1, 0)
        frame = cv2.addWeighted(mask_free, 0.3, frame, 1, 0)
        
        total = len(self.parking_spots)
        available = total - occupied
        
        return frame, {
            'total': total,
            'occupied': occupied,
            'available': available
        }
