"""
Simple parking detection: stationary vehicles + gaps
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """Initialize"""
        print("Loading YOLO...")
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)  # Red = occupied
        self.available_color = (0, 255, 0)  # Green = available
        
        # Stationary vehicle tracking
        self.vehicle_history = []
        self.history_size = 5  # Track last 5 frames
        
    def detect_vehicles(self, frame):
        """Detect vehicles"""
        return self.model(frame, conf=0.3, verbose=False)
    
    def get_stationary_vehicles(self, results):
        """Get only stationary (parked) vehicles"""
        # Get current detections
        current = []
        
        if results:
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            current.append({
                                'box': (int(x1), int(y1), int(x2), int(y2)),
                                'center': (cx, cy)
                            })
        
        # Add to history
        self.vehicle_history.append(current)
        if len(self.vehicle_history) > self.history_size:
            self.vehicle_history.pop(0)
        
        # Not enough history yet
        if len(self.vehicle_history) < 3:
            return [v['box'] for v in current]
        
        # Filter: keep only stationary (center doesn't move much)
        stationary = []
        
        for v in current:
            cx, cy = v['center']
            is_stationary = True
            
            # Check in previous frames
            for hist_frame in self.vehicle_history[:-1]:
                found_match = False
                for hv in hist_frame:
                    hx, hy = hv['center']
                    dist = np.sqrt((cx - hx)**2 + (cy - hy)**2)
                    
                    if dist < 25:  # Same vehicle (not moved)
                        found_match = True
                        break
                
                if not found_match:
                    is_stationary = False
                    break
            
            if is_stationary:
                stationary.append(v['box'])
        
        return stationary
    
    def find_empty_spots(self, occupied_vehicles, frame_w, frame_h):
        """Find empty spots between parked vehicles"""
        if len(occupied_vehicles) < 2:
            return []
        
        # Sort by y (rows) then x (columns)
        vehicles = sorted(occupied_vehicles, key=lambda v: (v[1], v[0]))
        
        # Group into rows
        rows = []
        current_row = [vehicles[0]]
        
        for v in vehicles[1:]:
            if abs(v[1] - current_row[0][1]) < frame_h * 0.15:
                current_row.append(v)
            else:
                if len(current_row) >= 2:
                    rows.append(current_row)
                current_row = [v]
        
        if len(current_row) >= 2:
            rows.append(current_row)
        
        empty_spots = []
        
        # For each row, find gaps
        for row in rows:
            row.sort(key=lambda v: v[0])  # Sort by x
            
            # Average vehicle width in this row
            widths = [v[2] - v[0] for v in row]
            avg_width = int(np.mean(widths))
            
            # Row bounds
            row_y1 = int(np.min([v[1] for v in row]))
            row_y2 = int(np.max([v[3] for v in row]))
            
            # Check gaps between consecutive vehicles
            for i in range(len(row) - 1):
                v1 = row[i]
                v2 = row[i + 1]
                
                gap_start = v1[2]  # Right edge of left vehicle
                gap_end = v2[0]    # Left edge of right vehicle
                gap_size = gap_end - gap_start
                
                # If gap is big enough for parking spot(s)
                if gap_size > avg_width * 0.8:
                    # How many spots fit?
                    num_spots = int(gap_size / avg_width)
                    
                    for j in range(num_spots):
                        spot_x1 = gap_start + j * avg_width
                        spot_x2 = spot_x1 + avg_width
                        
                        if spot_x2 <= gap_end:
                            empty_spots.append({
                                'x1': int(spot_x1),
                                'y1': row_y1,
                                'x2': int(spot_x2),
                                'y2': row_y2
                            })
        
        return empty_spots
    
    def draw_detections(self, frame, vehicle_boxes):
        """Draw occupied and available parking spots"""
        h, w = frame.shape[:2]
        
        # Get stationary vehicles only
        stationary = self.get_stationary_vehicles(self.model.predict(frame, conf=0.3, verbose=False))
        
        if not stationary:
            cv2.putText(frame, "Waiting for vehicles...", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            return frame, {'total': 0, 'occupied': 0, 'available': 0}
        
        # Find empty spots
        empty = self.find_empty_spots(stationary, w, h)
        
        # Draw occupied (red)
        for x1, y1, x2, y2 in stationary:
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.occupied_color, 3)
        
        # Draw available (green)
        for spot in empty:
            cv2.rectangle(frame, (spot['x1'], spot['y1']), (spot['x2'], spot['y2']), 
                         self.available_color, 3)
        
        total = len(stationary) + len(empty)
        occupied = len(stationary)
        available = len(empty)
        
        return frame, {
            'total': total,
            'occupied': occupied,
            'available': available
        }
