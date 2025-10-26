"""
Parking detection with vehicle pattern analysis
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """Initialize"""
        print("Initializing YOLOv8...")
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)
        self.available_color = (0, 255, 0)
        
        # Learning system
        self.learning_frames = []
        self.learning_complete = False
        self.max_learning_frames = 15
        self.parking_grid = []
        
    def detect_vehicles(self, frame):
        """Detect vehicles with YOLO"""
        return self.model(frame, conf=0.25, iou=0.45, verbose=False)
    
    def get_vehicle_bboxes(self, results):
        """Extract vehicle boxes"""
        boxes = []
        
        if results:
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in [2, 3, 5, 7]:  # vehicles
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return boxes
    
    def build_parking_grid(self, all_vehicles, frame_w, frame_h):
        """Build complete parking grid from collected vehicles"""
        if len(all_vehicles) < 5:
            return []
        
        # Sort by y then x
        sorted_v = sorted(all_vehicles, key=lambda v: (v[1], v[0]))
        
        # Group into rows
        rows = []
        current_row = [sorted_v[0]]
        
        for v in sorted_v[1:]:
            if abs(v[1] - current_row[0][1]) < frame_h * 0.12:
                current_row.append(v)
            else:
                if len(current_row) >= 2:
                    rows.append(current_row)
                current_row = [v]
        
        if len(current_row) >= 2:
            rows.append(current_row)
        
        if not rows:
            return []
        
        # Create grid
        grid = []
        
        for row in rows:
            row.sort(key=lambda v: v[0])  # sort by x
            
            # Calculate average vehicle size
            widths = [v[2] - v[0] for v in row]
            heights = [v[3] - v[1] for v in row]
            avg_w = int(np.median(widths))
            avg_h = int(np.median(heights))
            
            # Get row bounds
            y_min = int(np.min([v[1] for v in row]))
            y_max = int(np.max([v[3] for v in row]))
            
            # Calculate spacing
            if len(row) > 1:
                centers = [(v[0] + v[2])//2 for v in row]
                spacings = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
                avg_spacing = int(np.median(spacings))
            else:
                avg_spacing = avg_w + 20
            
            # Get row range
            x_start = row[0][0]
            x_end = row[-1][2]
            
            # Generate ALL spots (including empty ones)
            num_spots = int((x_end - x_start) / avg_spacing) + 2
            
            for i in range(num_spots):
                cx = x_start + i * avg_spacing
                
                if 0 <= cx - avg_w//2 and cx + avg_w//2 < frame_w:
                    grid.append({
                        'x1': cx - avg_w//2,
                        'y1': y_min,
                        'x2': cx + avg_w//2,
                        'y2': y_max
                    })
        
        return grid
    
    def draw_detections(self, frame, vehicle_boxes):
        """Main detection"""
        h, w = frame.shape[:2]
        
        # Learning phase: collect vehicle positions
        if not self.learning_complete:
            if vehicle_boxes:
                self.learning_frames.extend(vehicle_boxes)
            
            if len(self.learning_frames) >= self.max_learning_frames * 3:  # Got enough data
                # Build grid from all collected vehicles
                unique_vehicles = []
                for v in self.learning_frames:
                    # Remove duplicates (same position)
                    is_new = True
                    for uv in unique_vehicles:
                        if abs(v[0] - uv[0]) < 30 and abs(v[1] - uv[1]) < 30:
                            is_new = False
                            break
                    if is_new:
                        unique_vehicles.append(v)
                
                self.parking_grid = self.build_parking_grid(unique_vehicles, w, h)
                self.learning_complete = True
                
                if self.parking_grid:
                    print(f"✓✓✓ Built grid: {len(self.parking_grid)} parking spots")
                else:
                    print("⚠ Could not build grid")
            else:
                # Still learning
                cv2.putText(frame, f"Analyzing parking layout: {len(self.learning_frames)} vehicles found", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                for x1, y1, x2, y2 in vehicle_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
                return frame, {'total': 0, 'occupied': 0, 'available': 0}
        
        # No grid
        if not self.parking_grid:
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            return frame, {'total': 0, 'occupied': 0, 'available': 0}
        
        # Check grid occupancy
        occupied = 0
        
        for spot in self.parking_grid:
            is_occ = False
            
            for vx1, vy1, vx2, vy2 in vehicle_boxes:
                # Overlap
                ox = max(0, min(vx2, spot['x2']) - max(vx1, spot['x1']))
                oy = max(0, min(vy2, spot['y2']) - max(vy1, spot['y1']))
                overlap = ox * oy
                spot_area = (spot['x2'] - spot['x1']) * (spot['y2'] - spot['y1'])
                
                if overlap > spot_area * 0.4:
                    is_occ = True
                    break
            
            color = self.occupied_color if is_occ else self.available_color
            cv2.rectangle(frame, (spot['x1'], spot['y1']), (spot['x2'], spot['y2']), color, 2)
            
            if is_occ:
                occupied += 1
        
        total = len(self.parking_grid)
        
        return frame, {
            'total': total,
            'occupied': occupied,
            'available': total - occupied
        }
