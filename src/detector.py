"""
Parking detection with YOLOv8 and smart grid
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """Initialize"""
        print(f"Loading YOLOv8 model...")
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)
        self.available_color = (0, 255, 0)
        self.parking_grid = []
        self.grid_initialized = False
        
    def detect_vehicles(self, frame):
        """Detect ALL objects with YOLO"""
        # Lower confidence to detect more
        results = self.model(frame, conf=0.2, iou=0.5, verbose=False)
        return results
    
    def get_vehicle_bboxes(self, results):
        """Get vehicle bounding boxes"""
        vehicles = []
        
        if results:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Cars(2), motorcycles(3), buses(5), trucks(7)
                        if cls in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicles.append((int(x1), int(y1), int(x2), int(y2)))
        
        return vehicles
    
    def create_smart_grid(self, frame, vehicle_boxes):
        """Create parking grid based on detected vehicles"""
        h, w = frame.shape[:2]
        
        if not vehicle_boxes or len(vehicle_boxes) < 3:
            return []
        
        # Analyze vehicle positions to understand parking layout
        vehicle_data = []
        for x1, y1, x2, y2 in vehicle_boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            vw = x2 - x1
            vh = y2 - y1
            vehicle_data.append({'cx': cx, 'cy': cy, 'w': vw, 'h': vh, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        
        # Sort by y position to find rows
        vehicle_data.sort(key=lambda v: v['cy'])
        
        # Find rows (group vehicles with similar y)
        rows = []
        current_row = [vehicle_data[0]]
        
        for v in vehicle_data[1:]:
            if abs(v['cy'] - current_row[-1]['cy']) < h * 0.15:  # Same row
                current_row.append(v)
            else:
                if len(current_row) >= 3:  # Valid row needs 3+ vehicles
                    rows.append(current_row)
                current_row = [v]
        
        if len(current_row) >= 3:
            rows.append(current_row)
        
        if not rows:
            return []
        
        # Create grid cells for each row
        grid_cells = []
        
        for row in rows:
            # Sort row by x position
            row.sort(key=lambda v: v['cx'])
            
            # Get average vehicle size in this row
            avg_w = int(np.mean([v['w'] for v in row]))
            avg_h = int(np.mean([v['h'] for v in row]))
            
            # Get row bounds
            row_y_min = int(np.min([v['y1'] for v in row]))
            row_y_max = int(np.max([v['y2'] for v in row]))
            
            # Find leftmost and rightmost
            leftmost_x = row[0]['x1']
            rightmost_x = row[-1]['x2']
            
            # Calculate average spacing
            spacings = []
            for i in range(len(row) - 1):
                spacing = row[i+1]['cx'] - row[i]['cx']
                spacings.append(spacing)
            
            avg_spacing = int(np.mean(spacings)) if spacings else avg_w + 20
            
            # Generate grid for entire row (including gaps)
            num_spots = max(len(row) + 3, int((rightmost_x - leftmost_x) / avg_spacing))
            
            for i in range(num_spots):
                x_center = leftmost_x + i * avg_spacing
                
                if x_center - avg_w//2 >= 0 and x_center + avg_w//2 < w:
                    cell = {
                        'x1': x_center - avg_w//2,
                        'y1': row_y_min,
                        'x2': x_center + avg_w//2,
                        'y2': row_y_max
                    }
                    grid_cells.append(cell)
        
        return grid_cells
    
    def draw_detections(self, frame, vehicle_boxes):
        """Draw parking detection"""
        h, w = frame.shape[:2]
        
        # Initialize grid once
        if not self.grid_initialized:
            grid = self.create_smart_grid(frame, vehicle_boxes)
            
            if grid and len(grid) > 0:
                self.parking_grid = grid
                print(f"✓ Created smart grid: {len(grid)} parking spots")
            
            self.grid_initialized = True
        
        # Fallback if no grid
        if not self.parking_grid:
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            cv2.putText(frame, "Analyzing parking layout...", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            return frame, {
                'total': 0,
                'occupied': 0,
                'available': 0
            }
        
        # Check occupancy
        occupied = 0
        
        for cell in self.parking_grid:
            is_occupied = False
            
            # Check if any vehicle overlaps this cell
            for vx1, vy1, vx2, vy2 in vehicle_boxes:
                # Overlap check
                x_overlap = max(0, min(vx2, cell['x2']) - max(vx1, cell['x1']))
                y_overlap = max(0, min(vy2, cell['y2']) - max(vy1, cell['y1']))
                overlap_area = x_overlap * y_overlap
                cell_area = (cell['x2'] - cell['x1']) * (cell['y2'] - cell['y1'])
                
                if overlap_area > cell_area * 0.3:  # 30% overlap
                    is_occupied = True
                    break
            
            # Draw
            color = self.occupied_color if is_occupied else self.available_color
            cv2.rectangle(frame, (cell['x1'], cell['y1']), (cell['x2'], cell['y2']), color, 2)
            
            if is_occupied:
                occupied += 1
        
        total = len(self.parking_grid)
        available = total - occupied
        
        return frame, {
            'total': total,
            'occupied': occupied,
            'available': available
        }
