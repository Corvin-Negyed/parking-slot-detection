"""
Parking detection using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import Config


class ParkingDetector:
    def __init__(self, parking_spots=None):
        """Initialize detector"""
        print(f"Loading model: {Config.MODEL_PATH}")
        self.model = YOLO(Config.MODEL_PATH)
        self.occupied_color = (0, 0, 255)  # Red
        self.available_color = (0, 255, 0)  # Green
        self.parking_areas = {}  # Track parking areas
        
    def detect_vehicles(self, frame):
        """Detect vehicles"""
        results = self.model(frame, verbose=False)
        return results
    
    def get_vehicle_bboxes(self, results):
        """Get vehicle bounding boxes"""
        vehicles = []
        
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Cars, trucks, buses, motorcycles
                        if cls in [2, 3, 5, 7] and conf > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicles.append((int(x1), int(y1), int(x2), int(y2)))
        
        print(f"Detected {len(vehicles)} vehicles")
        return vehicles
    
    def update_parking_areas(self, vehicle_boxes):
        """Update parking area database from detected vehicles"""
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            cx = (vx1 + vx2) // 2
            cy = (vy1 + vy2) // 2
            
            # Find if this matches existing area
            matched = False
            for area_id, area in self.parking_areas.items():
                ax, ay = area['center']
                dist = np.sqrt((cx - ax)**2 + (cy - ay)**2)
                
                if dist < 70:  # Same area
                    matched = True
                    # Update area bounds
                    area['x1'] = min(area['x1'], vx1)
                    area['y1'] = min(area['y1'], vy1)
                    area['x2'] = max(area['x2'], vx2)
                    area['y2'] = max(area['y2'], vy2)
                    area['center'] = ((area['x1'] + area['x2']) // 2, (area['y1'] + area['y2']) // 2)
                    area['last_seen'] = 0
                    break
            
            if not matched:
                # New parking area
                area_id = len(self.parking_areas)
                self.parking_areas[area_id] = {
                    'x1': vx1, 'y1': vy1, 'x2': vx2, 'y2': vy2,
                    'center': (cx, cy),
                    'last_seen': 0
                }
        
        # Age out areas not seen recently
        to_remove = []
        for area_id, area in self.parking_areas.items():
            area['last_seen'] += 1
            if area['last_seen'] > 100:  # Not seen for 100 frames
                to_remove.append(area_id)
        
        for area_id in to_remove:
            del self.parking_areas[area_id]
    
    def draw_detections(self, frame, vehicle_boxes):
        """Draw parking areas and occupancy"""
        
        # Update parking area database
        self.update_parking_areas(vehicle_boxes)
        
        if not self.parking_areas:
            # No areas yet, just show detected vehicles
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            cv2.putText(frame, "Detecting parking areas...", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            return frame, {
                'total': 0,
                'occupied': 0,
                'available': 0
            }
        
        # Draw all parking areas with occupancy
        mask_occupied = np.zeros_like(frame)
        mask_available = np.zeros_like(frame)
        
        occupied_count = 0
        
        for area_id, area in self.parking_areas.items():
            # Check if currently occupied
            is_occupied = False
            
            for vx1, vy1, vx2, vy2 in vehicle_boxes:
                # Check overlap
                if not (vx2 < area['x1'] or vx1 > area['x2'] or
                       vy2 < area['y1'] or vy1 > area['y2']):
                    # Significant overlap
                    overlap_x = min(vx2, area['x2']) - max(vx1, area['x1'])
                    overlap_y = min(vy2, area['y2']) - max(vy1, area['y1'])
                    overlap_area = overlap_x * overlap_y
                    area_size = (area['x2'] - area['x1']) * (area['y2'] - area['y1'])
                    
                    if overlap_area > area_size * 0.3:  # 30% overlap
                        is_occupied = True
                        break
            
            # Draw area
            x1, y1 = area['x1'], area['y1']
            x2, y2 = area['x2'], area['y2']
            
            pts = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], np.int32)
            
            if is_occupied:
                occupied_count += 1
                cv2.fillPoly(mask_occupied, [pts], self.occupied_color)
            else:
                cv2.fillPoly(mask_available, [pts], self.available_color)
        
        # Blend
        frame = cv2.addWeighted(mask_occupied, 0.3, frame, 1, 0)
        frame = cv2.addWeighted(mask_available, 0.3, frame, 1, 0)
        
        total = len(self.parking_areas)
        available = total - occupied_count
        
        stats = {
            'total': total,
            'occupied': occupied_count,
            'available': available
        }
        
        print(f"Stats: Total={total}, Occupied={occupied_count}, Available={available}")
        
        return frame, stats
