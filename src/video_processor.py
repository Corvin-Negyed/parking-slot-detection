"""
Video processing module for parking detection
"""

import cv2
import time
import base64
import numpy as np
from src.detector import ParkingDetector
from src.database import DatabaseManager


class VideoProcessor:
    def __init__(self, video_source, parking_spots=None):
        """Initialize video processor"""
        self.video_source = video_source
        self.cap = None
        self.detector = ParkingDetector(parking_spots)
        self.db = DatabaseManager()
        self.is_running = False
        self.frame_skip = 1  # Process every frame
        self.frame_count = 0
        self.previous_states = {}
        self.current_stats = {'total': 0, 'occupied': 0, 'available': 0}
        
    def open_video(self):
        """Open video source"""
        print(f"Opening: {self.video_source}")
        
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open {self.video_source}")
            raise ValueError(f"Cannot open video: {self.video_source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video OK: {self.width}x{self.height} @ {self.fps}fps")
        
        # Reset for new video
        self.detector.vehicle_history = []
        self.frame_count = 0
        self.previous_states = {}
        
        return True
    
    def close_video(self):
        """Release video and close database"""
        if self.cap:
            self.cap.release()
        self.db.close()
        self.is_running = False
    
    def process_frame(self, frame):
        """Process single frame"""
        try:
            # Detect vehicles
            results = self.detector.detect_vehicles(frame)
            vehicle_boxes = self.detector.get_vehicle_bboxes(results)
            
            # Draw parking spots and check occupancy
            processed_frame, stats = self.detector.draw_detections(frame, vehicle_boxes)
            
            # Update current stats
            self.current_stats = stats
            
            # Add overlay
            self._draw_stats_overlay(processed_frame, stats)
            
            # Log changes
            self._check_and_log_changes(stats)
            
            return processed_frame, stats
            
        except Exception as e:
            print(f"Frame error: {e}")
            import traceback
            traceback.print_exc()
            return frame, self.current_stats
    
    def _draw_stats_overlay(self, frame, stats):
        """Draw statistics on frame"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, f"Total Spots: {stats['total']}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y += 30
        cv2.putText(frame, f"Occupied: {stats['occupied']}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y += 30
        cv2.putText(frame, f"Available: {stats['available']}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if stats['total'] > 0:
            occupancy = (stats['occupied'] / stats['total']) * 100
            y += 30
            cv2.putText(frame, f"Occupancy: {occupancy:.1f}%", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def _check_and_log_changes(self, stats):
        """Log occupancy changes to database"""
        current = stats['occupied']
        
        if 'occupied' not in self.previous_states:
            self.previous_states['occupied'] = current
            self.db.save_detection(stats)
            return
        
        if current != self.previous_states['occupied']:
            old = self.previous_states['occupied']
            print(f"Occupancy: {old} -> {current}")
            self.db.save_detection(stats)
            self.previous_states['occupied'] = current
    
    def process_video_stream(self):
        """Process video stream (generator)"""
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames
            if self.frame_count % self.frame_skip != 0:
                continue
            
            # Process
            processed_frame, stats = self.process_frame(frame)
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield frame_bytes, stats
    
    def stop(self):
        """Stop processing"""
        self.is_running = False

