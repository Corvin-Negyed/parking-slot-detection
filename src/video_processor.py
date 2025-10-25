"""
Video processing module for real-time parking detection.
Handles video streams and file processing with frame-by-frame analysis.
"""

import cv2
import time
import base64
import numpy as np
from src.detector import ParkingDetector
from src.database import DatabaseManager


class VideoProcessor:
    def __init__(self, video_source, parking_spots=None):
        """
        Initialize video processor
        
        Args:
            video_source: Path to video file or stream URL
            parking_spots: List of parking spot coordinates (optional)
        """
        self.video_source = video_source
        self.cap = None
        self.detector = ParkingDetector(parking_spots)
        self.db = DatabaseManager()
        self.is_running = False
        self.frame_skip = 2  # Process every nth frame for performance
        self.frame_count = 0
        self.previous_states = {}  # Track previous spot states
        
    def open_video(self):
        """Open video source (file or stream)"""
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.video_source}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # No default parking spots - will detect vehicles dynamically
        
        return True
    
    def close_video(self):
        """Release video capture and close database"""
        if self.cap:
            self.cap.release()
        self.db.close()
        self.is_running = False
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame and statistics
        """
        # Detect vehicles
        results = self.detector.detect_vehicles(frame)
        vehicle_boxes = self.detector.get_vehicle_bboxes(results)
        
        # Draw detected vehicles and get statistics
        processed_frame, stats = self.detector.draw_detections(frame, vehicle_boxes)
        
        # Add statistics overlay
        self._draw_stats_overlay(processed_frame, stats)
        
        # Check for state changes and log to database
        self._check_and_log_changes(stats)
        
        return processed_frame, stats
    
    def _draw_stats_overlay(self, frame, stats):
        """Draw statistics overlay on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        y_offset = 35
        cv2.putText(frame, f"Vehicles Detected: {stats['occupied']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _check_and_log_changes(self, stats):
        """Check for parking state changes and log to database"""
        current_occupied = stats['occupied']
        
        # Initialize previous state if first run
        if 'occupied' not in self.previous_states:
            self.previous_states['occupied'] = current_occupied
            return
        
        # Check if state changed
        if current_occupied != self.previous_states['occupied']:
            # Log the change
            status = 'occupied' if current_occupied > self.previous_states['occupied'] else 'available'
            
            # Save to database (using spot 0 for overall status)
            self.db.save_event(0, f"{current_occupied}_occupied")
            
            self.previous_states['occupied'] = current_occupied
    
    def process_video_stream(self):
        """
        Process video stream (generator for real-time streaming)
        
        Yields:
            Encoded frame bytes and statistics
        """
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                # End of video or error
                break
            
            self.frame_count += 1
            
            # Skip frames for performance
            if self.frame_count % self.frame_skip != 0:
                continue
            
            # Process frame
            processed_frame, stats = self.process_frame(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield frame_bytes, stats
    
    def get_frame_base64(self, frame):
        """
        Convert frame to base64 for web transmission
        
        Args:
            frame: Input frame
            
        Returns:
            Base64 encoded frame string
        """
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
    
    def set_parking_spots(self, spots):
        """Set custom parking spot coordinates"""
        self.detector.set_parking_spots(spots)
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
