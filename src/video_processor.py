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
        self.frame_skip = 3  # Process every 3rd frame
        self.frame_count = 0
        self.previous_states = {}  # Track previous spot states
        self.current_stats = {'total': 0, 'occupied': 0, 'available': 0}  # Latest stats
        
    def open_video(self):
        """Open video source (file or stream)"""
        print(f"Opening: {self.video_source}")
        
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open {self.video_source}")
            raise ValueError(f"Cannot open video: {self.video_source}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video OK: {self.width}x{self.height} @ {self.fps}fps")

        # Reset detector state for new video
        # Clear any previously detected/generated spots and caches
        if hasattr(self.detector, 'lines_detected'):
            self.detector.lines_detected = False
        if hasattr(self.detector, 'parking_spots'):
            self.detector.parking_spots = []
        if hasattr(self.detector, 'polygons_norm'):
            # reload polygons only if enabled by config; otherwise keep empty
            if Config.USE_POLYGON_FILE:
                self.detector.load_polygon_spots()
            else:
                self.detector.polygons_norm = []
        if hasattr(self.detector, 'scaled_cache'):
            self.detector.scaled_cache = {'key': None, 'polys': []}
        # Reset runtime counters
        self.frame_count = 0
        self.previous_states = {}
        
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
        try:
            # Detect vehicles
            results = self.detector.detect_vehicles(frame)
            vehicle_boxes = self.detector.get_vehicle_bboxes(results)
            
            # Draw detected vehicles and get statistics
            processed_frame, stats = self.detector.draw_detections(frame, vehicle_boxes)
            
            # Update current stats for API access
            self.current_stats = stats
            
            # Add statistics overlay
            self._draw_stats_overlay(processed_frame, stats)
            
            # Check for state changes and log to database
            self._check_and_log_changes(stats)
            
            return processed_frame, stats
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame, self.current_stats
    
    def _draw_stats_overlay(self, frame, stats):
        """Draw statistics overlay on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw stats
        y_offset = 35
        cv2.putText(frame, f"Total Spots: {stats['total']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Occupied: {stats['occupied']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Available: {stats['available']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate and show occupancy percentage
        if stats['total'] > 0:
            occupancy_pct = (stats['occupied'] / stats['total']) * 100
            y_offset += 30
            cv2.putText(frame, f"Occupancy: {occupancy_pct:.1f}%", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def _check_and_log_changes(self, stats):
        """Check for occupancy changes and log to database"""
        current_occupied = stats['occupied']
        
        # Initialize previous state if first run
        if 'occupied' not in self.previous_states:
            self.previous_states['occupied'] = current_occupied
            # Log initial state
            self.db.save_detection(stats)
            return
        
        # Check if occupancy changed
        if current_occupied != self.previous_states['occupied']:
            # Print for debugging
            old_count = self.previous_states['occupied']
            print(f"Occupancy changed: {old_count} -> {current_occupied} (Total: {stats['total']}, Available: {stats['available']})")
            
            # Log the change to database
            self.db.save_detection(stats)
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
