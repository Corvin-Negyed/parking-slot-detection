
import cv2
import time
import numpy as np
import threading
from datetime import datetime
from src.detector import ParkingDetector # V12 detector
from src.database import DatabaseManager
from src.config import Config # Ayarları kullan

class VideoProcessor:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = None
        self.fps = 30.0 # Default, updated in open_video
        self.width = 1280 # Default
        self.height = 720 # Default
        self.original_width = 1280
        self.original_height = 720

        self.detector = None
        self.frame_skip_counter = 0
        self.db = DatabaseManager()
        self.is_running = False
        self.frame_count = 0
        self.current_stats = {'total': 0, 'occupied': 0, 'available': 0}
        
        self.detection_interval_seconds = Config.DETECTION_INTERVAL_SECONDS
        self.detection_interval_frames = 1
        self.last_detection_frame = -1
        self.last_occupied_spots = []
        self.last_empty_spots = []
        self.process_every_n_frames = Config.PROCESS_EVERY_N_FRAMES
        
        self.log_interval = Config.DB_LOG_INTERVAL
        self.last_log_time = 0
        self.frame_lock = threading.Lock()

    def open_video(self):
        with self.frame_lock:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_source}")

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not self.fps or self.fps <= 0:
                 self.fps = 30.0

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.detection_interval_frames = max(1, int(self.detection_interval_seconds * self.fps))

        try:
            self.detector = ParkingDetector(fps=self.fps)
            self.detector.reset_state()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize detector: {e}")
        
        self.last_occupied_spots = []
        self.last_empty_spots = []
        self.frame_count = 0
        self.frame_skip_counter = 0
        self.last_log_time = 0
        self.last_detection_frame = -1
        return True

    def close_video(self):
        with self.frame_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        self.db.close()
        self.is_running = False

    def _draw_ui_elements(self, frame, occupied_spots, empty_spots):
        if frame is None or frame.size == 0:
            return frame, {'total': 0, 'occupied': 0, 'available': 0}
        
        try:
            if not self.detector:
                return frame, {'total': 0, 'occupied': 0, 'available': 0}
            
            overlay = frame.copy()
            alpha = 0.3

            for x1, y1, x2, y2 in occupied_spots:
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), self.detector.occupied_color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.detector.occupied_color, 2)

            for x1, y1, x2, y2 in empty_spots:
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), self.detector.available_color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.detector.available_color, 2)

            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw center lines (thin blue lines) to show parking rows
            if hasattr(self.detector, 'parking_grid') and self.detector.parking_grid:
                for item in self.detector.parking_grid:
                    if item.get('type') == 'center_line':
                        line_points = item.get('line', [])
                        if len(line_points) == 2:
                            pt1 = (int(line_points[0][0]), int(line_points[0][1]))
                            pt2 = (int(line_points[1][0]), int(line_points[1][1]))
                            # Draw thin blue line (BGR: (255, 0, 0) = blue)
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 1)  # Thin blue line

            occupied_count = len(occupied_spots)
            available_count = len(empty_spots)
            total_count = occupied_count + available_count
            stats = {'total': total_count, 'occupied': occupied_count, 'available': available_count}

            return frame, stats
            
        except Exception as e:
            print(f"Error in _draw_ui_elements: {e}")
            return frame, {'total': 0, 'occupied': 0, 'available': 0}


    def process_frame(self, frame):
        if self.detector is None:
            return frame, self.current_stats
        
        try:
            current_time = time.time()
            self.frame_count += 1
            
            # Frame skipping for smoother performance
            self.frame_skip_counter += 1
            should_process = (self.frame_skip_counter % self.process_every_n_frames == 0)

            # Always resize for performance
            if Config.RESIZE_WIDTH > 0 and frame.shape[1] > Config.RESIZE_WIDTH:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                new_w = Config.RESIZE_WIDTH
                new_h = int(new_w / aspect_ratio)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Only update dimensions on first resize
                if self.frame_count == 1:
                    self.width, self.height = new_w, new_h

            run_detection_this_frame = False
            recalculate_spots_this_frame = False

            # Only process if we should (frame skip logic)
            if should_process:
                # Check if we're in initial learning phase
                is_initial = self.detector.check_initial_phase()
                
                if is_initial:
                    # Learning phase: detect every processed frame
                    run_detection_this_frame = True
                    recalculate_spots_this_frame = True
                else:
                    # Monitoring phase: detect at intervals
                    if self.last_detection_frame == -1 or (self.frame_count - self.last_detection_frame >= self.detection_interval_frames):
                        run_detection_this_frame = True
                        self.last_detection_frame = self.frame_count

                # Run YOLO detection if needed
                if run_detection_this_frame:
                    yolo_results = self.detector.detect_vehicles(frame)
                    stationary_changed = self.detector.update_stationary_cars(yolo_results)
                    
                    if stationary_changed or not self.detector.grid_established:
                        recalculate_spots_this_frame = True

                # Recalculate parking spots if needed
                if recalculate_spots_this_frame:
                    occ, emp = self.detector.find_parking_spots(self.width, self.height)
                    self.last_occupied_spots = occ
                    self.last_empty_spots = emp
                else:
                    occ = self.last_occupied_spots
                    emp = self.last_empty_spots
            else:
                # Use cached results for skipped frames
                occ = self.last_occupied_spots
                emp = self.last_empty_spots

            # Always draw UI (fast operation)
            processed_frame, stats = self._draw_ui_elements(frame, occ, emp)

            # Log to database at intervals
            self._update_stats_and_log(stats, current_time)

            return processed_frame, stats

        except Exception as e:
            import traceback
            print(f"Error in process_frame: {e}")
            traceback.print_exc()
            return frame, self.current_stats


    def _update_stats_and_log(self, stats, current_time):
        self.current_stats = stats
        if stats.get('total', 0) >= 0 and (current_time - self.last_log_time) >= self.log_interval:
            try:
                self.db.save_detection(stats)
                self.last_log_time = current_time
                pass
            except Exception as e:
                pass

    def process_video_stream(self):
        self.is_running = True
        
        if not self.detector:
            return

        while self.is_running:
            try:
                with self.frame_lock:
                    if not self.is_running or not self.cap or not self.cap.isOpened():
                        break
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        break
                
                processed_frame, stats = self.process_frame(frame)
                
                if processed_frame is not None and processed_frame.size > 0:
                    ret_buf, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret_buf and buffer is not None:
                        frame_bytes = buffer.tobytes()
                        if frame_bytes:
                            yield frame_bytes, stats
                
                time.sleep(0.01)
                
            except GeneratorExit:
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                continue


    def stop(self):
        self.is_running = False