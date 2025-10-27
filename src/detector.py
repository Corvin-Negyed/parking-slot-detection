
import cv2
import numpy as np
from ultralytics import YOLO
from src.config import Config
import time

class ParkingDetector:
    def __init__(self, fps=30):
        try:
            self.model = YOLO(Config.MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
        
        self.occupied_color = (0, 0, 255)
        self.available_color = (0, 255, 0)
        self.vehicle_history = []
        self.history_size = Config.STATIONARY_FRAMES
        self.stationary_threshold_frames = max(3, self.history_size // 2)
        self.stationary_pixel_threshold = Config.STATIONARY_PIXELS
        self.current_stationary_boxes = []
        self.parking_grid = []
        self.orientation = 'UNKNOWN'
        self.stage = 1
        self.initial_phase_seconds = 5.0
        self.initial_phase_start_time = time.time()
        self.is_initial_phase = True
        self.grid_established = False
        self.last_grid_rebuild_time = 0
        self.grid_rebuild_interval = 10.0

    def reset_state(self):
        self.vehicle_history = []
        self.current_stationary_boxes = []
        self.parking_grid = []
        self.orientation = 'UNKNOWN'
        self.stage = 1
        self.initial_phase_start_time = time.time()
        self.is_initial_phase = True
        self.grid_established = False
        pass

    def detect_vehicles(self, frame):
        # classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        # iou=0.2 for VERY tight NMS (minimal overlap)
        return self.model(frame, conf=0.3, classes=[2, 3, 5, 7], verbose=False, 
                         imgsz=1024, device='cpu', half=False, iou=0.2, max_det=100)

    def update_stationary_cars(self, results):
        current_detections = []
        try:
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # SHRINK box by 10% to avoid overlap
                    width = x2 - x1
                    height = y2 - y1
                    shrink_w = width * 0.05
                    shrink_h = height * 0.05
                    
                    x1 = int(x1 + shrink_w)
                    y1 = int(y1 + shrink_h)
                    x2 = int(x2 - shrink_w)
                    y2 = int(y2 - shrink_h)
                    
                    # Calculate center and dimensions
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Minimal filtering - only remove obvious noise
                    if width < 20 or height < 20:
                        continue
                    
                    # Very loose aspect ratio (allow all car shapes)
                    aspect_ratio = width / height if height > 0 else 0
                    if aspect_ratio < 0.2 or aspect_ratio > 8.0:
                        continue
                    
                    # Minimal area check
                    area = width * height
                    if area < 500:
                        continue
                    
                    current_detections.append({
                        'box': (x1, y1, x2, y2), 
                        'center': (center_x, center_y), 
                        'w': width, 
                        'h': height
                    })
        except Exception as e:
            pass

        self.vehicle_history.append(current_detections)
        if len(self.vehicle_history) > self.history_size:
            self.vehicle_history.pop(0)

        new_stationary_boxes = []
        if len(self.vehicle_history) >= self.stationary_threshold_frames:
            for v in self.vehicle_history[-1]:
                is_stationary = True
                for i in range(2, self.stationary_threshold_frames + 1):
                    if len(self.vehicle_history) < i:
                        is_stationary = False
                        break
                    hist_frame = self.vehicle_history[-i]
                    found_close = False
                    for hv in hist_frame:
                        dist = np.sqrt((v['center'][0] - hv['center'][0])**2 + 
                                     (v['center'][1] - hv['center'][1])**2)
                        if dist < self.stationary_pixel_threshold:
                            found_close = True
                            break
                    if not found_close:
                        is_stationary = False
                        break
                
                if is_stationary:
                    new_stationary_boxes.append(v)

        # Check if changed
        changed = len(new_stationary_boxes) != len(self.current_stationary_boxes)
        if not changed and len(new_stationary_boxes) > 0:
            # Deep comparison
            old_centers = sorted([b['center'] for b in self.current_stationary_boxes])
            new_centers = sorted([b['center'] for b in new_stationary_boxes])
            for oc, nc in zip(old_centers, new_centers):
                # Increased tolerance to prevent constant grid recalculation
                if abs(oc[0] - nc[0]) > 15 or abs(oc[1] - nc[1]) > 15:
                    changed = True
                    break
        
        if changed:
            self.current_stationary_boxes = new_stationary_boxes
            pass
        
        return changed

    def determine_orientation(self):
        if not self.current_stationary_boxes:
            return 'UNKNOWN'
        
        widths = [v['w'] for v in self.current_stationary_boxes]
        heights = [v['h'] for v in self.current_stationary_boxes]
        
        if not widths or not heights:
            return 'UNKNOWN'
        
        median_w = np.median(widths)
        median_h = np.median(heights)
        
        # Determine orientation based on aspect ratio
        if median_w > (median_h * 1.3):
            return 'HORIZONTAL'
        elif median_h > (median_w * 1.3):
            return 'VERTICAL'
        else:
            # Ambiguous - use spatial distribution
            return self._determine_orientation_by_distribution()
    
    def _determine_orientation_by_distribution(self):
        if len(self.current_stationary_boxes) < 2:
            return 'HORIZONTAL'  # Default
        
        centers = [v['center'] for v in self.current_stationary_boxes]
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        
        # Calculate variance in x and y directions
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)
        
        # If variance is higher in X, vehicles are spread horizontally
        if x_var > y_var * 1.5:
            return 'HORIZONTAL'
        elif y_var > x_var * 1.5:
            return 'VERTICAL'
        else:
            return 'HORIZONTAL'  # Default

    def build_parking_grid(self, frame_w, frame_h):
        if not self.current_stationary_boxes:
            return []
        
        self.orientation = self.determine_orientation()
        widths = [v['w'] for v in self.current_stationary_boxes]
        heights = [v['h'] for v in self.current_stationary_boxes]
        
        # Use 75th percentile instead of median (ignore small detections)
        median_w = int(np.percentile(widths, 75))
        median_h = int(np.percentile(heights, 75))
        
        print(f"Vehicle sizes: min={min(widths)}, median={int(np.median(widths))}, 75th={median_w}, max={max(widths)}")
        
        if self.orientation == 'HORIZONTAL':
            grid = self._build_perfect_horizontal_grid(frame_w, frame_h, median_w, median_h)
        else:
            grid = self._build_perfect_vertical_grid(frame_w, frame_h, median_w, median_h)
        
        self.parking_grid = grid
        self.grid_established = True
        self.stage = 3
        return grid
    
    def _build_perfect_horizontal_grid(self, frame_w, frame_h, median_w, median_h):
        grid = []
        
        print(f"\n{'='*60}")
        print(f"BUILDING GRID: {len(self.current_stationary_boxes)} cars")
        print(f"Car size: median_w={median_w}, median_h={median_h}")
        print(f"{'='*60}")
        
        # STEP 1: Group cars by drawing lines through centers (NATURAL WAY)
        rows = self._group_cars_by_center_lines(median_w)
        
        print(f"Found {len(rows)} lines")
        
        # STEP 2: Add all parked cars to grid first (WITH ROW INDEX!)
        for row_idx, row in enumerate(rows):
            for v in row:
                grid.append({'box': v['box'], 'center': v['center'], 'is_vehicle': True, 'row': row_idx})
        
        # STEP 3: Find horizontal gaps (within rows)
        total_horizontal_slots = 0

        for row_idx, row in enumerate(rows):
            if not row:
                continue
            
            row_sorted = sorted(row, key=lambda v: v['center'][0])
            print(f"  Line {row_idx+1}: {len(row_sorted)} cars")

            # Calculate row boundaries - USE MEDIAN CAR SIZE, not full row height!
            row_y_min = min(v['box'][1] for v in row_sorted)
            row_y_max = max(v['box'][3] for v in row_sorted)
            row_x_min = min(v['box'][0] for v in row_sorted)
            row_x_max = max(v['box'][2] for v in row_sorted)
            
            # Calculate proper slot dimensions - match neighboring car sizes
            row_heights = [v['h'] for v in row_sorted]
            row_widths = [v['w'] for v in row_sorted]
            row_avg_height = np.mean(row_heights)
            row_avg_width = np.mean(row_widths)
            
            # SURGICAL GAP ANALYSIS: Precision distance-based calibration
            distance_factor = row_avg_height / median_h
            
            # EXTREME SENSITIVITY for distant cars - catch every tiny gap!
            if distance_factor < 0.3:  # Super ultra-distant cars (deepest background)
                min_gap_horizontal = row_avg_width * 0.15  # 15% - EXTREME sensitivity for tiniest gaps
            elif distance_factor < 0.5:  # Ultra-distant cars  
                min_gap_horizontal = row_avg_width * 0.2   # 20% - MAXIMUM sensitivity
            elif distance_factor < 0.7:  # Very distant cars
                min_gap_horizontal = row_avg_width * 0.25  # 25% - HIGH sensitivity
            elif distance_factor < 0.9:  # Distant cars
                min_gap_horizontal = row_avg_width * 0.35  # 35% - GOOD sensitivity
            elif distance_factor < 1.1:  # Medium cars
                min_gap_horizontal = row_avg_width * 0.5   # 50% - moderate
            else:  # Close cars (>1.1)
                min_gap_horizontal = row_avg_width * 0.6   # 60% - conservative for large cars
                
            print(f"    SURGICAL: Distance factor: {distance_factor:.3f}, Min gap: {min_gap_horizontal:.1f}px")
            
            # PRECISION SLOT DIMENSIONS: Exact pixel calculations
            slot_height = int(row_avg_height + 0.5)  # Proper rounding
            row_center_y = (row_y_min + row_y_max) / 2
            slot_y_min = int(row_center_y - slot_height / 2 + 0.5)  # Precise rounding
            slot_y_max = int(row_center_y + slot_height / 2 + 0.5)
            
            # ADAPTIVE MARGIN: Ultra-small for distant cars to maximize gap usage
            if distance_factor < 0.5:  # Ultra-distant cars
                margin_factor = 0.02  # Only 2% margin for tiny distant cars
                parking_margin = max(2, int(row_avg_width * margin_factor + 0.5))  # Minimum 2px
            elif distance_factor < 0.8:  # Distant cars  
                margin_factor = 0.05  # 5% margin for small cars
                parking_margin = max(4, int(row_avg_width * margin_factor + 0.5))  # Minimum 4px
            else:  # Close cars
                margin_factor = 0.08  # Standard 8% margin
                parking_margin = max(6, int(row_avg_width * margin_factor + 0.5))  # Minimum 6px

            # 1. ENHANCED Begin gap detection - especially for distant cars
            first_car = row_sorted[0]
            begin_gap = first_car['box'][0]  # Distance from left edge
            
            # SMART edge detection - avoid creating slots in middle of nowhere
            max_edge_distance = median_w * 1.2 if distance_factor < 0.6 else median_w * 1.0  # Much stricter
            
            # ADDITIONAL CHECK: Only create edge slot if it looks like actual parking space
            # Check if there are other cars nearby in X direction (left side parking pattern)
            nearby_cars_left = any(
                abs(car['center'][0] - (begin_gap / 2)) < median_w * 2 and 
                abs(car['center'][1] - first_car['center'][1]) < median_h
                for car in self.current_stationary_boxes if car != first_car
            )
            
            # Only create slot if gap is reasonable AND there's parking pattern evidence
            if (min_gap_horizontal < begin_gap < max_edge_distance and 
                (nearby_cars_left or begin_gap < median_w * 0.8)):  # Very small gaps are OK
                total_horizontal_slots += 1
                print(f"    Slot found: BEGIN gap={begin_gap:.0f}px (min={min_gap_horizontal:.0f}px, max={max_edge_distance:.0f}px)")
                
                # CONTROLLED BEGIN GAP: Match first car but limit size
                first_car_width = first_car['w']
                first_car_height = first_car['h']
                first_car_center_y = first_car['center'][1]
                
                # Height matches first car
                begin_y_min = int(first_car_center_y - first_car_height / 2)
                begin_y_max = int(first_car_center_y + first_car_height / 2)
                
                # Width control: Don't exceed first car width
                actual_begin_width = (begin_gap - parking_margin) - parking_margin
                if actual_begin_width > first_car_width:
                    # Center the green box within begin gap with width limit
                    begin_center_x = begin_gap / 2
                    controlled_begin_x_min = int(begin_center_x - first_car_width / 2)
                    controlled_begin_x_max = int(begin_center_x + first_car_width / 2)
                    gap_box = (max(parking_margin, controlled_begin_x_min), begin_y_min, controlled_begin_x_max, begin_y_max)
                else:
                    # Normal begin gap
                    gap_box = (parking_margin, begin_y_min, begin_gap - parking_margin, begin_y_max)
                
                grid.append({'box': gap_box, 'center': (begin_gap/2, first_car_center_y), 'is_vehicle': False, 'row': row_idx})

            # 2. Gaps between cars (split large gaps into multiple slots)
            for i in range(len(row_sorted) - 1):
                v1 = row_sorted[i]
                v2 = row_sorted[i+1]
                gap_start, gap_end = v1['box'][2], v2['box'][0]
                gap_width = gap_end - gap_start

                if gap_width > min_gap_horizontal:
                    # PRECISE NEIGHBOR SIZING: Use immediate neighbors' dimensions
                    left_car = v1
                    right_car = v2
                    
                    # Average the two immediate neighbors for slot dimensions
                    avg_neighbor_width = (left_car['w'] + right_car['w']) / 2
                    avg_neighbor_height = (left_car['h'] + right_car['h']) / 2
                    avg_neighbor_y = (left_car['center'][1] + right_car['center'][1]) / 2
                    
                    # SURGICAL SLOT CALCULATION: Precision space analysis
                    # Available usable space after accounting for margins
                    usable_gap = gap_width - (2 * parking_margin)
                    
                    # PRECISION: Each car needs width + safety margins + maneuvering space
                    safety_margin = parking_margin * 0.5  # Additional safety per car
                    space_per_car = avg_neighbor_width + (2 * parking_margin) + safety_margin
                    
                    # DISTANCE-ADAPTIVE SLOT CALCULATION
                    theoretical_slots = usable_gap / space_per_car
                    
                    # More lenient fit confidence for distant cars (smaller gaps)
                    if distance_factor < 0.5:  # Ultra-distant cars
                        fit_confidence_threshold = 0.7  # 70% confidence (more lenient)
                    elif distance_factor < 0.8:  # Distant cars
                        fit_confidence_threshold = 0.75  # 75% confidence
                    else:  # Close cars
                        fit_confidence_threshold = 0.85  # 85% confidence (strict)
                    
                    if theoretical_slots >= fit_confidence_threshold:
                        num_slots = max(1, int(theoretical_slots))
                        
                        # More lenient final verification for distant cars
                        if num_slots > 1:
                            space_for_last_slot = usable_gap - ((num_slots - 1) * space_per_car)
                            # More lenient minimum for distant cars
                            min_viable_ratio = 0.7 if distance_factor < 0.6 else 0.8  # 70% vs 80%
                            min_viable_slot = avg_neighbor_width * min_viable_ratio
                            
                            if space_for_last_slot < min_viable_slot:
                                num_slots -= 1  # Remove last slot if too cramped
                    else:
                        num_slots = 1  # Conservative: single slot only
                    
                    # EXACT RED BOX MATCHING: Green box = Red box size (no compromise!)
                    left_width, left_height = left_car['w'], left_car['h'] 
                    right_width, right_height = right_car['w'], right_car['h']
                    
                    # ABSOLUTE TARGET: Use LARGEST neighbor as reference (no smaller!)
                    target_width = max(left_width, right_width)
                    target_height = max(left_height, right_height)
                    
                    # Y position: Use the LARGEST car's Y for consistency
                    reference_car = left_car if left_car['h'] >= right_car['h'] else right_car
                    reference_y = reference_car['center'][1]
                    
                    # FIXED GREEN BOX DIMENSIONS: Exact match to target
                    green_box_width = target_width
                    green_box_height = target_height
                    
                    # PRECISE positioning: Center exactly like red box
                    precise_y_min = int(reference_y - target_height / 2)
                    precise_y_max = int(reference_y + target_height / 2)
                    
                    if num_slots == 1:
                        # Single slot - match immediate neighbors exactly
                        total_horizontal_slots += 1
                        print(f"    Slot found: gap={gap_width:.0f}px (min={min_gap_horizontal:.0f}px) - neighbor size")
                        # ABSOLUTE SIZING: Green box = EXACT target size (no exceptions!)
                        gap_center_x = (gap_start + gap_end) / 2
                        
                        # FORCE TARGET DIMENSIONS: Always use exact red box size
                        exact_x_min = int(gap_center_x - target_width / 2)
                        exact_x_max = int(gap_center_x + target_width / 2)
                        
                        # GREEN BOX: Exact red box dimensions, centered in gap
                        gap_box = (exact_x_min, precise_y_min, exact_x_max, precise_y_max)
                        
                        print(f"    GREEN BOX: {target_width}x{target_height}px at ({exact_x_min},{precise_y_min}) - EXACT red box match")
                        
                        grid.append({'box': gap_box, 'center': (gap_center_x, reference_y), 'is_vehicle': False, 'row': row_idx})
                    else:
                        # Multiple slots - each matches neighbor dimensions
                        slot_width = gap_width / num_slots
                        print(f"    Large gap: {gap_width:.0f}px -> {num_slots} slots ({slot_width:.0f}px each) - neighbor sized")
                        
                        for slot_idx in range(num_slots):
                            slot_start = gap_start + (slot_idx * slot_width)
                            slot_end = gap_start + ((slot_idx + 1) * slot_width)
                            
                            # EXACT MULTI-SLOT: Each slot = EXACT red box size
                            slot_center_x = (slot_start + slot_end) / 2
                            
                            # FORCE EXACT DIMENSIONS: No compromise for multi-slots either
                            exact_slot_x_min = int(slot_center_x - target_width / 2)
                            exact_slot_x_max = int(slot_center_x + target_width / 2)
                            
                            # Each GREEN BOX: Exact red box dimensions
                            slot_box = (exact_slot_x_min, precise_y_min, exact_slot_x_max, precise_y_max)
                            
                            print(f"    MULTI GREEN BOX {slot_idx+1}: {target_width}x{target_height}px at ({exact_slot_x_min},{precise_y_min}) - EXACT match")
                            
                            total_horizontal_slots += 1
                            grid.append({'box': slot_box, 'center': (slot_center_x, reference_y), 'is_vehicle': False, 'row': row_idx})

            # 3. ENHANCED End gap detection - especially for distant cars
            last_car = row_sorted[-1]
            end_gap = frame_w - last_car['box'][2]  # Distance to right edge
            
            # SMART end edge detection - avoid creating slots in empty areas
            max_end_edge_distance = median_w * 1.2 if distance_factor < 0.6 else median_w * 1.0  # Much stricter
            
            # ADDITIONAL CHECK: Only create end slot if it looks like actual parking space  
            # Check if there are other cars nearby in X direction (right side parking pattern)
            nearby_cars_right = any(
                abs(car['center'][0] - (last_car['box'][2] + end_gap / 2)) < median_w * 2 and 
                abs(car['center'][1] - last_car['center'][1]) < median_h
                for car in self.current_stationary_boxes if car != last_car
            )
            
            # Only create slot if gap is reasonable AND there's parking pattern evidence  
            if (min_gap_horizontal < end_gap < max_end_edge_distance and 
                (nearby_cars_right or end_gap < median_w * 0.8)):  # Very small gaps are OK
                total_horizontal_slots += 1
                print(f"    Slot found: END gap={end_gap:.0f}px (min={min_gap_horizontal:.0f}px)")
                
                # CONTROLLED END GAP: Match last car but limit size
                last_car_width = last_car['w']
                last_car_height = last_car['h']
                last_car_center_y = last_car['center'][1]
                
                # Height matches last car
                end_y_min = int(last_car_center_y - last_car_height / 2)
                end_y_max = int(last_car_center_y + last_car_height / 2)
                
                # Width control: Don't exceed last car width
                actual_end_width = (frame_w - 10 - parking_margin) - (last_car['box'][2] + parking_margin)
                if actual_end_width > last_car_width:
                    # Center the green box within end gap with width limit
                    end_gap_center_x = (last_car['box'][2] + frame_w - 10) / 2
                    controlled_end_x_min = int(end_gap_center_x - last_car_width / 2)
                    controlled_end_x_max = int(end_gap_center_x + last_car_width / 2)
                    gap_box = (controlled_end_x_min, end_y_min, min(frame_w - 10 - parking_margin, controlled_end_x_max), end_y_max)
                else:
                    # Normal end gap
                    gap_box = (last_car['box'][2] + parking_margin, end_y_min, frame_w - 10 - parking_margin, end_y_max)
                
                grid.append({'box': gap_box, 'center': ((last_car['box'][2] + frame_w)/2, last_car_center_y), 'is_vehicle': False, 'row': row_idx})

        # STEP 4: Find vertical gaps (between rows)
        total_vertical_slots = 0
        min_gap_vertical = median_h * 1.2  # 120% of car height for vertical gaps

        if len(rows) > 1:
            print(f"\nChecking vertical gaps between {len(rows)} rows...")
            
            for i in range(len(rows) - 1):
                row_top = rows[i]
                row_bottom = rows[i+1]
                
                # Get row boundaries
                top_row_bottom = max(v['box'][3] for v in row_top)
                bottom_row_top = min(v['box'][1] for v in row_bottom)
                gap_height = bottom_row_top - top_row_bottom
                
                if gap_height > min_gap_vertical:
                    # Find overlapping X regions between rows
                    top_x_ranges = [(v['box'][0], v['box'][2]) for v in row_top]
                    bottom_x_ranges = [(v['box'][0], v['box'][2]) for v in row_bottom]
                    
                    # Create a unified X coverage map
                    all_x_points = []
                    for x1, x2 in top_x_ranges + bottom_x_ranges:
                        all_x_points.extend([x1, x2])
                    
                    if all_x_points:
                        x_min, x_max = min(all_x_points), max(all_x_points)
                        
                        # Scan for gaps in X direction where vertical slots can fit
                        occupied_x_ranges = sorted(top_x_ranges + bottom_x_ranges)
                        
                        # Merge overlapping ranges
                        merged_ranges = []
                        for start, end in occupied_x_ranges:
                            if merged_ranges and start <= merged_ranges[-1][1] + median_w * 0.2:  # Small tolerance
                                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
                            else:
                                merged_ranges.append((start, end))
                        
                        # Find gaps between merged ranges
                        for j in range(len(merged_ranges) - 1):
                            gap_start = merged_ranges[j][1]
                            gap_end = merged_ranges[j+1][0]
                            gap_width = gap_end - gap_start
                            
                            if gap_width > median_w * 0.8:  # Wide enough for a car
                                total_vertical_slots += 1
                                print(f"    Vertical slot found: Row {i+1}-{i+2} gap={gap_width:.0f}px x {gap_height:.0f}px")
                                gap_box = (gap_start + 3, top_row_bottom + 3, gap_end - 3, bottom_row_top - 3)
                                # Note: vertical slots are between rows, so no single row index
                                grid.append({'box': gap_box, 'center': ((gap_start+gap_end)/2, (top_row_bottom+bottom_row_top)/2), 'is_vehicle': False, 'type': 'vertical', 'between_rows': [i, i+1]})
        
        # STEP 5: GUARANTEED LINES - Every car gets a line, no exceptions!
        lines_drawn = 0
        processed_cars = set()
        unique_depths = []  # Store each depth level
        
        # PHASE 1: PRECISION depth grouping - use SAME logic as row building
        for row in rows:  # Use the already-built rows instead of re-grouping
            if row:  # Skip empty rows
                unique_depths.append(row)
                for car in row:
                    processed_cars.add(id(car))
        
        # Check for any missed cars that weren't in rows
        for car in self.current_stationary_boxes:
            if id(car) not in processed_cars:
                # This car wasn't grouped, add as individual row
                unique_depths.append([car])
                processed_cars.add(id(car))
        
        # PHASE 2: Create LINES with duplicate prevention
        drawn_lines = []  # Track drawn line Y positions
        
        for depth_idx, depth_cars in enumerate(unique_depths):
            if len(depth_cars) >= 2:  # MINIMUM 2 cars required for line
                # Calculate precise Y level and X span
                all_y_coords = [car['center'][1] for car in depth_cars]
                all_x_coords = [car['center'][0] for car in depth_cars]
                
                depth_y_avg = np.mean(all_y_coords)
                min_x = min(all_x_coords)
                max_x = max(all_x_coords)
                
                # CHECK FOR DUPLICATE LINE: Skip if too close to existing line
                is_duplicate = False
                for existing_y in drawn_lines:
                    if abs(depth_y_avg - existing_y) < 15:  # 15px tolerance
                        is_duplicate = True
                        print(f"    SKIPPED DUPLICATE: Line at Y≈{depth_y_avg:.0f} too close to existing Y≈{existing_y:.0f}")
                        break
                
                if not is_duplicate:
                    # MAXIMUM EXTENSION: Line should span as much as possible
                    avg_car_width = np.mean([car.get('w', 80) for car in depth_cars])
                    extension = max(80, avg_car_width)
                    
                    # FULL SPAN LINE: Edge to edge with extensions
                    line_start_x = max(5, min_x - extension)
                    line_end_x = min(frame_w - 5, max_x + extension)
                    
                    # Create LINE for this depth (2+ cars)
                    grid.append({
                        'type': 'center_line',
                        'line': [(int(line_start_x), int(depth_y_avg)), (int(line_end_x), int(depth_y_avg))],
                        'row': depth_idx,
                        'color': (255, 0, 0)  # Blue line (BGR format)
                    })
                    lines_drawn += 1
                    drawn_lines.append(depth_y_avg)  # Track this line Y position
                    
                    print(f"    DEPTH LINE {depth_idx+1}: {len(depth_cars)} cars at Y≈{depth_y_avg:.0f}, span={line_end_x-line_start_x:.0f}px")
            elif len(depth_cars) == 1:
                # Single car: No line needed, just mark as processed
                print(f"    SINGLE CAR: Depth {depth_idx+1}, 1 car at Y≈{depth_cars[0]['center'][1]:.0f} (no line drawn)")
        
        # PHASE 3: BACKUP CHECK - Only for truly isolated cars that should be in a group
        # Skip backup lines since single cars don't need lines anymore
        isolated_cars = 0
        for car in self.current_stationary_boxes:
            car_in_group = False
            
            # Check if this car is in a group of 2+ cars
            for depth_cars in unique_depths:
                if len(depth_cars) >= 2:  # Only check groups with 2+ cars
                    for depth_car in depth_cars:
                        if abs(car['center'][1] - depth_car['center'][1]) <= 35:
                            car_in_group = True
                            break
                if car_in_group:
                    break
            
            if not car_in_group:
                isolated_cars += 1
        
        if isolated_cars > 0:
            print(f"    ISOLATED CARS: {isolated_cars} single cars (no lines needed)")
        
        # No backup lines created - single cars don't need lines
        
        print(f"    TOTAL CARS: {len(self.current_stationary_boxes)}, LINES DRAWN: {lines_drawn}")
        
        print(f"{'='*60}")
        print(f"TOTAL: {total_horizontal_slots} horizontal + {total_vertical_slots} vertical = {total_horizontal_slots + total_vertical_slots} parking slots")
        print(f"Added {lines_drawn} MAXIMUM LENGTH center lines for visualization")
        print(f"{'='*60}\n")
        return grid
    
    def _build_perfect_vertical_grid(self, frame_w, frame_h, median_w, median_h):
        grid = []
        columns = self._group_by_columns(self.current_stationary_boxes, median_w)
        
        for col in columns:
            col_sorted = sorted(col, key=lambda v: v['center'][1])
            
            for i, vehicle in enumerate(col_sorted):
                v_box = vehicle['box']
                grid.append({
                    'box': v_box,
                    'center': vehicle['center'],
                    'is_vehicle': True
                })
        
        return grid

    def _group_by_rows(self, vehicles, median_h):
        if not vehicles:
            return []
        
        rows = []
        sorted_by_y = sorted(vehicles, key=lambda v: v['center'][1])
        current_row = [sorted_by_y[0]]
        
        for vehicle in sorted_by_y[1:]:
            # Use 0.3x median_h or max 30px - very precise line separation
            threshold = min(median_h * 0.3, 30)
            if abs(vehicle['center'][1] - current_row[-1]['center'][1]) < threshold:
                current_row.append(vehicle)
            else:
                rows.append(current_row)
                current_row = [vehicle]
        rows.append(current_row)
        
        return rows

    def _group_by_columns(self, vehicles, median_w):
        if not vehicles:
            return []
        
        columns = []
        sorted_by_x = sorted(vehicles, key=lambda v: v['center'][0])
        current_col = [sorted_by_x[0]]
        
        for vehicle in sorted_by_x[1:]:
            # CRITICAL: Use last vehicle, 0.7x threshold
            if abs(vehicle['center'][0] - current_col[-1]['center'][0]) < median_w * 0.7:
                current_col.append(vehicle)
            else:
                columns.append(current_col)
                current_col = [vehicle]
        columns.append(current_col)
        
        return columns

    def _group_cars_by_center_lines(self, median_w):
        if not self.current_stationary_boxes:
            return []
        
        if len(self.current_stationary_boxes) == 1:
            return [self.current_stationary_boxes]
        
        cars = self.current_stationary_boxes.copy()
        
        # Calculate adaptive tolerance based on distance and car sizes
        car_heights = [v['h'] for v in cars]
        median_height = np.median(car_heights)
        
        rows = []
        used = set()
        
        for i, car in enumerate(cars):
            if i in used:
                continue
                
            # Start new row with this car
            current_row = [car]
            used.add(i)
            car_y = car['center'][1]
            car_x = car['center'][0]
            car_height = car['h']
            
            # PHASE 1: Find potential candidates by Y proximity
            candidates = []
            for j, other_car in enumerate(cars):
                if j in used:
                    continue
                
                other_y = other_car['center'][1]
                other_height = other_car['h']
                
                # SURGICAL PRECISION: Adaptive tolerance based on exact car dimensions
                size_factor = min(car_height, other_height) / median_height
                base_tolerance = 25  # Precision base - reduced from 30
                
                # GENEROUS CALIBRATION for max line length (increased tolerances)
                if size_factor < 0.4:  # Ultra far cars (deepest background)
                    tolerance = base_tolerance * 2.4  # 60px - increased for longer lines
                elif size_factor < 0.6:  # Very far cars  
                    tolerance = base_tolerance * 2.0  # 50px - increased for longer lines
                elif size_factor < 0.8:  # Far cars
                    tolerance = base_tolerance * 1.6  # 40px - increased for longer lines
                elif size_factor > 1.3:  # Very close cars
                    tolerance = base_tolerance * 0.8  # 20px - slightly increased
                elif size_factor > 1.0:  # Close cars
                    tolerance = base_tolerance * 1.0  # 25px - increased
                else:  # Medium cars (0.8-1.0 range)
                    tolerance = base_tolerance * 1.2  # 30px - increased
                
                y_diff = abs(car_y - other_y)
                
                if y_diff <= tolerance:
                    candidates.append((j, other_car))
            
            # PHASE 2: VERY LENIENT linearity check for max line length
            for j, candidate_car in candidates:
                # Prioritize max line length over perfect straightness
                test_row = current_row + [candidate_car]
                
                # Much more lenient criteria for longer lines
                if len(test_row) <= 3:  # Up to 3 cars: always accept
                    current_row.append(candidate_car)
                    used.add(j)
                elif self._is_reasonable_horizontal_line(test_row):  # 4+ cars: reasonable check
                    current_row.append(candidate_car)
                    used.add(j)
                else:
                    # Even if not perfectly straight, add if Y difference is small
                    y_diff = abs(current_row[-1]['center'][1] - candidate_car['center'][1])
                    if y_diff <= 50:  # 50px tolerance (increased from 40px)
                        current_row.append(candidate_car)
                        used.add(j)
            
            # Only keep rows with reasonable number of cars
            if len(current_row) >= 1:
                rows.append(current_row)
        
        # Sort rows by Y position (top to bottom)
        rows.sort(key=lambda row: np.mean([v['center'][1] for v in row]))
        
        # RELAXED CLEANUP: Allow more rows for line drawing
        clean_rows = []
        for row in rows:
            if len(row) >= 2:
                # More lenient linearity check for line drawing
                if self._is_reasonable_horizontal_line(row):  # More permissive function
                    clean_rows.append(row)
                else:
                    print(f"    Rejected extremely diagonal row with {len(row)} cars")
                    # Still add single cars from rejected rows for line drawing
                    for car in row:
                        clean_rows.append([car])
            else:
                clean_rows.append(row)  # Single cars are always OK
        
        rows = clean_rows
        
        # CONSERVATIVE MERGING: Prevent wrong line merging
        merged_rows = []
        for row in rows:
            if not merged_rows:
                merged_rows.append(row)
                continue
                
            last_row_y = np.mean([v['center'][1] for v in merged_rows[-1]])
            current_row_y = np.mean([v['center'][1] for v in row])
            
            # VERY STRICT MERGE: Prevent accidental line merging
            last_row_heights = [v['h'] for v in merged_rows[-1]]
            current_row_heights = [v['h'] for v in row]
            avg_height = (np.mean(last_row_heights) + np.mean(current_row_heights)) / 2
            
            # AGGRESSIVE MERGE RULE: Always prioritize long lines over precision
            merge_threshold = max(20, avg_height * 0.8)  # 80% tolerance - very aggressive!
            
            # ULTRA-RELAXED size compatibility - almost always merge
            size_ratio = np.mean(current_row_heights) / np.mean(last_row_heights)
            size_compatible = 0.4 <= size_ratio <= 2.5  # EXTREME range - almost any size works
            
            # AGGRESSIVE X-adjacency check - primary merge criteria
            last_row_x_coords = [v['center'][0] for v in merged_rows[-1]]
            current_row_x_coords = [v['center'][0] for v in row]
            
            last_row_x_min = min(last_row_x_coords)
            last_row_x_max = max(last_row_x_coords) 
            current_row_x_min = min(current_row_x_coords)
            current_row_x_max = max(current_row_x_coords)
            
            # Check X overlap or proximity
            x_overlap = (min(last_row_x_max, current_row_x_max) - max(last_row_x_min, current_row_x_min)) > 0
            x_gap = min(abs(current_row_x_min - last_row_x_max), abs(last_row_x_min - current_row_x_max))
            x_proximity = x_gap < median_w * 3  # Within 3 car widths
            
            # FORCE MERGE CONDITIONS (very aggressive)
            force_merge = (
                x_overlap or  # Any X overlap = merge
                x_proximity or  # Close in X = merge  
                (len(merged_rows[-1]) <= 3 and len(row) <= 3) or  # Small rows = always merge
                abs(current_row_y - last_row_y) < 30  # Close Y = always merge
            )
            
            if (abs(current_row_y - last_row_y) < merge_threshold and 
                size_compatible and 
                force_merge):  # FORCE merge if any condition met
                merged_rows[-1].extend(row)
                # Determine which rule triggered the merge
                merge_reason = []
                if x_overlap: merge_reason.append("X-overlap")
                if x_proximity: merge_reason.append(f"X-proximity({x_gap:.0f}px)")
                if len(merged_rows[-1])-len(row) <= 3 and len(row) <= 3: merge_reason.append("small-rows")
                if abs(current_row_y - last_row_y) < 30: merge_reason.append("close-Y")
                
                print(f"    MERGED: {len(merged_rows[-1])-len(row)}+{len(row)} cars → {len(merged_rows[-1])} cars at Y≈{last_row_y:.0f} & Y≈{current_row_y:.0f} ({', '.join(merge_reason)})")
            else:
                merged_rows.append(row)
                print(f"    SEPARATE: {len(row)} cars at Y≈{current_row_y:.0f} kept separate (Y-diff={abs(current_row_y-last_row_y):.0f}px, size={size_ratio:.2f}, no-force-rule)")
        
        rows = merged_rows
        
        print(f"Adaptive center-line grouping: {len(rows)} rows found")
        for i, row in enumerate(rows):
            avg_y = np.mean([v['center'][1] for v in row])
            heights = [v['h'] for v in row]
            avg_height = np.mean(heights)
            distance_category = "Near" if avg_height > median_height * 1.1 else "Far" if avg_height < median_height * 0.8 else "Mid"
            print(f"  Row {i+1}: {len(row)} cars at Y≈{avg_y:.0f} ({distance_category} - h≈{avg_height:.0f})")
        
        return rows

    def _is_straight_horizontal_line(self, cars):
        if len(cars) < 2:
            return True  # Single car is always acceptable
            
        if len(cars) == 2:
            # For 2 cars, check if Y difference is reasonable vs X difference
            car1, car2 = cars[0], cars[1]
            x_diff = abs(car1['center'][0] - car2['center'][0])
            y_diff = abs(car1['center'][1] - car2['center'][1])
            
            # Line should be mostly horizontal: Y change should be much smaller than X change
            if x_diff == 0:  # Vertical alignment - not good for horizontal parking
                return False
            
            slope = y_diff / x_diff if x_diff > 0 else float('inf')
            # Allow small slope (up to 15 degrees ≈ 0.27 slope)
            return slope <= 0.3
        
        # For 3+ cars, use linear regression to check straightness
        x_coords = [car['center'][0] for car in cars]
        y_coords = [car['center'][1] for car in cars]
        
        # Sort by X to ensure proper line direction
        sorted_points = sorted(zip(x_coords, y_coords))
        x_coords = [p[0] for p in sorted_points]
        y_coords = [p[1] for p in sorted_points]
        
        # Calculate linear regression
        n = len(x_coords)
        if n < 3:
            return True
            
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        # Calculate slope and R-squared
        numerator = sum((x_coords[i] - x_mean) * (y_coords[i] - y_mean) for i in range(n))
        denominator = sum((x_coords[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:  # All cars have same X - vertical line
            return False
            
        slope = numerator / denominator
        
        # SURGICAL PRECISION: Ultra-strict slope analysis
        max_slope = 0.25  # Reduced from 0.3 - maximum ~14 degrees (more horizontal)
        if abs(slope) > max_slope:
            return False
            
        # PRECISION R-SQUARED: Enhanced linearity measurement
        y_pred = [y_mean + slope * (x_coords[i] - x_mean) for i in range(n)]
        ss_res = sum((y_coords[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y_coords[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
        
        # SURGICAL STANDARD: Higher linearity requirement (R² > 0.75)
        min_r_squared = 0.75  # Increased from 0.7 for better straight lines
        
        # Additional check: Maximum Y deviation from perfect line
        max_y_deviation = max(abs(y_coords[i] - y_pred[i]) for i in range(n))
        max_allowed_deviation = np.mean([car['h'] for car in cars if 'h' in car]) * 0.3  # 30% of average car height
        
        return r_squared > min_r_squared and max_y_deviation <= max_allowed_deviation

    def _is_reasonable_horizontal_line(self, cars):
        if len(cars) < 2:
            return True
            
        if len(cars) == 2:
            # For 2 cars, more generous slope tolerance
            car1, car2 = cars[0], cars[1]
            x_diff = abs(car1['center'][0] - car2['center'][0])
            y_diff = abs(car1['center'][1] - car2['center'][1])
            
            if x_diff == 0:
                return False
            
            slope = y_diff / x_diff if x_diff > 0 else float('inf')
            # More lenient: up to 25 degrees (≈ 0.47 slope)
            return slope <= 0.5
        
        # For 3+ cars, more lenient requirements
        x_coords = [car['center'][0] for car in cars]
        y_coords = [car['center'][1] for car in cars]
        
        # Sort by X
        sorted_points = sorted(zip(x_coords, y_coords))
        x_coords = [p[0] for p in sorted_points]
        y_coords = [p[1] for p in sorted_points]
        
        n = len(x_coords)
        if n < 3:
            return True
            
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        numerator = sum((x_coords[i] - x_mean) * (y_coords[i] - y_mean) for i in range(n))
        denominator = sum((x_coords[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return False
            
        slope = numerator / denominator
        
        # More lenient slope (up to 30 degrees ≈ 0.58)
        if abs(slope) > 0.6:
            return False
            
        # More lenient R-squared requirement
        y_pred = [y_mean + slope * (x_coords[i] - x_mean) for i in range(n)]
        ss_res = sum((y_coords[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y_coords[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
        
        # Much more lenient: R² > 0.5 (was 0.75)
        return r_squared > 0.5

    def update_grid_occupancy(self):
        if not self.parking_grid:
            return [], []
        
        occupied_spots = []
        empty_spots = []
        
        # Group current vehicles by their depth/row - MUST match grid creation logic!
        vehicle_rows = self._group_vehicles_by_depth()
        
        for slot in self.parking_grid:
            # Skip center lines - they don't have parking slots
            if slot.get('type') == 'center_line':
                continue
                
            slot_box = slot['box']
            has_vehicle = False
            
            # Only check vehicles in the same depth/row as this slot
            if slot.get('is_vehicle', False):
                # This is a parked car, always occupied
                occupied_spots.append(slot_box)
                continue
            
            # UNIVERSAL RULE: Every slot ONLY checks its own row's vehicles!
            slot_row = slot.get('row', -1)
            slot_type = slot.get('type', 'horizontal')
            
            # Get ONLY the vehicles that can actually occupy this slot
            relevant_vehicles = []
            
            if slot_type == 'vertical':
                # Vertical slots: between two specific rows
                between_rows = slot.get('between_rows', [])
                if between_rows and len(between_rows) == 2:
                    # Only vehicles from the two adjacent rows can occupy this
                    for row_idx in between_rows:
                        if 0 <= row_idx < len(vehicle_rows):
                            relevant_vehicles.extend(vehicle_rows[row_idx])
                            
            elif slot_row >= 0 and slot_row < len(vehicle_rows):
                # HORIZONTAL SLOTS: STRICTLY ONLY SAME ROW!
                # This is the UNIVERSAL RULE - no other row can interfere
                relevant_vehicles = vehicle_rows[slot_row]
                
                
            else:
                # Fallback: No row info means grid wasn't built properly
                # Use ULTRA-strict Y matching to prevent any cross-row issues
                slot_center_y = (slot_box[1] + slot_box[3]) / 2
                
                for vehicle in self.current_stationary_boxes:
                    v_center_y = vehicle['center'][1]
                    v_height = vehicle.get('h', 50)  # Default height if missing
                    
                    # ADAPTIVE RULE: Smaller cars (farther) get more tolerance
                    if v_height < 40:  # Very far/small cars
                        tolerance = 60  # Much more generous
                    elif v_height < 60:  # Far cars
                        tolerance = 40
                    else:  # Close cars
                        tolerance = 20
                    
                    if abs(v_center_y - slot_center_y) < tolerance:
                        relevant_vehicles.append(vehicle)
            
            # SURGICAL OCCUPANCY ANALYSIS: Multi-layer precision detection
            for vehicle in relevant_vehicles:
                v_box = vehicle['box']
                v_center = vehicle['center']
                v_width = vehicle.get('w', v_box[2] - v_box[0])
                v_height = vehicle.get('h', v_box[3] - v_box[1])
                
                # LAYER 1: Precision center analysis
                slot_center_x = (slot_box[0] + slot_box[2]) / 2
                slot_center_y = (slot_box[1] + slot_box[3]) / 2
                slot_width = slot_box[2] - slot_box[0]
                slot_height = slot_box[3] - slot_box[1]
                
                # Distance from vehicle center to slot center
                center_distance = np.sqrt((v_center[0] - slot_center_x)**2 + (v_center[1] - slot_center_y)**2)
                max_allowed_distance = min(slot_width, slot_height) * 0.4  # 40% of smaller dimension
                
                # LAYER 2: Dimensional compatibility check
                size_compatibility = (v_width <= slot_width * 1.2) and (v_height <= slot_height * 1.2)
                
                # LAYER 3: Precise overlap calculation
                overlap = self._boxes_overlap(slot_box, v_box)
                
                # SURGICAL DECISION: Multi-criteria occupancy determination
                criteria_met = 0
                
                # Criterion 1: Center proximity (30% weight)
                if center_distance <= max_allowed_distance:
                    criteria_met += 30
                    
                # Criterion 2: Size compatibility (20% weight) 
                if size_compatibility:
                    criteria_met += 20
                    
                # Criterion 3: Overlap percentage (50% weight)
                if overlap > 0.35:  # Precision threshold
                    criteria_met += 50
                elif overlap > 0.25:  # Partial overlap
                    criteria_met += 25
                    
                # SURGICAL THRESHOLD: 60% confidence required for occupation
                if criteria_met >= 60:
                    has_vehicle = True
                    break
            
            if has_vehicle:
                occupied_spots.append(slot_box)
            else:
                empty_spots.append(slot_box)
        
        return occupied_spots, empty_spots

    def _group_vehicles_by_depth(self):
        # CRITICAL: Use IDENTICAL logic to _group_cars_by_center_lines()
        # Calculate median_w from current vehicles
        widths = [v['w'] for v in self.current_stationary_boxes]
        median_w = int(np.percentile(widths, 75)) if widths else 80
        return self._group_cars_by_center_lines(median_w)

    def _boxes_overlap(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        if box1_area == 0:
            return 0.0
        
        return intersection_area / box1_area

    def find_parking_spots(self, frame_w, frame_h):
        current_time = time.time()
        
        # Initial grid build
        if not self.grid_established and len(self.current_stationary_boxes) >= 1:
            self.build_parking_grid(frame_w, frame_h)
            self.last_grid_rebuild_time = current_time
        
        # Periodic grid rebuild (every 10 seconds)
        if self.grid_established:
            time_since_rebuild = current_time - self.last_grid_rebuild_time
            
            if time_since_rebuild >= self.grid_rebuild_interval:
                # Rebuild grid to detect new/removed vehicles
                if len(self.current_stationary_boxes) >= 1:
                    self.build_parking_grid(frame_w, frame_h)
                    self.last_grid_rebuild_time = current_time
            
            # Always update occupancy (fast operation)
            return self.update_grid_occupancy()
        else:
            # Not ready yet
            if len(self.current_stationary_boxes) > 0:
                occupied = [v['box'] for v in self.current_stationary_boxes]
                return occupied, []
            return [], []

    def check_initial_phase(self):
        if self.is_initial_phase:
            elapsed_time = time.time() - self.initial_phase_start_time
            if elapsed_time > self.initial_phase_seconds:
                self.is_initial_phase = False
                self.stage = 2
                pass
        return self.is_initial_phase
