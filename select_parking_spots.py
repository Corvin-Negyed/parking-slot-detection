#!/usr/bin/env python3
"""
Parking spot selection tool
Click to define parking spot corners (4 points per spot)
Press 's' to save, 'r' to reset, 'q' to quit
"""

import cv2
import json
import sys

class ParkingSpotSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Cannot open image: {image_path}")
            sys.exit(1)
        
        self.display = self.image.copy()
        self.spots = []
        self.current_spot = []
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_spot.append([x, y])
            
            # Draw point
            cv2.circle(self.display, (x, y), 3, (0, 255, 0), -1)
            
            # Draw lines between points
            if len(self.current_spot) > 1:
                cv2.line(self.display, 
                        tuple(self.current_spot[-2]), 
                        tuple(self.current_spot[-1]), 
                        (0, 255, 0), 2)
            
            # Complete spot (4 corners)
            if len(self.current_spot) == 4:
                cv2.line(self.display, 
                        tuple(self.current_spot[-1]), 
                        tuple(self.current_spot[0]), 
                        (0, 255, 0), 2)
                
                self.spots.append(self.current_spot.copy())
                self.current_spot = []
                
                # Draw spot number
                if self.spots:
                    center_x = sum(p[0] for p in self.spots[-1]) // 4
                    center_y = sum(p[1] for p in self.spots[-1]) // 4
                    cv2.putText(self.display, str(len(self.spots)), 
                               (center_x, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def run(self):
        cv2.namedWindow('Parking Spot Selector')
        cv2.setMouseCallback('Parking Spot Selector', self.mouse_callback)
        
        print("Instructions:")
        print("- Click 4 points to define each parking spot")
        print("- Press 's' to save spots to parking_spots.json")
        print("- Press 'r' to reset")
        print("- Press 'q' to quit")
        
        while True:
            cv2.imshow('Parking Spot Selector', self.display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.display = self.image.copy()
                self.spots = []
                self.current_spot = []
                print("Reset!")
            elif key == ord('s'):
                self.save_spots()
        
        cv2.destroyAllWindows()
    
    def save_spots(self):
        if not self.spots:
            print("No spots to save!")
            return
        
        data = {
            'image': self.image_path,
            'spots': self.spots,
            'total': len(self.spots)
        }
        
        with open('parking_spots.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.spots)} parking spots to parking_spots.json")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python select_parking_spots.py <image_or_video_path>")
        print("Example: python select_parking_spots.py parking_lot.jpg")
        sys.exit(1)
    
    selector = ParkingSpotSelector(sys.argv[1])
    selector.run()

