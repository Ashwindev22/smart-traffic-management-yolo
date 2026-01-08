import cv2
import time
import numpy as np
from ultralytics import YOLO
from threading import Thread

class TrafficSignalController:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        
        # Vehicle class IDs in COCO dataset (car, truck, bus, motorcycle)
        self.vehicle_classes = [2, 3, 5, 7]
        
        # Traffic signal parameters
        self.base_green_time = 10  # seconds (default green time)
        self.min_green_time = 5    # minimum green time
        self.max_green_time = 30   # maximum green time
        self.yellow_time = 3       # yellow signal time
        self.threshold = 5         # vehicle count threshold for time adjustment
        
        # Lane information
        self.lanes = {
            'lane1': {
                'video_source': 'videoplayback1.mp4',  # replace with your video source
                'current_count': 0,
                'signal': 'red',
                'timer': 0,
                'added_time': 0,
                'frame': None,
                'processed_frame': None
            },
            'lane2': {
                'video_source': 'videoplayback2.mp4',  # replace with your video source
                'current_count': 0,
                'signal': 'red',
                'timer': 0,
                'added_time': 0,
                'frame': None,
                'processed_frame': None
            }
        }
        
        # Start with lane1 getting green signal
        self.current_green = 'lane1'
        self.lanes['lane1']['signal'] = 'green'
        self.lanes['lane1']['timer'] = self.base_green_time
        self.lanes['lane1']['added_time'] = 0
        
        # Display parameters
        self.display_width = 1280
        self.display_height = 480
        
        # Thread control
        self.running = True
    
    def detect_vehicles(self, lane_key):
        """Detect vehicles in the specified lane using YOLOv8 and draw bounding boxes"""
        cap = cv2.VideoCapture(self.lanes[lane_key]['video_source'])
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Store original frame
            self.lanes[lane_key]['frame'] = frame.copy()
            
            # Run vehicle detection
            results = self.model(frame)
            
            # Count vehicles and draw bounding boxes
            vehicle_count = 0
            annotated_frame = frame.copy()
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in self.vehicle_classes:
                        vehicle_count += 1
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Display class and confidence
                        label = f"{self.model.names[int(box.cls)]} {box.conf[0]:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
            # Update lane count and processed frame
            self.lanes[lane_key]['current_count'] = vehicle_count
            self.lanes[lane_key]['processed_frame'] = self.add_overlay(annotated_frame, lane_key)
            
            # Add some delay to reduce processing load
            time.sleep(0.05)
        
        cap.release()
    
    def add_overlay(self, frame, lane_key):
        """Add traffic control information overlay to the frame"""
        lane = self.lanes[lane_key]
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Lane title and status
        title = f"{lane_key.upper()} - Signal: {lane['signal'].upper()}"
        cv2.putText(frame, title, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        
        # Vehicle count
        count_text = f"Vehicles: {lane['current_count']}"
        cv2.putText(frame, count_text, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        
        # Timer
        timer_text = f"Time left: {int(lane['timer'])}s"
        cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        
        # Time adjustment if applicable
        if lane['added_time'] != 0:
            adjust_text = f"Time {'+' if lane['added_time'] > 0 else ''}{lane['added_time']}s"
            adjust_color = (0, 255, 0) if lane['added_time'] > 0 else (0, 0, 255)
            cv2.putText(frame, adjust_text, (w - 120, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, adjust_color, 1)
        
        # Current signal indicator
        signal_color = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0)
        }.get(lane['signal'], (255, 255, 255))
        
        cv2.circle(frame, (w - 30, 30), 15, signal_color, -1)
        cv2.circle(frame, (w - 30, 30), 15, (255, 255, 255), 2)
        
        return frame
    
    def calculate_green_time(self, vehicle_count):
        """
        Calculate green time based on vehicle count
        - If count > threshold: base time
        - If count <= threshold: reduced time proportionally
        """
        if vehicle_count > self.threshold:
            return self.base_green_time
        
        # Linear reduction: time = min_time + (base-min)*(count/threshold)
        reduced_time = self.min_green_time + (self.base_green_time - self.min_green_time) * (vehicle_count / self.threshold)
        return max(self.min_green_time, min(round(reduced_time), self.max_green_time))
    
    def control_signals(self):
        """Control traffic signals based on vehicle counts"""
        while self.running:
            current_lane = self.current_green
            other_lane = 'lane2' if current_lane == 'lane1' else 'lane1'
            
            # Get current green time (may have been adjusted)
            green_time = self.lanes[current_lane]['timer']
            start_time = time.time()
            
            # Green signal for current lane
            self.lanes[current_lane]['signal'] = 'green'
            self.lanes[other_lane]['signal'] = 'red'
            
            # Display countdown
            while time.time() - start_time < green_time and self.running:
                remaining = max(0, green_time - (time.time() - start_time))
                self.lanes[current_lane]['timer'] = remaining
                time.sleep(0.1)
            
            if not self.running:
                break
            
            # Yellow signal transition
            self.lanes[current_lane]['signal'] = 'yellow'
            self.lanes[other_lane]['signal'] = 'yellow'
            time.sleep(self.yellow_time)
            
            # Switch to other lane
            self.current_green = other_lane
            
            # Calculate new green time based on vehicle count
            vehicle_count = self.lanes[other_lane]['current_count']
            new_green_time = self.calculate_green_time(vehicle_count)
            
            # Calculate how much time was added/removed
            time_change = new_green_time - self.base_green_time
            self.lanes[other_lane]['added_time'] = time_change
            self.lanes[other_lane]['timer'] = new_green_time
    
    def display_videos(self):
        """Display the processed videos side by side with all information"""
        cv2.namedWindow('Traffic Control System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Traffic Control System', self.display_width, self.display_height)
        
        while self.running:
            # Get processed frames (or black frames if not available yet)
            frame1 = self.lanes['lane1']['processed_frame'] if self.lanes['lane1']['processed_frame'] is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            frame2 = self.lanes['lane2']['processed_frame'] if self.lanes['lane2']['processed_frame'] is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Combine frames side by side
            combined_frame = np.hstack((frame1, frame2))
            
            # Add global status
            status_text = f"Current Green: {self.current_green.upper()}"
            cv2.putText(combined_frame, status_text, (self.display_width//2 - 100, 30), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Traffic Control System', combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        
        cv2.destroyAllWindows()
    
    def run(self):
        # Start vehicle detection threads
        lane1_thread = Thread(target=self.detect_vehicles, args=('lane1',))
        lane2_thread = Thread(target=self.detect_vehicles, args=('lane2',))
        
        # Start signal control thread
        control_thread = Thread(target=self.control_signals)
        
        lane1_thread.start()
        lane2_thread.start()
        control_thread.start()
        
        # Start video display in main thread
        self.display_videos()
        
        # Wait for threads to finish
        lane1_thread.join()
        lane2_thread.join()
        control_thread.join()

if __name__ == "__main__":
    controller = TrafficSignalController()
    controller.run()