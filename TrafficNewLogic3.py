import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 models
vehicle_model = YOLO("yolov8n.pt")
ambulance_model = YOLO("best.pt")  # Specialized model for ambulance detection

# Define vehicle categories
small_vehicles = {'car', 'motorbike'}
large_vehicles = {'bus', 'truck'}
emergency_vehicles = {'Ambulance', 'fire truck'}

# Video files
input_video_1 = 'traffic1.mp4'
input_video_2 = 'ambulance_traffic.mp4'

# Traffic signal parameters
BASE_GREEN_TIME = 15   # seconds
MIN_GREEN_TIME = 5     # seconds
MAX_GREEN_TIME = 20    # seconds
YELLOW_TIME = 2        # seconds
VEHICLE_THRESHOLD = 25  # Minimum vehicles for full green time
AMBULANCE_CONFIDENCE_THRESHOLD = .94  # 90% confidence threshold

# Open videos
cap1 = cv2.VideoCapture(input_video_1)
cap2 = cv2.VideoCapture(input_video_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both videos.")
    exit()

# Get video properties
fps = int(cap1.get(cv2.CAP_PROP_FPS))
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter('output_video_1.mp4', fourcc, fps, (frame_width, frame_height))
out2 = cv2.VideoWriter('output_video_2.mp4', fourcc, fps, (frame_width, frame_height))

# Traffic signal state
class TrafficSignal:
    def __init__(self):
        self.current_green = 1  # 1 or 2
        self.green_timer = BASE_GREEN_TIME
        self.yellow_timer = 0
        self.signal_state = 'green'  # 'green', 'yellow', or 'red'
        self.last_switch = time.time()
        self.added_time = 0

signal = TrafficSignal()

def calculate_green_time(vehicle_count):
    """Calculate green time based on vehicle count"""
    if vehicle_count > VEHICLE_THRESHOLD:
        return BASE_GREEN_TIME
    
    # Linear reduction between threshold and min time
    reduced_time = MIN_GREEN_TIME + (BASE_GREEN_TIME - MIN_GREEN_TIME) * (vehicle_count / VEHICLE_THRESHOLD)
    return max(MIN_GREEN_TIME, min(round(reduced_time), MAX_GREEN_TIME))

def update_signal_state():
    """Update traffic signal state based on timer and vehicle presence"""
    elapsed = time.time() - signal.last_switch
    
    if signal.signal_state == 'green':
        current_count = (small_count_1 + large_count_1 + emergency_count_1 
                       if signal.current_green == 1 
                       else small_count_2 + large_count_2 + emergency_count_2)
        
        # Switch immediately if no vehicles detected in current green lane
        if current_count == 0:
            signal.signal_state = 'yellow'
            signal.yellow_timer = YELLOW_TIME
            signal.last_switch = time.time()
        elif elapsed >= signal.green_timer:
            signal.signal_state = 'yellow'
            signal.yellow_timer = YELLOW_TIME
            signal.last_switch = time.time()
    
    elif signal.signal_state == 'yellow':
        if elapsed >= signal.yellow_timer:
            # Switch to other lane
            signal.current_green = 2 if signal.current_green == 1 else 1
            signal.signal_state = 'green'
            
            # Calculate new green time based on vehicle count
            current_count = (small_count_1 + large_count_1 + emergency_count_1 
                           if signal.current_green == 1 
                           else small_count_2 + large_count_2 + emergency_count_2)
            
            new_green_time = calculate_green_time(current_count)
            signal.added_time = new_green_time - BASE_GREEN_TIME
            signal.green_timer = new_green_time
            signal.last_switch = time.time()

def add_traffic_info(frame, is_green, is_yellow=False):
    """Add traffic signal visualization to frame"""
    # Traffic light background (top-left corner)
    cv2.rectangle(frame, (10, 10), (100, 160), (50, 50, 50), cv2.FILLED)
    
    # Red light (always shown)
    color = (0, 0, 255) if not is_green and not is_yellow else (50, 50, 50)
    cv2.circle(frame, (55, 45), 20, color, -1)
    cv2.circle(frame, (55, 45), 20, (200, 200, 200), 2)
    
    # Yellow light
    color = (0, 255, 255) if is_yellow else (50, 50, 50)
    cv2.circle(frame, (55, 85), 20, color, -1)
    cv2.circle(frame, (55, 85), 20, (200, 200, 200), 2)
    
    # Green light
    color = (0, 255, 0) if is_green else (50, 50, 50)
    cv2.circle(frame, (55, 125), 20, color, -1)
    cv2.circle(frame, (55, 125), 20, (200, 200, 200), 2)
    
    # Timer display (below traffic light)
    remaining_time = max(0, signal.green_timer - (time.time() - signal.last_switch)) if is_green else 0
    remaining_time = max(0, signal.yellow_timer - (time.time() - signal.last_switch)) if is_yellow else remaining_time
    timer_text = f"{int(remaining_time)}s"
    cv2.putText(frame, timer_text, (45, 160), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    
    # Time adjustment info (right side of traffic light)
    if signal.added_time != 0 and is_green:
        adjust_text = f"{'+' if signal.added_time > 0 else ''}{signal.added_time}s"
        color = (0, 255, 0) if signal.added_time > 0 else (0, 0, 255)
        cv2.putText(frame, adjust_text, (110, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    
    return frame

def add_vehicle_count(frame, small_count, large_count, emergency_count):
    """Add vehicle count information to frame (top-right corner)"""
    # Background rectangle
    cv2.rectangle(frame, (frame.shape[1] - 220, 10), (frame.shape[1] - 10, 110), (50, 50, 50), cv2.FILLED)
    
    # Title
    cv2.putText(frame, "VEHICLE COUNT", (frame.shape[1] - 210, 30), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    # Vehicle counts
    cv2.putText(frame, f'Small: {small_count}', (frame.shape[1] - 210, 60), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f'Large: {large_count}', (frame.shape[1] - 210, 85), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f'Emergency: {emergency_count}', (frame.shape[1] - 210, 110), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def detect_vehicles(frame, model, ambulance_check=False):
    """Detect vehicles in frame and return counts"""
    small_count = 0
    large_count = 0
    emergency_count = 0
    ambulance_detected = False
    
    # Run detection
    results = model(frame)
    
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls[0].item())]
        confidence = box.conf[0].item()
        
        # Skip if confidence is too low for ambulance detection
        if ambulance_check and confidence < AMBULANCE_CONFIDENCE_THRESHOLD:
            continue
            
        # Emergency vehicle detection (priority)
        if ambulance_check and class_name == 'Ambulance':
            emergency_count += 1
            ambulance_detected = True
            color = (0, 0, 255)  # Red for emergency
        elif class_name in small_vehicles:
            small_count += 1
            color = (255, 255, 0)  # Cyan for small
        elif class_name in large_vehicles:
            large_count += 1
            color = (0, 255, 255)  # Yellow for large
        elif class_name in emergency_vehicles:
            emergency_count += 1
            color = (0, 0, 255)  # Red for emergency
        else:
            continue
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label with class and confidence
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame, small_count, large_count, emergency_count, ambulance_detected

# Main processing loop
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break

    # Detect vehicles in both frames
    frame1, small_count_1, large_count_1, emergency_count_1, ambulance_detected_1 = detect_vehicles(
        frame1, vehicle_model)
    
    frame2, small_count_2, large_count_2, emergency_count_2, ambulance_detected_2 = detect_vehicles(
        frame2, vehicle_model)
    
    # Additional ambulance check with specialized model (90% confidence threshold)
    _, _, _, additional_emergency_1, additional_ambulance_1 = detect_vehicles(
        frame1, ambulance_model, ambulance_check=True)
    
    _, _, _, additional_emergency_2, additional_ambulance_2 = detect_vehicles(
        frame2, ambulance_model, ambulance_check=True)
    
    emergency_count_1 += additional_emergency_1
    emergency_count_2 += additional_emergency_2
    ambulance_detected_1 = ambulance_detected_1 or additional_ambulance_1
    ambulance_detected_2 = ambulance_detected_2 or additional_ambulance_2

    # Check for emergency priority (only if high confidence detection)
    if ambulance_detected_1 or ambulance_detected_2:
        if ambulance_detected_1 and not ambulance_detected_2:
            signal.current_green = 1
            signal.signal_state = 'green'
            signal.green_timer = MAX_GREEN_TIME  # Extended time for emergency
            signal.last_switch = time.time()
            signal.added_time = MAX_GREEN_TIME - BASE_GREEN_TIME
        elif ambulance_detected_2 and not ambulance_detected_1:
            signal.current_green = 2
            signal.signal_state = 'green'
            signal.green_timer = MAX_GREEN_TIME  # Extended time for emergency
            signal.last_switch = time.time()
            signal.added_time = MAX_GREEN_TIME - BASE_GREEN_TIME
    
    # Update signal state
    update_signal_state()
    
    # Determine which lane has green/yellow light
    is_green_1 = signal.current_green == 1 and signal.signal_state == 'green'
    is_yellow_1 = signal.current_green == 1 and signal.signal_state == 'yellow'
    is_green_2 = signal.current_green == 2 and signal.signal_state == 'green'
    is_yellow_2 = signal.current_green == 2 and signal.signal_state == 'yellow'
    
    # Add traffic info to frames
    frame1 = add_traffic_info(frame1, is_green_1, is_yellow_1)
    frame2 = add_traffic_info(frame2, is_green_2, is_yellow_2)
    
    # Add vehicle counts (positioned at top-right)
    frame1 = add_vehicle_count(frame1, small_count_1, large_count_1, emergency_count_1)
    frame2 = add_vehicle_count(frame2, small_count_2, large_count_2, emergency_count_2)
    
    # Save frames to video
    out1.write(frame1)
    out2.write(frame2)
    
    # Display frames side by side
    combined_frame = np.hstack((frame1, frame2))
    cv2.imshow('Traffic Control System', combined_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
print("Processing complete. Videos saved as output_video_1.mp4 and output_video_2.mp4.")