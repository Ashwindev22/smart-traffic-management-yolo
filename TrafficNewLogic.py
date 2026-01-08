import cv2
import time
from ultralytics import YOLO
import threading

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define input video files
video1_path = "videoplayback1.mp4"
video2_path = "videoplayback2.mp4"

# Open video streams
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Traffic signal settings
PREDEFINED_GREEN_TIME = 5  # Default green time in seconds
VEHICLE_THRESHOLD = 8  # Minimum vehicle count for full green time
REDUCTION_FACTOR = 0.5  # Fraction of time reduced for lower vehicle count

def count_vehicles(frame):
    results = model(frame)
    vehicle_count = 0
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:  # Car, Motorcycle, Bus, Truck
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vehicle_count, frame

def process_lane(cap, lane_name, display_x, display_y):
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        return False  # End of video
    
    vehicle_count, frame = count_vehicles(frame)
    
    # Adjust green time
    if vehicle_count < VEHICLE_THRESHOLD:
        green_time = PREDEFINED_GREEN_TIME * (1 - REDUCTION_FACTOR)
    else:
        green_time = PREDEFINED_GREEN_TIME
    
    print(f"{lane_name} - Vehicles: {vehicle_count}, Green Time: {green_time:.1f}s")
    
    while time.time() - start_time < green_time:
        ret, frame = cap.read()
        if not ret:
            break
        vehicle_count, frame = count_vehicles(frame)
        
        # Display info
        cv2.putText(frame, f"{lane_name} - Green ({green_time:.1f}s left)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show video side by side
        frame_resized = cv2.resize(frame, (640, 480))
        cv2.imshow(lane_name, frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True

def run_traffic_system():
    while True:
        t1 = threading.Thread(target=process_lane, args=(cap1, "Lane 1", 0, 0))
        t2 = threading.Thread(target=process_lane, args=(cap2, "Lane 2", 640, 0))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        if not cap1.isOpened() or not cap2.isOpened():
            break

run_traffic_system()

cap1.release()
cap2.release()
cv2.destroyAllWindows()
