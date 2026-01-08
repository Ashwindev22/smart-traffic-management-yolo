import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define vehicle categories
small_vehicles = {'car', 'motorbike'}
large_vehicles = {'bus', 'truck'}
emergency_vehicles = {'ambulance', 'fire truck'}

# Video files
input_video_1 = 'videoplayback1.MP4'
input_video_2 = 'videoplayback2.MP4'

# Open videos
cap1 = cv2.VideoCapture(input_video_1)
cap2 = cv2.VideoCapture(input_video_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both videos.")
    exit()

# Traffic signal settings
current_green = 1  # Start with video 1

# Function to add traffic signal and vehicle count
def add_traffic_info(frame, red_light, green_light, small_count, large_count, emergency_count):
    # Draw traffic signal in the top left
    cv2.rectangle(frame, (0, 0), (140, 65), (255, 255, 255), cv2.FILLED)
    
    cv2.circle(frame, (26, 30), 16, (0, 0, 255), 2)  # Red light
    cv2.circle(frame, (96, 30), 16, (0, 255, 0), 2)  # Green light

    if red_light:
        cv2.circle(frame, (26, 30), 16, (0, 0, 255), cv2.FILLED)
    if green_light:
        cv2.circle(frame, (96, 30), 16, (0, 255, 0), cv2.FILLED)
    
    return frame

def add_vehicle_count(frame, small_count, large_count, emergency_count):
    # Draw vehicle count in the bottom left
    frame_h = frame.shape[0]
    cv2.rectangle(frame, (0, frame_h - 50), (180, frame_h), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, f'Small: {small_count}', (10, frame_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, f'Large: {large_count}', (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, f'Emergency: {emergency_count}', (10, frame_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

while cap1.isOpened() or cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break

    small_count_1, large_count_1, emergency_count_1 = 0, 0, 0
    small_count_2, large_count_2, emergency_count_2 = 0, 0, 0

    # Process frame 1
    results1 = model(frame1)
    for box in results1[0].boxes:
        class_name = results1[0].names[int(box.cls[0].item())]
        if class_name in small_vehicles:
            small_count_1 += 1
            color = (255, 255, 0)
        elif class_name in large_vehicles:
            large_count_1 += 1
            color = (0, 255, 255)
        elif class_name in emergency_vehicles:
            emergency_count_1 += 1
            color = (0, 0, 255)
        else:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)

    # Process frame 2
    results2 = model(frame2)
    for box in results2[0].boxes:
        class_name = results2[0].names[int(box.cls[0].item())]
        if class_name in small_vehicles:
            small_count_2 += 1
            color = (255, 255, 0)
        elif class_name in large_vehicles:
            large_count_2 += 1
            color = (0, 255, 255)
        elif class_name in emergency_vehicles:
            emergency_count_2 += 1
            color = (0, 0, 255)
        else:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)

    # Determine which video gets the green light
    total_vehicles_1 = small_count_1 + large_count_1 + emergency_count_1
    total_vehicles_2 = small_count_2 + large_count_2 + emergency_count_2
    current_green = 1 if total_vehicles_1 >= total_vehicles_2 else 2

    # Add traffic info
    frame1 = add_traffic_info(frame1, current_green == 2, current_green == 1, small_count_1, large_count_1, emergency_count_1)
    frame2 = add_traffic_info(frame2, current_green == 1, current_green == 2, small_count_2, large_count_2, emergency_count_2)

    # Add vehicle count
    frame1 = add_vehicle_count(frame1, small_count_1, large_count_1, emergency_count_1)
    frame2 = add_vehicle_count(frame2, small_count_2, large_count_2, emergency_count_2)
    
    # Display frames
    cv2.imshow('Video 1', frame1)
    cv2.imshow('Video 2', frame2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Processing complete.")
