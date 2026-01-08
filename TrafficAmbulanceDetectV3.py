import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
vehicle_model = YOLO("yolov8n.pt")
ambulance_model = YOLO("best.pt")  # Load the specialized model for ambulance detection

# Define vehicle categories
small_vehicles = {'car', 'motorbike'}
large_vehicles = {'bus', 'truck'}
emergency_vehicles = {'Ambulance', 'fire truck'}

# Video files
input_video_1 = 'traffic1.MP4'
input_video_2 = 'traffic2.mp4'

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

# Parameters
predefined_time = 10  # Default green light duration in seconds
threshold = 10  # Vehicle count threshold

green_time_remaining = predefined_time
current_video = 1  # Start with video 1
time_switched = time.time()

def add_traffic_info(frame, is_green, time_left):
    cv2.rectangle(frame, (0, 0), (140, 65), (255, 255, 255), cv2.FILLED)
    cv2.circle(frame, (26, 30), 16, (0, 0, 255), 2)  # Red light
    cv2.circle(frame, (96, 30), 16, (0, 255, 0), 2)  # Green light
    if is_green:
        cv2.circle(frame, (96, 30), 16, (0, 255, 0), cv2.FILLED)
    else:
        cv2.circle(frame, (26, 30), 16, (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, f'Time: {time_left:.1f}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break
    
    small_count_1 = large_count_1 = emergency_count_1 = 0
    small_count_2 = large_count_2 = emergency_count_2 = 0
    ambulance_detected_1 = ambulance_detected_2 = False

    ambulance_results1 = ambulance_model(frame1)
    for box in ambulance_results1[0].boxes:
        if ambulance_results1[0].names[int(box.cls[0].item())] == 'Ambulance':
            ambulance_detected_1 = True
            emergency_count_1 += 1
            cv2.putText(frame1, 'EMERGENCY', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            break
    
    ambulance_results2 = ambulance_model(frame2)
    for box in ambulance_results2[0].boxes:
        if ambulance_results2[0].names[int(box.cls[0].item())] == 'Ambulance':
            ambulance_detected_2 = True
            emergency_count_2 += 1
            cv2.putText(frame2, 'EMERGENCY', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            break

    results1 = vehicle_model(frame1)
    for box in results1[0].boxes:
        class_name = results1[0].names[int(box.cls[0].item())]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = (255, 255, 0) if class_name in small_vehicles else (0, 255, 255) if class_name in large_vehicles else (0, 0, 255)
        cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
        if class_name in small_vehicles:
            small_count_1 += 1
        elif class_name in large_vehicles:
            large_count_1 += 1

    results2 = vehicle_model(frame2)
    for box in results2[0].boxes:
        class_name = results2[0].names[int(box.cls[0].item())]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = (255, 255, 0) if class_name in small_vehicles else (0, 255, 255) if class_name in large_vehicles else (0, 0, 255)
        cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)
        if class_name in small_vehicles:
            small_count_2 += 1
        elif class_name in large_vehicles:
            large_count_2 += 1
    
    if time.time() - time_switched >= green_time_remaining or emergency_count_1 or emergency_count_2:
        if emergency_count_1:
            current_video = 1
            green_time_remaining = predefined_time
        elif emergency_count_2:
            current_video = 2
            green_time_remaining = predefined_time
        else:
            total_vehicles_1 = small_count_1 + large_count_1 + emergency_count_1
            total_vehicles_2 = small_count_2 + large_count_2 + emergency_count_2
            if total_vehicles_1 >= total_vehicles_2:
                current_video = 1
                green_time_remaining = predefined_time if total_vehicles_1 >= threshold else predefined_time / 2
            else:
                current_video = 2
                green_time_remaining = predefined_time if total_vehicles_2 >= threshold else predefined_time / 2
        time_switched = time.time()

    time_left = max(0, green_time_remaining - (time.time() - time_switched))
    frame1 = add_traffic_info(frame1, current_video == 1, time_left)
    frame2 = add_traffic_info(frame2, current_video == 2, time_left)
    cv2.imshow('Video 1', frame1)
    cv2.imshow('Video 2', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
