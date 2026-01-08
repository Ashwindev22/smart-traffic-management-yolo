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
input_video_2 = 'ambulance_traffic.MP4'

# Open videos
cap1 = cv2.VideoCapture(input_video_1)
cap2 = cv2.VideoCapture(input_video_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both videos.")
    exit()

# Function to add traffic signal
def add_traffic_info(frame, is_green):
    cv2.rectangle(frame, (0, 0), (140, 65), (255, 255, 255), cv2.FILLED)
    
    cv2.circle(frame, (26, 30), 16, (0, 0, 255), 2)  # Red light
    cv2.circle(frame, (96, 30), 16, (0, 255, 0), 2)  # Green light

    if is_green:
        cv2.circle(frame, (96, 30), 16, (0, 255, 0), cv2.FILLED)
    else:
        cv2.circle(frame, (26, 30), 16, (0, 0, 255), cv2.FILLED)
    
    return frame

# Function to add vehicle count
def add_vehicle_count(frame, small_count, large_count, emergency_count):
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
    ambulance_detected_1, ambulance_detected_2 = False, False

    # Check for ambulance detection in frame 1
    ambulance_results1 = ambulance_model(frame1)
    for box in ambulance_results1[0].boxes:
        class_name = ambulance_results1[0].names[int(box.cls[0].item())]
        if class_name == 'Ambulance':
            ambulance_detected_1 = True
            emergency_count_1 += 1
            color = (0, 0, 255)
            break
    #print("Detected with best.pt:", [ambulance_results1[int(box.cls[0].item())] for box in ambulance_results1[0].boxes])

    # Process frame 1 for vehicles
    results1 = vehicle_model(frame1)
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
        
    
    # Check for ambulance detection in frame 2
    ambulance_results2 = ambulance_model(frame2)
    for box in ambulance_results2[0].boxes:
        class_name = ambulance_results2[0].names[int(box.cls[0].item())]
        if class_name == 'Ambulance':
            ambulance_detected_2 = True
            emergency_count_2 += 1
            color = (0, 0, 255)
            break

    # Process frame 2 for vehicles
    results2 = vehicle_model(frame2)
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

    
    
   
    # Determine green light priority
    if ambulance_detected_1:
        green_1, green_2 = True, False
    elif ambulance_detected_2:
        green_1, green_2 = False, True
    else:
        total_vehicles_1 = small_count_1 + large_count_1 + emergency_count_1
        total_vehicles_2 = small_count_2 + large_count_2 + emergency_count_2
        green_1 = total_vehicles_1 >= total_vehicles_2
        green_2 = not green_1

    # Add traffic info
    frame1 = add_traffic_info(frame1, green_1)
    frame2 = add_traffic_info(frame2, green_2)
    
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
