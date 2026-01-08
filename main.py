import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define vehicle classes
vehicle_classes = {'car', 'truck', 'bus', 'motorbike'}

# Input videos
input_video_1 = 'videoplayback1.MP4'
input_video_2 = 'videoplayback2.MP4'

# Open videos
cap1 = cv2.VideoCapture(input_video_1)
cap2 = cv2.VideoCapture(input_video_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both videos.")
    exit()

# Video properties
fps = int(cap1.get(cv2.CAP_PROP_FPS))
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Traffic light logic
MIN_GREEN_TIME = 5  # Minimum green light duration (seconds)
EXTRA_TIME_PER_VEHICLE = 1  # Extra second per vehicle
current_green = 1  # Start with video 1
last_switch_time = time.time()
paused_frame1, paused_frame2 = None, None

def add_traffic_info(frame, red_light, green_light):
    """Draws traffic signal indicators on the frame."""
    frame_h = frame.shape[0]
    cv2.rectangle(frame, (0, frame_h - 60), (155, frame_h - 15), (255, 255, 255), cv2.FILLED)
    
    cv2.circle(frame, (26, frame_h - 38), 16, (0, 0, 255), 2)  # Red light
    cv2.circle(frame, (126, frame_h - 38), 16, (0, 255, 0), 2)  # Green light

    if red_light:
        cv2.circle(frame, (26, frame_h - 38), 16, (0, 0, 255), cv2.FILLED)
    if green_light:
        cv2.circle(frame, (126, frame_h - 38), 16, (0, 255, 0), cv2.FILLED)

    return frame

while cap1.isOpened() or cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break

    vehicle_count_1, vehicle_count_2 = 0, 0

    if ret1:
        results1 = model(frame1)
        vehicle_count_1 = sum(1 for box in results1[0].boxes if results1[0].names[int(box.cls[0].item())] in vehicle_classes)
        for box in results1[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame1 = add_traffic_info(frame1, current_green != 1, current_green == 1)
    else:
        frame1 = paused_frame1 if paused_frame1 is not None else None
    
    if ret2:
        results2 = model(frame2)
        vehicle_count_2 = sum(1 for box in results2[0].boxes if results2[0].names[int(box.cls[0].item())] in vehicle_classes)
        for box in results2[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame2 = add_traffic_info(frame2, current_green != 2, current_green == 2)
    else:
        frame2 = paused_frame2 if paused_frame2 is not None else None
    
    if frame1 is not None:
        cv2.imshow('Video 1', frame1)
        paused_frame1 = frame1.copy()
    if frame2 is not None:
        cv2.imshow('Video 2', frame2)
        paused_frame2 = frame2.copy()
    
    if time.time() - last_switch_time > MIN_GREEN_TIME:
        current_green = 1 if vehicle_count_1 > vehicle_count_2 else 2
        last_switch_time = time.time()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Processing complete.")
