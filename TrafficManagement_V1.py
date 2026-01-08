# 








import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', etc. for different sizes

# Open video files
input_video_1 = 'videoplayback1.MP4'
input_video_2 = 'videoplayback2.MP4'
cap1 = cv2.VideoCapture(input_video_1)
cap2 = cv2.VideoCapture(input_video_2)

# Initialize traffic light states
traffic_light_1 = 'green'
traffic_light_2 = 'red'

# Function to draw traffic light
def draw_traffic_light(frame, state, position):
    if state == 'green':
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red
    cv2.circle(frame, position, 20, color, -1)  # Draw a filled circle

# Function to draw bounding boxes and labels
def draw_boxes(frame, results):
    vehicle_count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 2:  # Class 2 is for cars in COCO dataset
                vehicle_count += 1
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Get class label
                label = model.names[int(box.cls)]
                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return vehicle_count

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Detect vehicles in frame1 and draw bounding boxes
    results1 = model(frame1)
    vehicle_count1 = draw_boxes(frame1, results1)

    # Detect vehicles in frame2 and draw bounding boxes
    results2 = model(frame2)
    vehicle_count2 = draw_boxes(frame2, results2)

    # Manage traffic lights based on vehicle count
    if vehicle_count1 > vehicle_count2:
        traffic_light_1 = 'green'
        traffic_light_2 = 'red'
    else:
        traffic_light_1 = 'red'
        traffic_light_2 = 'green'

    # Draw traffic lights on frames
    draw_traffic_light(frame1, traffic_light_1, (50, 50))
    draw_traffic_light(frame2, traffic_light_2, (50, 50))

    # Display vehicle counts in the top-right corner
    cv2.putText(frame1, f'Vehicles: {vehicle_count1}', (frame1.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame2, f'Vehicles: {vehicle_count2}', (frame2.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture objects and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()