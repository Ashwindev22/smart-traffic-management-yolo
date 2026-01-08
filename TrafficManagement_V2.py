import cv2
from ultralytics import YOLO
import time

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

# Initialize pause states
pause_camera_1 = False
pause_camera_2 = True  # Start with Camera 2 paused

runOnlyOnce = True

vehicle_count1 = 0
vehicle_count2 = 0

# Timer variables
MIN_TRAFFIC_LIGHT_DURATION = 2  # Minimum duration for each traffic light state (in seconds)
last_switch_time = time.time()  # Timestamp of the last traffic light switch

def put_text_with_background(frame, text, position, font_scale, text_color, background_color, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):

    # Get the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Unpack the position (x, y)
    x, y = position

    # Draw the background rectangle
    cv2.rectangle(frame, (x, y - text_height), (x + text_width, y + baseline), background_color, -1)

    # Put the text on the frame
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

    return frame

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
    
    if runOnlyOnce ==  True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        frame1Cpy = frame1.copy()
        frame2Cpy = frame2.copy()
        runOnlyOnce = False

   

    # Process Camera 1 if not paused
    if not pause_camera_1 or traffic_light_1 == 'red':
        if not pause_camera_1:
            ret1, frame1 = cap1.read()
            if not ret1:
                break

        # Detect vehicles in frame1 and draw bounding boxes
        if traffic_light_1 != 'red':
            results1 = model(frame1)   
            vehicle_count1 = draw_boxes(frame1, results1)

        
        # cv2.putText(frame1, f'Vehicles: {vehicle_count1}', (frame1.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        # Display Camera 1 frame
        if traffic_light_1 != 'red':
            frame1Cpy = frame1.copy()
        draw_traffic_light(frame1Cpy, traffic_light_1, (50, 50))
        put_text_with_background(frame1Cpy,f'Vehicles: {vehicle_count1}', (frame1.shape[1] - 200, 30), 1, (0,255,0), (0,0,0), 2 )
        cv2.imshow('Camera 1', frame1Cpy)




    # Process Camera 2 if not paused
    if not pause_camera_2 or traffic_light_2 == 'red':
        if not pause_camera_2:
            ret2, frame2 = cap2.read()
            if not ret2:
                break

        # Detect vehicles in frame2 and draw bounding boxes
        if traffic_light_2 != 'red':
            results2 = model(frame2)
            vehicle_count2 = draw_boxes(frame2, results2)

        
        # cv2.putText(frame2, f'Vehicles: {vehicle_count2}', (frame2.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Display Camera 2 frame
        if traffic_light_2 != 'red':
            frame2Cpy  = frame2.copy()

        draw_traffic_light(frame2Cpy, traffic_light_2, (50, 50))
        put_text_with_background(frame2Cpy,f'Vehicles: {vehicle_count2}', (frame2.shape[1] - 200, 30), 1, (0,255,0), (0,0,0), 2 )
        cv2.imshow('Camera 2', frame2Cpy)





    # Manage traffic lights and pause states
    current_time = time.time()
    if current_time - last_switch_time >= MIN_TRAFFIC_LIGHT_DURATION:
        # Determine which camera has more traffic



        if vehicle_count1 < vehicle_count2:
            # Camera 1 has more traffic, so it gets the green light
            traffic_light_1 = 'green'
            traffic_light_2 = 'red'
            pause_camera_1 = False  # Resume Camera 1
            pause_camera_2 = True   # Pause Camera 2
        else:
            # Camera 2 has more traffic, so it gets the green light
            traffic_light_1 = 'red'
            traffic_light_2 = 'green'
            pause_camera_1 = True   # Pause Camera 1
            pause_camera_2 = False  # Resume Camera 2

        # Update the last switch time
        last_switch_time = current_time

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture objects and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()