import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure you have the model file downloaded

# Define vehicle classes (COCO dataset classes)
vehicle_classes = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck
emergency_classes = [13]  # 13: emergency vehicle (assuming ambulance/fire truck is labeled as such)

# Video capture
video1 = cv2.VideoCapture("videoplayback1.mp4")  # Replace with your video path or camera stream
video2 = cv2.VideoCapture("videoplayback2.mp4")  # Replace with your video path or camera stream

# Fixed time frame for each channel
base_green_time = 10  # Base green time in seconds
max_green_time = 30  # Maximum green time in seconds
emergency_green_time = 15  # Green time for emergency vehicles

# Function to count vehicles and prioritize emergency vehicles
def count_vehicles(results):
    vehicle_count = 0
    emergency_vehicle_detected = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in vehicle_classes:
                vehicle_count += 1
            if class_id in emergency_classes:
                emergency_vehicle_detected = True

    return vehicle_count, emergency_vehicle_detected

# Function to display the video with bounding boxes and counts
def display_video(frame, vehicle_count, channel):
    cv2.putText(frame, f"Channel {channel} Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(f"Channel {channel}", frame)

# Main loop
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1 or not ret2:
        print("Video ended or failed to read frame.")
        break

    # Perform inference on both videos
    results1 = model(frame1, stream=True)  # Use stream=True for real-time processing
    results2 = model(frame2, stream=True)

    # Count vehicles and check for emergency vehicles
    vehicle_count1, emergency1 = count_vehicles(results1)
    vehicle_count2, emergency2 = count_vehicles(results2)

    # Display videos with bounding boxes and counts
    display_video(frame1, vehicle_count1, 1)
    display_video(frame2, vehicle_count2, 2)

    # Determine green signal time based on vehicle count and emergency vehicles
    if emergency1:
        green_time1 = emergency_green_time
        green_time2 = 0
    elif emergency2:
        green_time1 = 0
        green_time2 = emergency_green_time
    else:
        green_time1 = min(base_green_time + vehicle_count1, max_green_time)
        green_time2 = min(base_green_time + vehicle_count2, max_green_time)

    # Print green signal allocation
    if green_time1 > 0:
        print(f"Green signal for Channel 1 for {green_time1} seconds")
        time.sleep(green_time1)  # Simulate green signal duration
    elif green_time2 > 0:
        print(f"Green signal for Channel 2 for {green_time2} seconds")
        time.sleep(green_time2)  # Simulate green signal duration

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video1.release()
video2.release()
cv2.destroyAllWindows()