import cv2
from ultralytics import YOLO

# Load YOLOv8 models
vehicle_model = YOLO("yolov8n.pt")
ambulance_model = YOLO("best.pt")  # Load specialized ambulance detection model

# Define vehicle categories
small_vehicles = {'car', 'motorbike'}
large_vehicles = {'bus', 'truck'}
emergency_vehicles = {'Ambulance', 'fire truck'}

# Load images
image1_path = 'trafficimage.jpg'
image2_path = 'test1.jpg'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

if image1 is None or image2 is None:
    print("Error: Could not load one or both images.")
    exit()

# Resize images to the same height
height = min(image1.shape[0], image2.shape[0])
image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

# Ensure same number of channels
if len(image1.shape) == 2:
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
if len(image2.shape) == 2:
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

# Function to add traffic signal
def add_traffic_info(image, is_green):
    cv2.rectangle(image, (0, 0), (140, 65), (255, 255, 255), cv2.FILLED)
    cv2.circle(image, (26, 30), 16, (0, 0, 255), 2)  # Red light
    cv2.circle(image, (96, 30), 16, (0, 255, 0), 2)  # Green light

    if is_green:
        cv2.circle(image, (96, 30), 16, (0, 255, 0), cv2.FILLED)
    else:
        cv2.circle(image, (26, 30), 16, (0, 0, 255), cv2.FILLED)
    
    return image

# Function to add vehicle count
def add_vehicle_count(image, small_count, large_count, emergency_count):
    h, w, _ = image.shape
    cv2.rectangle(image, (0, h - 50), (180, h), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, f'Small: {small_count}', (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, f'Large: {large_count}', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, f'Emergency: {emergency_count}', (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image

# Process Image 1
small_count_1, large_count_1, emergency_count_1 = 0, 0, 0
ambulance_detected_1 = False
ambulance_results1 = ambulance_model(image1)
for box in ambulance_results1[0].boxes:
    if ambulance_results1[0].names[int(box.cls[0].item())] == 'Ambulance':
        ambulance_detected_1 = True
        break

results1 = vehicle_model(image1)
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
    cv2.rectangle(image1, (x1, y1), (x2, y2), color, 2)

# Process Image 2
small_count_2, large_count_2, emergency_count_2 = 0, 0, 0
ambulance_detected_2 = False
ambulance_results2 = ambulance_model(image2)
for box in ambulance_results2[0].boxes:
    if ambulance_results2[0].names[int(box.cls[0].item())] == 'Ambulance':
        ambulance_detected_2 = True
        break

results2 = vehicle_model(image2)
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
    cv2.rectangle(image2, (x1, y1), (x2, y2), color, 2)

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

# Add traffic info and vehicle count
image1 = add_traffic_info(image1, green_1)
image2 = add_traffic_info(image2, green_2)
image1 = add_vehicle_count(image1, small_count_1, large_count_1, emergency_count_1)
image2 = add_vehicle_count(image2, small_count_2, large_count_2, emergency_count_2)

# Combine images for display
combined_image = cv2.hconcat([image1, image2])
cv2.imshow('Traffic Signal Decision', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Processing complete.")