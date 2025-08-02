import cv2
import time
from ultralytics import YOLO

model = YOLO(r"C:\Users\vaibh\OneDrive\Desktop\New folder\Folder Python\Folder ML\opencv_example\pose-detection-keypoints-estimation-yolov8\yolov8n-pose.pt")  

video_path = r"C:\Users\vaibh\OneDrive\Desktop\New folder\Folder Python\Folder ML\opencv_example\Subject_B.MP4"
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not access the video file.")
    exit(1)

start_time = time.time()

while time.time() - start_time < 30:  # Process for 10 seconds
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame)

    for r in results:
        annotated_frame = r.plot()  

    cv2.imshow("Pose Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

video.release()
cv2.destroyAllWindows()
