import cv2
import time
import threading
from ultralytics import YOLO

camera_index        = 1
frame_width         = 1280
frame_height        = 720
model_path          = 'best-5.onnx'
image_size_input    = 640
confidence          = 0.8

cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

model = YOLO(model_path, task="detect")

frame_lock = threading.Lock()
latest_frame = None
running = True

def capture_frames():
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            latest_frame = frame

capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

frame_times = []

while running:
    start_time = time.time()

    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame

    results = model(frame, conf=confidence, imgsz=image_size_input)
    frame = results[0].plot()

    frame_times.append(time.time() - start_time)
    if len(frame_times) > 30:
        frame_times.pop(0)
    
    fps = len(frame_times) / sum(frame_times)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('KRSRI 2025', frame)

    if cv2.waitKey(1) == ord('q'):
        running = False
        break

capture_thread.join()
cap.release()
cv2.destroyAllWindows()
