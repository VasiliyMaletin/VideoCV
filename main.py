import cv2
from ultralytics import YOLO
import datetime
from helper import create_video_writer


CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

cap = cv2.VideoCapture('videos/cows.mp4')
writer = create_video_writer(cap, 'cows_detect.mp4')
model = YOLO('ai_models/yolov8n.pt')

while True:
    start = datetime.datetime.now()
    success, img = cap.read()

    if not success:
        break

    detections = model(img)[0]
    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(img, (xmin, ymin - 25), (xmin + 40, ymin), GREEN, -1)
        cv2.putText(img, str('cow'), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(img, fps, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    writer.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
