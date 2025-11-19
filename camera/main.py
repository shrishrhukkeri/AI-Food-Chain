from ultralytics import YOLO
import cv2
model = YOLO("yolov8s.pt")
allowed = ["person", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe","Monkey"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    keep = []
    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        if cls_name in allowed:
            keep.append(box)

    results.boxes = keep
    annotated = results.plot()

    cv2.imshow("Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()