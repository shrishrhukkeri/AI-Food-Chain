from ultralytics import YOLO
import cv2
from flask import Flask, Response, render_template_string, jsonify
import threading
import time
import numpy as np
import requests

# -------------------- CONFIG ---------------------
ESPCAM_IP = "172.20.10.2"
ESPCAM_PORT = 81
STREAM_URL = f"http://{ESPCAM_IP}:{ESPCAM_PORT}/stream"
CAPTURE_URL = f"http://{ESPCAM_IP}:{ESPCAM_PORT}/capture"
USE_STREAM = True   # Force MJPEG stream mode
# --------------------------------------------------

print("Loading YOLO model...")
model = YOLO("yolov8x.pt")

allowed = ["person","bird","cat","dog","horse","sheep","cow",
           "elephant","bear","zebra","giraffe"]

# Flask app
app = Flask(__name__)

# Globals
frame_lock = threading.Lock()
current_frame = None
detection_running = True
stream_cap = None

stats = {
    "frames_processed": 0,
    "detections": 0,
    "fps": 0,
    "last_detection": "None"
}

# -------------------- STREAM INIT ---------------------

def init_stream():
    """Open MJPEG stream WITHOUT TCP check."""
    global stream_cap
    print(f"Connecting to MJPEG stream: {STREAM_URL}")

    stream_cap = cv2.VideoCapture(STREAM_URL)

    # Try reading 3 frames to confirm
    for _ in range(3):
        ret, frame = stream_cap.read()
        if ret and frame is not None:
            print("✓ Stream connected.")
            return True
        time.sleep(0.2)

    print("✗ Could not read from MJPEG stream.")
    return False

# -------------------- FALLBACK CAPTURE ---------------

def fetch_frame_from_capture():
    """Fetch single JPEG snapshot."""
    try:
        resp = requests.get(CAPTURE_URL, timeout=3)
        if resp.status_code == 200:
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except:
        pass
    return None

# -------------------- FETCH STREAM ---------------------

def fetch_frame_from_stream():
    global stream_cap
    if stream_cap is None:
        return None

    ret, frame = stream_cap.read()
    if ret and frame is not None:
        return frame
    return None

# -------------------- DETECTION THREAD -----------------

def detect_objects():
    global current_frame, stream_cap, detection_running

    print("Starting detection thread...")

    use_stream_mode = USE_STREAM
    if use_stream_mode:
        if not init_stream():
            print("⚠️ Falling back to /capture.")
            use_stream_mode = False

    frame_times = []

    while detection_running:
        start = time.time()

        # 1️⃣ Try stream
        frame = None
        if use_stream_mode:
            frame = fetch_frame_from_stream()
            if frame is None:
                print("Stream failed → switching to /capture")
                use_stream_mode = False

        # 2️⃣ Try capture
        if not use_stream_mode:
            frame = fetch_frame_from_capture()

        if frame is None:
            time.sleep(0.05)
            continue

        # YOLO inference
        results = model(frame)[0]

        keep = []
        detected_now = []

        for box in results.boxes:
            cls = model.names[int(box.cls)]
            if cls in allowed:
                keep.append(box)
                detected_now.append(cls)

        results.boxes = keep
        annotated = results.plot()

        # Stats
        stats["frames_processed"] += 1
        if detected_now:
            stats["detections"] += len(detected_now)
            stats["last_detection"] = ", ".join(set(detected_now))

        # FPS
        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        stats["fps"] = 1 / (sum(frame_times) / len(frame_times))

        # Overlay
        cv2.putText(annotated,
                    f"FPS: {stats['fps']:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

        if detected_now:
            cv2.putText(annotated,
                        f"Detected: {stats['last_detection']}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

        # Save frame
        with frame_lock:
            current_frame = annotated.copy()

        time.sleep(0.02)

# -------------------- STREAM TO BROWSER -----------------

def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.01)
                continue

            ret, buf = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
            frame = buf.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# -------------------- FLASK ROUTES ---------------------

@app.route("/")
def index():
    return render_template_string("""
        <html>
        <body style="background:#111; color:white; text-align:center;">
            <h1>ESP32-CAM YOLOv8 Detection</h1>
            <img src="{{url_for('video_feed')}}" width="80%">
            <h3>Frames: <span id="f">0</span></h3>
            <h3>Detections: <span id="d">0</span></h3>
            <h3>FPS: <span id="p">0</span></h3>

            <script>
                setInterval(() => {
                    fetch('/stats').then(r => r.json()).then(s => {
                        document.getElementById('f').innerText = s.frames_processed;
                        document.getElementById('d').innerText = s.detections;
                        document.getElementById('p').innerText = s.fps.toFixed(1);
                    });
                }, 1000);
            </script>
        </body>
        </html>
    """)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def get_stats():
    return jsonify(stats)

# -------------------- MAIN ---------------------

if __name__ == "__main__":
    th = threading.Thread(target=detect_objects, daemon=True)
    th.start()

    print("Open: http://localhost:5005")
    app.run(host="0.0.0.0", port=5005, threaded=True)
