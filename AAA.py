"""
face_cctv.py

Low-latency RTSP face recognition + Flask dashboard prototype.

Requirements:
  - ultralytics (YOLOv8)  -> pip install ultralytics
  - insightface           -> pip install insightface
  - onnxruntime-gpu or onnxruntime
  - opencv-python
  - flask
  - numpy
Optionally faiss for fast search.

How it works:
  - CaptureThread reads frames from RTSP into a latest-frame buffer (minimal buffering).
  - WorkerThread runs detection (YOLOv8-face) on resized frames and uses InsightFace
    to compute embeddings, then matches embeddings against known embeddings.
  - Tracker list keeps faces tracked between detections to reduce re-detections.
  - Flask serves a small dashboard to view recent attendance and unknown face crops.
"""

import os
import time
import cv2
import numpy as np
import sqlite3
from threading import Thread, Lock
from datetime import datetime
from queue import Queue, Empty
from flask import Flask, render_template_string, send_from_directory
import logging

# Models
from ultralytics import YOLO          # YOLOv8
import insightface

# ---------------- CONFIG ----------------
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/102"
KNOWN_DIR = "known_faces"
UNKNOWN_DIR = "unknown_faces"
DB_PATH = "attendance.db"

# Performance/tuning
PROCESS_EVERY_N_FRAMES = 2    # detect every 2 frames
RESIZE_FOR_DET = (640, 360)   # detection input size (keep aspect ratio)
DETECTION_CONF = 0.4
IOU_THRESH = 0.45
EMBED_SIM_THRESHOLD = 0.55    # cosine similarity threshold for a match (tune)
MAX_TRACKER_MISSES = 12
TRACKER_TYPE = "CSRT"         # CSRT is robust, KCF is faster but less stable

# Use GPU if available (Ultralytics will use torch.cuda if available)
USE_GPU = True

# ----------------------------------------

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# --- Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# --- Database
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp TEXT
)
""")
conn.commit()
db_lock = Lock()

def mark_attendance(name):
    with db_lock:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # prevent duplicates in short window
        cur.execute("SELECT timestamp FROM attendance WHERE name=? ORDER BY id DESC LIMIT 1", (name,))
        r = cur.fetchone()
        if r:
            last_ts = datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_ts).total_seconds() < 60:  # 60s threshold
                return
        cur.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, ts))
        conn.commit()
    logging.info(f"Attendance marked: {name} @ {ts}")

# --- Load known faces using InsightFace to compute embeddings
logging.info("Preparing InsightFace model...")
face_app = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
# ctx_id = 0 for GPU, -1 for CPU
try:
    # try GPU if available; insightface will pick providers
    face_app.prepare(ctx_id=0, det_size=(640,640))
    logging.info("InsightFace prepared with GPU (ctx_id=0)")
except Exception:
    face_app.prepare(ctx_id=-1, det_size=(640,640))
    logging.info("InsightFace prepared with CPU (ctx_id=-1)")

known_embeddings = []
known_names = []

if not os.path.isdir(KNOWN_DIR):
    os.makedirs(KNOWN_DIR)
    logging.warning(f"Create {KNOWN_DIR}/ and add one image per known person (filename -> label).")

for fname in os.listdir(KNOWN_DIR):
    if not fname.lower().endswith(('.jpg','.jpeg','.png')):
        continue
    path = os.path.join(KNOWN_DIR, fname)
    bgr = cv2.imread(path)
    if bgr is None:
        logging.warning("Could not read known face image: " + path)
        continue
    faces = face_app.get(bgr)  # returns list of Face objects
    if not faces:
        logging.warning("No face found in " + path)
        continue
    emb = faces[0].embedding
    label = os.path.splitext(fname)[0]
    known_embeddings.append(emb)
    known_names.append(label)
    logging.info(f"Loaded {label}")

known_embeddings = np.array(known_embeddings) if known_embeddings else np.zeros((0,512))

# Utility cosine similarity
def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def find_match(emb):
    if known_embeddings.shape[0] == 0:
        return -1, 0.0
    sims = np.dot(known_embeddings, emb / (np.linalg.norm(emb)+1e-8))
    idx = np.argmax(sims)
    score = float(sims[idx])
    if score >= EMBED_SIM_THRESHOLD:
        return idx, score
    return -1, score

# --- YOLOv8 face model
logging.info("Loading YOLOv8-face model (Ultralytics)...")
# This uses ultralytics pretrained models; user can change model to 'yolov8n-face.pt' or a custom model file
# If you have a face-specific YOLO weight, point to it here.
try:
    yolo = YOLO('yolov8n.pt')  # lightweight default; if you have face model use 'yolov8n-face.pt'
except Exception as e:
    logging.info("Default yolov8n model not found locally. ultralytics will attempt download on first run.")
    yolo = YOLO('yolov8n.pt')

# --- Capture thread (keeps only latest frame)
class CaptureThread(Thread):
    def __init__(self, src, name="capture"):
        super().__init__(daemon=True)
        self.src = src
        self.frame = None
        self.lock = Lock()
        self.cap = None
        self.stopped = False

    def run(self):
        # Try different capture backends / options:
        logging.info("Starting capture thread...")
        # Optionally use GStreamer pipeline on Linux for minimal buffering — user can configure here
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        # minimize OpenCV buffer:
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            logging.error("Failed to open stream. Check RTSP URL and network.")
            return
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logging.warning("Frame empty, reconnecting...")
                time.sleep(0.5)
                # try to re-open
                try:
                    self.cap.release()
                except:
                    pass
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            with self.lock:
                self.frame = frame
            # tiny sleep to yield cpu
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            if self.cap:
                self.cap.release()
        except:
            pass

# --- Simple tracker container
def create_tracker():
    if TRACKER_TYPE == "CSRT":
        return cv2.TrackerCSRT_create()
    else:
        return cv2.TrackerKCF_create()

# --- Worker thread: detection + recognition
class WorkerThread(Thread):
    def __init__(self, capture, show_preview=True):
        super().__init__(daemon=True)
        self.capture = capture
        self.show_preview = show_preview
        self.trackers = []  # list of dict { tracker, name, misses }
        self.frame_idx = 0
        self.stopped = False

    def run(self):
        logging.info("Worker started.")
        while not self.stopped:
            frame = self.capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_idx += 1

            # Update trackers first (fast):
            new_trackers = []
            for t in self.trackers:
                ok, box = t['tracker'].update(frame)
                if ok:
                    x,y,w,h = box
                    t['bbox'] = (int(x), int(y), int(x+w), int(y+h))
                    t['misses'] = 0
                    new_trackers.append(t)
                else:
                    t['misses'] += 1
                    if t['misses'] < MAX_TRACKER_MISSES:
                        new_trackers.append(t)
                    else:
                        logging.debug("Tracker expired for " + str(t.get('name')))
            self.trackers = new_trackers

            # Run detection only every N frames to reduce work
            if self.frame_idx % PROCESS_EVERY_N_FRAMES == 0:
                # resize for detection for speed
                small = cv2.resize(frame, RESIZE_FOR_DET)
                # Run YOLO detection (Ultralytics) - returns results
                # We use conf and iou to filter; model returns boxes in xyxy in small frame coords
                results = yolo.predict(small, imgsz=RESIZE_FOR_DET, conf=DETECTION_CONF, iou=IOU_THRESH, verbose=False)
                # results is a list per image (we sent 1 image)
                dets = []
                if len(results) > 0:
                    r = results[0]
                    # r.boxes.xyxyn gives normalized boxes; r.boxes.xyxy gives absolute in small
                    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
                    for box, conf in zip(boxes, confs):
                        x1,y1,x2,y2 = box
                        # map to original frame coords
                        fx = frame.shape[1] / small.shape[1]
                        fy = frame.shape[0] / small.shape[0]
                        bx1 = int(x1 * fx); by1 = int(y1 * fy); bx2 = int(x2 * fx); by2 = int(y2 * fy)
                        dets.append((bx1, by1, bx2, by2, float(conf)))

                # handle detections: compute embedding and match
                for (bx1,by1,bx2,by2,conf) in dets:
                    # check overlap with existing trackers; if overlap, skip new detection
                    skip = False
                    for t in self.trackers:
                        tx1,ty1,tx2,ty2 = t['bbox']
                        # compute intersection
                        ix1 = max(bx1, tx1); iy1 = max(by1, ty1)
                        ix2 = min(bx2, tx2); iy2 = min(by2, ty2)
                        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                        if iw * ih > 0:
                            skip = True
                            break
                    if skip:
                        continue

                    # crop face region (with small margin)
                    h, w = frame.shape[:2]
                    pad = int(0.1 * (bx2 - bx1))
                    x1 = max(0, bx1 - pad); y1 = max(0, by1 - pad)
                    x2 = min(w-1, bx2 + pad); y2 = min(h-1, by2 + pad)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # compute embedding via InsightFace
                    try:
                        faces = face_app.get(face_crop)
                        if not faces:
                            # sometimes get fails; optionally save unknown
                            unknown_path = os.path.join(UNKNOWN_DIR, f"unknown_{int(time.time()*1000)}.jpg")
                            cv2.imwrite(unknown_path, face_crop)
                            logging.info("No face detected by InsightFace — saved crop.")
                            continue
                        emb = faces[0].embedding
                    except Exception as e:
                        logging.exception("InsightFace embedding error: " + str(e))
                        continue

                    idx, score = find_match(emb)
                    name = "Unknown"
                    if idx >= 0:
                        name = known_names[idx]
                        mark_attendance(name)
                    else:
                        # save unknown face for review
                        unknown_path = os.path.join(UNKNOWN_DIR, f"unknown_{int(time.time()*1000)}.jpg")
                        cv2.imwrite(unknown_path, face_crop)
                        logging.info("Unknown saved: " + unknown_path)

                    # create tracker for this face
                    tr = create_tracker()
                    try:
                        tr.init(frame, (x1, y1, x2-x1, y2-y1))
                        self.trackers.append({'tracker': tr, 'bbox': (x1,y1,x2,y2), 'name': name, 'misses': 0})
                    except Exception as e:
                        logging.warning("Tracker init failed: " + str(e))

            # Overlay trackers and show preview
            vis = frame.copy()
            for t in self.trackers:
                x1,y1,x2,y2 = t['bbox']
                label = t['name']
                color = (0,255,0) if label != "Unknown" else (0,0,255)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cv2.putText(vis, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.show_preview:
                cv2.imshow("Face CCTV Preview", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Preview closed by user.")
                    self.stopped = True
                    break

        logging.info("Worker stopped.")

# --- Flask dashboard
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Attendance Dashboard</title>
  <style>
    body{font-family: Arial; margin:20px}
    table{border-collapse: collapse; width:100%}
    th,td{border:1px solid #ccc; padding:8px}
    img{height:100px; margin:6px}
  </style>
</head>
<body>
  <h2>Recent Attendance</h2>
  <table>
    <tr><th>ID</th><th>Name</th><th>Timestamp</th></tr>
    {% for row in rows %}
      <tr><td>{{row[0]}}</td><td>{{row[1]}}</td><td>{{row[2]}}</td></tr>
    {% endfor %}
  </table>

  <h2>Unknown Faces</h2>
  {% for f in unknowns %}
    <img src="/unknown/{{f}}">
  {% endfor %}
</body>
</html>
"""

@app.route("/")
def index():
    cur.execute("SELECT id, name, timestamp FROM attendance ORDER BY id DESC LIMIT 50")
    rows = cur.fetchall()
    unknowns = sorted(os.listdir(UNKNOWN_DIR), reverse=True)[:100]
    return render_template_string(TEMPLATE, rows=rows, unknowns=unknowns)

@app.route("/unknown/<fname>")
def unknown(fname):
    return send_from_directory(UNKNOWN_DIR, fname)

# --- Entrypoint
def main():
    cap = CaptureThread(RTSP_URL)
    cap.start()
    time.sleep(1.0)  # warm up
    worker = WorkerThread(cap, show_preview=True)
    worker.start()

    # Start Flask in another thread
    flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()

    try:
        while True:
            if worker.stopped:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt, stopping...")
    finally:
        worker.stopped = True
        cap.stop()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    main()
