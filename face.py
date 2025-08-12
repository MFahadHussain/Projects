# face_cctv_full.py
"""
Low-latency CCTV Face Recognition System
- Threaded RTSP capture (latest frame only)
- YOLOv8 detection (Ultralytics) + InsightFace recognition
- Simple tracker (CSRT) for between-detection frames
- Flask dashboard showing recent attendance and unknown faces
- SQLite DB for attendance
"""

import os
import time
import cv2
import numpy as np
import sqlite3
import logging
from threading import Thread, Lock
from datetime import datetime
from flask import Flask, render_template_string, send_from_directory
from ultralytics import YOLO
import insightface

# ---------- CONFIG ----------
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/102?tcp"
KNOWN_DIR = "known_faces"
UNKNOWN_DIR = "unknown_faces"
DB_PATH = "attendance.db"

# Performance tunables
PROCESS_EVERY_N_FRAMES = 2       # detect every N frames
RESIZE_FOR_DET = (640, 360)      # detection input size (width, height)
DETECTION_CONF = 0.35
IOU_THRESH = 0.45
EMBED_SIM_THRESHOLD = 0.55       # cosine threshold for match (tune 0.5-0.65)
MAX_TRACKER_MISSES = 12
TRACKER_TYPE = "CSRT"
SHOW_PREVIEW = True              # set False to disable preview window

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# Ensure folders exist
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ---------- DB setup ----------
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

def mark_attendance(name: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_lock:
        cur.execute("SELECT timestamp FROM attendance WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        row = cur.fetchone()
        if row:
            last_ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_ts).total_seconds() < 60:
                return
        cur.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, ts))
        conn.commit()
    logging.info(f"Attendance: {name} @ {ts}")

# ---------- InsightFace (recognition) ----------
logging.info("Preparing InsightFace (recognition)...")
face_app = insightface.app.FaceAnalysis(allowed_modules=['detection','recognition'])
# try GPU ctx_id=0, else CPU -1
try:
    face_app.prepare(ctx_id=0, det_size=(640,640))
    logging.info("InsightFace ready on GPU (ctx_id=0)")
except Exception:
    face_app.prepare(ctx_id=-1, det_size=(640,640))
    logging.info("InsightFace ready on CPU (ctx_id=-1)")

# Load known face embeddings
known_embeddings = []
known_names = []
def build_known_db():
    global known_embeddings, known_names
    known_embeddings = []
    known_names = []
    for f in os.listdir(KNOWN_DIR):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(KNOWN_DIR, f)
        img = cv2.imread(path)
        if img is None:
            logging.warning("Cannot read known face image: " + path)
            continue
        faces = face_app.get(img)
        if not faces:
            logging.warning("No face found in known image: " + f)
            continue
        emb = faces[0].embedding
        name = os.path.splitext(f)[0]
        known_embeddings.append(emb)
        known_names.append(name)
        logging.info(f"Loaded known face: {name}")
    if len(known_embeddings) > 0:
        known_embeddings = np.vstack(known_embeddings)
    else:
        known_embeddings = np.zeros((0,512))
build_known_db()

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def find_match(emb):
    if known_embeddings.shape[0] == 0:
        return -1, 0.0
    sims = np.dot(known_embeddings, emb / (np.linalg.norm(emb)+1e-8))
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    if score >= EMBED_SIM_THRESHOLD:
        return idx, score
    return -1, score

# ---------- YOLOv8 detection (Ultralytics) ----------
logging.info("Loading YOLOv8 model (yolov8n by default). This may download weights...")
yolo = YOLO('yolov8n.pt')  # replace with a face-specific weight if available

# ---------- Capture thread ----------
class CaptureThread(Thread):
    def __init__(self, src):
        super().__init__(daemon=True)
        self.src = src
        self.lock = Lock()
        self.frame = None
        self.stopped = False
        self.cap = None

    def run(self):
        logging.info("Capture thread starting...")
        # Try with FFMPEG backend
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            logging.error("Could not open RTSP stream.")
            return
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logging.warning("Empty frame, attempting reconnect...")
                try:
                    self.cap.release()
                except:
                    pass
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            with self.lock:
                self.frame = frame
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

# ---------- Tracker factory ----------
def create_tracker():
    if TRACKER_TYPE == "CSRT":
        return cv2.TrackerCSRT_create()
    else:
        return cv2.TrackerKCF_create()

# ---------- Worker: detection + recognition + tracking ----------
class Worker(Thread):
    def __init__(self, capture: CaptureThread):
        super().__init__(daemon=True)
        self.capture = capture
        self.trackers = []  # list of dicts {'tracker', 'bbox', 'name', 'misses'}
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

            # Update trackers
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

            # Run detection every N frames
            if self.frame_idx % PROCESS_EVERY_N_FRAMES == 0:
                # Resize for detection
                small = cv2.resize(frame, RESIZE_FOR_DET)
                results = yolo.predict(small, imgsz=RESIZE_FOR_DET, conf=DETECTION_CONF, iou=IOU_THRESH, verbose=False)
                dets = []
                if len(results) > 0:
                    r = results[0]
                    if hasattr(r, 'boxes') and len(r.boxes) > 0:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        # Map boxes to original frame size
                        fx = frame.shape[1] / small.shape[1]
                        fy = frame.shape[0] / small.shape[0]
                        for (x1,y1,x2,y2), conf in zip(boxes, confs):
                            bx1 = int(x1 * fx); by1 = int(y1 * fy); bx2 = int(x2 * fx); by2 = int(y2 * fy)
                            dets.append((bx1,by1,bx2,by2,float(conf)))

                # Process detections
                for bx1,by1,bx2,by2,conf in dets:
                    # skip if overlaps existing tracker
                    overlap = False
                    for t in self.trackers:
                        tx1,ty1,tx2,ty2 = t['bbox']
                        ix1 = max(bx1, tx1); iy1 = max(by1, ty1)
                        ix2 = min(bx2, tx2); iy2 = min(by2, ty2)
                        iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                        if iw * ih > 0:
                            overlap = True
                            break
                    if overlap:
                        continue

                    # Crop face with padding
                    h, w = frame.shape[:2]
                    pad = int(0.05*(bx2-bx1))
                    x1 = max(0, bx1-pad); y1 = max(0, by1-pad)
                    x2 = min(w-1, bx2+pad); y2 = min(h-1, by2+pad)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # Get embedding via InsightFace
                    try:
                        faces = face_app.get(face_crop)
                        if not faces:
                            # Save unknown crop
                            path = os.path.join(UNKNOWN_DIR, f"unknown_{int(time.time()*1000)}.jpg")
                            cv2.imwrite(path, face_crop)
                            logging.info("InsightFace found no face; saved crop.")
                            continue
                        emb = faces[0].embedding
                    except Exception as e:
                        logging.exception("InsightFace error: " + str(e))
                        continue

                    idx, score = find_match(emb)
                    name = "Unknown"
                    if idx >= 0:
                        name = known_names[idx]
                        mark_attendance(name)
                    else:
                        # Save unknown
                        p = os.path.join(UNKNOWN_DIR, f"unknown_{int(time.time()*1000)}.jpg")
                        cv2.imwrite(p, face_crop)
                        logging.info("Unknown saved: " + p)

                    # create tracker
                    tr = create_tracker()
                    try:
                        tr.init(frame, (x1, y1, x2-x1, y2-y1))
                        self.trackers.append({'tracker': tr, 'bbox': (x1,y1,x2,y2), 'name': name, 'misses': 0})
                    except Exception as e:
                        logging.warning("Tracker init failed: " + str(e))

            # Visualize
            vis = frame.copy()
            for t in self.trackers:
                x1,y1,x2,y2 = t['bbox']
                label = t['name']
                color = (0,255,0) if label != "Unknown" else (0,0,255)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cv2.putText(vis, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if SHOW_PREVIEW:
                cv2.imshow("Face CCTV Preview", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Preview closed by user.")
                    self.stopped = True
                    break

        logging.info("Worker stopped.")

# ---------- Flask dashboard ----------
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
    cur.execute("SELECT id, name, timestamp FROM attendance ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()
    unknowns = sorted(os.listdir(UNKNOWN_DIR), reverse=True)[:200]
    return render_template_string(TEMPLATE, rows=rows, unknowns=unknowns)

@app.route("/unknown/<fname>")
def unknown_file(fname):
    return send_from_directory(UNKNOWN_DIR, fname)

# ---------- Main ----------
def main():
    # build known DB (embeddings + names)
    build_known_db()
    # start capture and worker threads
    cap = CaptureThread(RTSP_URL)
    cap.start()
    time.sleep(1.0)  # warm up
    worker = Worker(cap)
    worker.start()

    # start flask
    flask_thread = Thread(target=lambda: app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()

    try:
        while True:
            if worker.stopped:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt, shutting down...")
    finally:
        worker.stopped = True
        cap.stop()
        conn.close()
        cv2.destroyAllWindows()
        logging.info("Exited cleanly.")

if __name__ == "__main__":
    main()
