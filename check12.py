import os
import time
import cv2
import numpy as np
import sqlite3
from threading import Thread, Lock
from datetime import datetime
from queue import Queue, Empty
from flask import Flask, render_template_string, send_from_directory, Response
import logging
import base64
from io import BytesIO

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
cur.execute("""
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT,
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
    log_activity(f"Attendance marked: {name}")

def log_activity(event):
    with db_lock:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("INSERT INTO activity_log (event, timestamp) VALUES (?, ?)", (event, ts))
        conn.commit()

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
try:
    yolo = YOLO('yolov8n.pt')  # lightweight default
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
        logging.info("Starting capture thread...")
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
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
        self.latest_frame = None
        self.frame_lock = Lock()
        self.detection_queue = Queue(maxsize=10)  # For sending detections to dashboard
        
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
                        log_activity(f"Tracker expired for {t.get('name', 'Unknown')}")
            self.trackers = new_trackers
            
            # Run detection only every N frames to reduce work
            if self.frame_idx % PROCESS_EVERY_N_FRAMES == 0:
                small = cv2.resize(frame, RESIZE_FOR_DET)
                results = yolo.predict(small, imgsz=RESIZE_FOR_DET, conf=DETECTION_CONF, iou=IOU_THRESH, verbose=False)
                dets = []
                if len(results) > 0:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
                    for box, conf in zip(boxes, confs):
                        x1,y1,x2,y2 = box
                        fx = frame.shape[1] / small.shape[1]
                        fy = frame.shape[0] / small.shape[0]
                        bx1 = int(x1 * fx); by1 = int(y1 * fy); bx2 = int(x2 * fx); by2 = int(y2 * fy)
                        dets.append((bx1, by1, bx2, by2, float(conf)))
                
                # handle detections: compute embedding and match
                for (bx1,by1,bx2,by2,conf) in dets:
                    skip = False
                    for t in self.trackers:
                        tx1,ty1,tx2,ty2 = t['bbox']
                        ix1 = max(bx1, tx1); iy1 = max(by1, ty1)
                        ix2 = min(bx2, tx2); iy2 = min(by2, ty2)
                        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                        if iw * ih > 0:
                            skip = True
                            break
                    if skip:
                        continue
                    
                    h, w = frame.shape[:2]
                    pad = int(0.1 * (bx2 - bx1))
                    x1 = max(0, bx1 - pad); y1 = max(0, by1 - pad)
                    x2 = min(w-1, bx2 + pad); y2 = min(h-1, by2 + pad)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    
                    try:
                        faces = face_app.get(face_crop)
                        if not faces:
                            unknown_path = os.path.join(UNKNOWN_DIR, f"unknown_{int(time.time()*1000)}.jpg")
                            cv2.imwrite(unknown_path, face_crop)
                            log_activity(f"Unknown face detected and saved")
                            # Add to detection queue for dashboard
                            try:
                                self.detection_queue.put_nowait(("unknown", unknown_path, time.time()))
                            except:
                                pass
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
                        log_activity(f"Recognized: {name} (confidence: {score:.2f})")
                        # Add to detection queue for dashboard
                        try:
                            self.detection_queue.put_nowait(("known", name, time.time()))
                        except:
                            pass
                    else:
                        unknown_path = os.path.join(UNKNOWN_DIR, f"unknown_{int(time.time()*1000)}.jpg")
                        cv2.imwrite(unknown_path, face_crop)
                        log_activity(f"Unknown face detected and saved")
                        # Add to detection queue for dashboard
                        try:
                            self.detection_queue.put_nowait(("unknown", unknown_path, time.time()))
                        except:
                            pass
                    
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
            
            # Update latest frame for dashboard
            with self.frame_lock:
                self.latest_frame = vis.copy()
            
            if self.show_preview:
                cv2.imshow("Face CCTV Preview", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Preview closed by user.")
                    self.stopped = True
                    break
        logging.info("Worker stopped.")
    
    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def get_detections(self):
        detections = []
        try:
            while True:
                detections.append(self.detection_queue.get_nowait())
        except Empty:
            pass
        return detections

# --- Flask dashboard
app = Flask(__name__)

# Global reference to worker thread for dashboard
worker_thread = None

TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Face Recognition Dashboard</title>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f5f7fa;
      color: #333;
    }
    .header {
      text-align: center;
      margin-bottom: 30px;
      padding: 20px;
      background: linear-gradient(135deg, #1a2a6c, #2c3e50);
      color: white;
      border-radius: 10px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header h1 {
      margin: 0;
      font-size: 2.5em;
    }
    .header p {
      margin: 10px 0 0;
      font-size: 1.1em;
      opacity: 0.9;
    }
    .dashboard {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-gap: 20px;
    }
    .panel {
      background: white;
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .panel h2 {
      margin-top: 0;
      color: #2c3e50;
      border-bottom: 2px solid #3498db;
      padding-bottom: 10px;
    }
    .video-container {
      position: relative;
      width: 100%;
      padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
      height: 0;
      overflow: hidden;
      border-radius: 8px;
      background: #000;
    }
    .video-container video {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .status-indicator {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-right: 8px;
    }
    .status-live {
      background-color: #2ecc71;
      box-shadow: 0 0 8px #2ecc71;
      animation: pulse 1.5s infinite;
    }
    .status-offline {
      background-color: #e74c3c;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
      100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 15px;
    }
    th, td {
      padding: 12px 15px;
      text-align: left;
      border-bottom: 1px solid #e1e1e1;
    }
    th {
      background-color: #f8f9fa;
      font-weight: 600;
      color: #2c3e50;
    }
    tr:hover {
      background-color: #f8f9fa;
    }
    .face-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      grid-gap: 15px;
      margin-top: 15px;
    }
    .face-item {
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      transition: transform 0.3s ease;
    }
    .face-item:hover {
      transform: scale(1.05);
    }
    .face-item img {
      width: 100%;
      height: 120px;
      object-fit: cover;
      display: block;
    }
    .face-item .label {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0,0,0,0.7);
      color: white;
      padding: 5px;
      font-size: 0.8em;
      text-align: center;
    }
    .detection-feed {
      max-height: 300px;
      overflow-y: auto;
      margin-top: 15px;
    }
    .detection-item {
      display: flex;
      align-items: center;
      padding: 10px;
      border-bottom: 1px solid #eee;
    }
    .detection-item img {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      object-fit: cover;
      margin-right: 15px;
    }
    .detection-info {
      flex-grow: 1;
    }
    .detection-name {
      font-weight: bold;
      margin-bottom: 3px;
    }
    .detection-time {
      font-size: 0.8em;
      color: #777;
    }
    .detection-type {
      padding: 3px 8px;
      border-radius: 12px;
      font-size: 0.7em;
      font-weight: bold;
      text-transform: uppercase;
    }
    .type-known {
      background-color: #d4edda;
      color: #155724;
    }
    .type-unknown {
      background-color: #f8d7da;
      color: #721c24;
    }
    .full-width {
      grid-column: 1 / -1;
    }
    .refresh-btn {
      background-color: #3498db;
      color: white;
      border: none;
      padding: 8px 15px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.9em;
      margin-top: 10px;
      transition: background-color 0.3s;
    }
    .refresh-btn:hover {
      background-color: #2980b9;
    }
  </style>
  <script>
    function updateLiveFeed() {
      const img = document.getElementById('live-feed');
      if (img) {
        img.src = '/video_feed?' + new Date().getTime();
      }
    }
    
    function updateDetections() {
      fetch('/detections')
        .then(response => response.json())
        .then(data => {
          const container = document.getElementById('detection-feed');
          container.innerHTML = '';
          
          if (data.length === 0) {
            container.innerHTML = '<p>No recent detections</p>';
            return;
          }
          
          data.forEach(item => {
            const div = document.createElement('div');
            div.className = 'detection-item';
            
            const img = document.createElement('img');
            if (item.type === 'known') {
              img.src = '/known/' + item.name + '.jpg';
            } else {
              img.src = item.path;
            }
            
            const info = document.createElement('div');
            info.className = 'detection-info';
            
            const name = document.createElement('div');
            name.className = 'detection-name';
            name.textContent = item.type === 'known' ? item.name : 'Unknown Face';
            
            const time = document.createElement('div');
            time.className = 'detection-time';
            time.textContent = new Date(item.timestamp * 1000).toLocaleTimeString();
            
            const type = document.createElement('span');
            type.className = 'detection-type type-' + item.type;
            type.textContent = item.type;
            
            info.appendChild(name);
            info.appendChild(time);
            info.appendChild(type);
            
            div.appendChild(img);
            div.appendChild(info);
            container.appendChild(div);
          });
        })
        .catch(error => console.error('Error updating detections:', error));
    }
    
    // Update live feed every second
    setInterval(updateLiveFeed, 1000);
    
    // Update detections every 5 seconds
    setInterval(updateDetections, 5000);
    
    // Initial load
    document.addEventListener('DOMContentLoaded', function() {
      updateLiveFeed();
      updateDetections();
    });
  </script>
</head>
<body>
  <div class="header">
    <h1>Face Recognition Dashboard</h1>
    <p>Real-time monitoring and attendance tracking system</p>
  </div>
  
  <div class="dashboard">
    <div class="panel">
      <h2>
        <span class="status-indicator status-live"></span>
        Live Video Feed
      </h2>
      <div class="video-container">
        <img id="live-feed" src="/video_feed" alt="Live Feed">
      </div>
    </div>
    
    <div class="panel">
      <h2>Recent Detections</h2>
      <div id="detection-feed" class="detection-feed">
        <p>Loading detections...</p>
      </div>
      <button class="refresh-btn" onclick="updateDetections()">Refresh Detections</button>
    </div>
    
    <div class="panel">
      <h2>Attendance Records</h2>
      <table>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Timestamp</th>
        </tr>
        {% for row in attendance %}
          <tr>
            <td>{{row[0]}}</td>
            <td>{{row[1]}}</td>
            <td>{{row[2]}}</td>
          </tr>
        {% endfor %}
      </table>
    </div>
    
    <div class="panel">
      <h2>Activity Log</h2>
      <table>
        <tr>
          <th>Time</th>
          <th>Event</th>
        </tr>
        {% for row in activity %}
          <tr>
            <td>{{row[1]}}</td>
            <td>{{row[0]}}</td>
          </tr>
        {% endfor %}
      </table>
    </div>
    
    <div class="panel full-width">
      <h2>Known Faces</h2>
      <div class="face-gallery">
        {% for name in known_faces %}
          <div class="face-item">
            <img src="/known/{{name}}.jpg" alt="{{name}}">
            <div class="label">{{name}}</div>
          </div>
        {% endfor %}
      </div>
    </div>
    
    <div class="panel full-width">
      <h2>Unknown Faces</h2>
      <div class="face-gallery">
        {% for f in unknown_faces %}
          <div class="face-item">
            <img src="/unknown/{{f}}" alt="Unknown Face">
            <div class="label">Unknown</div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    # Get attendance records
    cur.execute("SELECT id, name, timestamp FROM attendance ORDER BY id DESC LIMIT 20")
    attendance = cur.fetchall()
    
    # Get activity log
    cur.execute("SELECT event, timestamp FROM activity_log ORDER BY id DESC LIMIT 20")
    activity = cur.fetchall()
    
    # Get known faces
    known_faces = [os.path.splitext(f)[0] for f in os.listdir(KNOWN_DIR) 
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    # Get unknown faces
    unknown_faces = sorted(os.listdir(UNKNOWN_DIR), reverse=True)[:20]
    
    return render_template_string(
        TEMPLATE, 
        attendance=attendance, 
        activity=activity,
        known_faces=known_faces,
        unknown_faces=unknown_faces
    )

@app.route("/video_feed")
def video_feed():
    if worker_thread is None:
        return "Worker not initialized", 404
    
    def generate():
        while True:
            frame = worker_thread.get_latest_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            time.sleep(0.03)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detections")
def get_detections():
    if worker_thread is None:
        return []
    
    detections = worker_thread.get_detections()
    return jsonify(detections)

@app.route("/known/<fname>")
def known(fname):
    return send_from_directory(KNOWN_DIR, fname)

@app.route("/unknown/<fname>")
def unknown(fname):
    return send_from_directory(UNKNOWN_DIR, fname)

# --- Entrypoint
def main():
    global worker_thread
    
    cap = CaptureThread(RTSP_URL)
    cap.start()
    time.sleep(1.0)  # warm up
    
    worker_thread = WorkerThread(cap, show_preview=True)
    worker_thread.start()
    
    # Start Flask in another thread
    flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    
    try:
        while True:
            if worker_thread.stopped:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt, stopping...")
    finally:
        worker_thread.stopped = True
        cap.stop()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    main()
