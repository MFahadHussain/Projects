"""
CCTV Face Recognition System - Single-file MVP
=============================================

Features:
- RTSP ingestion (OpenCV VideoCapture)
- Face detection + embeddings using InsightFace's FaceAnalysis
- Simple FAISS vector index for nearest-neighbor search
- SQLite metadata for enrollments and logs
- Flask web UI: live MJPEG stream + simple API to enroll faces
- Simple tracklet-based de-duplication (per-camera recent seen cache)

Notes / Requirements:
- Recommended Python 3.9+
- Install dependencies (see requirements.txt below)

requirements.txt (example)
--------------------------
flask
opencv-python-headless
insightface
faiss-cpu
numpy
pillow
sqlalchemy
gunicorn

Install with:
    pip install -r requirements.txt

Model downloads:
InsightFace will automatically fetch models on first run (FaceAnalysis).

Usage:
    python cctv_face_recognition_app.py

Then open http://localhost:5000/ to view the dashboard.

This is a minimal, pragmatic MVP designed to be a solid starting point.
It's not hardened for production. See comments in-code for places to improve
(e.g., GPU acceleration, Triton/TensorRT, DeepSORT, Milvus, TLS, RBAC).

"""

import os
import io
import time
import threading
import sqlite3
from datetime import datetime
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
from flask import Flask, Response, request, jsonify, send_file, render_template_string

# Optional: faiss-cpu or faiss-gpu
try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss is required. install faiss-cpu via pip before running. Error: " + str(e))

# InsightFace for detection + embedding
try:
    from insightface.app import FaceAnalysis
except Exception as e:
    raise RuntimeError("insightface is required. pip install insightface. Error: " + str(e))

# ========== Configuration ==========
RTSP_STREAMS = [
    # Add your RTSP URLs here
    # Example: "rtsp://username:password@192.168.1.100:554/Streaming/Channels/101"
]

CAM_PROCESS_FPS = 2  # process this many frames per second per camera (reduce load)
EMBEDDING_DIM = 512  # insightface default (ArcFace family)
MATCH_THRESHOLD = 0.45  # cosine distance threshold (tuned per deployment)
SAVE_CROPS_DIR = "face_crops"
DB_PATH = "faces.db"

os.makedirs(SAVE_CROPS_DIR, exist_ok=True)

# ========== Database helper (SQLite) ==========
class DB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()
        self.lock = threading.Lock()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            metadata TEXT,
            embedding BLOB,
            created_at TEXT
        )
        ''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT,
            name TEXT,
            distance REAL,
            thumbnail_path TEXT,
            ts TEXT
        )
        ''')
        self.conn.commit()

    def add_enrollment(self, name: str, embedding: np.ndarray, metadata: str = ""):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("INSERT INTO enrollments (name, metadata, embedding, created_at) VALUES (?, ?, ?, ?)",
                        (name, metadata, embedding.tobytes(), datetime.utcnow().isoformat()))
            self.conn.commit()
            return cur.lastrowid

    def get_all_enrollments(self) -> List[Tuple[int, str, np.ndarray]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, embedding FROM enrollments")
        rows = cur.fetchall()
        out = []
        for r in rows:
            emb = np.frombuffer(r[2], dtype=np.float32)
            out.append((r[0], r[1], emb))
        return out

    def add_log(self, camera: str, name: str, distance: float, thumbnail_path: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("INSERT INTO logs (camera, name, distance, thumbnail_path, ts) VALUES (?, ?, ?, ?, ?)",
                        (camera, name, distance, thumbnail_path, datetime.utcnow().isoformat()))
            self.conn.commit()

db = DB()

# ========== FaceEngine (InsightFace wrapper) ==========
class FaceEngine:
    def __init__(self):
        print("Initializing InsightFace FaceAnalysis (this may download models)...")
        self.app = FaceAnalysis(name="antelope", providers=['CPUExecutionProvider'])
        # name "antelope" is a compact model selection; adjust for accuracy/speed
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print("FaceAnalysis ready")

    def detect_and_embed(self, bgr_img: np.ndarray) -> List[Dict[str, Any]]:
        # InsightFace expects BGR images (OpenCV style)
        faces = self.app.get(bgr_img)
        # Each face has bbox, kps, normed_embedding
        results = []
        for f in faces:
            res = {
                'bbox': f.bbox.astype(int).tolist(),  # [x1,y1,x2,y2]
                'kps': f.kps.tolist(),
                'embedding': f.normed_embedding.astype(np.float32) if hasattr(f, 'normed_embedding') else f.embedding.astype(np.float32),
                'det_score': float(f.det_score)
            }
            results.append(res)
        return results

face_engine = FaceEngine()

# ========== FAISS index management ==========
class FaissIndex:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # use inner product on normalized embeddings
        # store mapping id -> (db_id, name)
        self.metadata = []
        self.lock = threading.Lock()

    def rebuild_from_db(self):
        enrolls = db.get_all_enrollments()
        vecs = []
        self.metadata = []
        for (dbid, name, emb) in enrolls:
            # ensure L2-normalized embeddings for cosine similarity via IP
            v = emb.astype(np.float32)
            if np.linalg.norm(v) == 0:
                continue
            v = v / np.linalg.norm(v)
            vecs.append(v)
            self.metadata.append((dbid, name))
        if vecs:
            mat = np.vstack(vecs)
            with self.lock:
                self.index.reset()
                self.index.add(mat)
        else:
            with self.lock:
                self.index.reset()

    def add(self, emb: np.ndarray, dbid: int, name: str):
        v = emb.astype(np.float32)
        if np.linalg.norm(v) == 0:
            return
        v = v / np.linalg.norm(v)
        with self.lock:
            self.index.add(v.reshape(1, -1))
            self.metadata.append((dbid, name))

    def search(self, emb: np.ndarray, topk: int = 5) -> List[Tuple[int, float, str]]:
        if np.linalg.norm(emb) == 0:
            return []
        v = emb.astype(np.float32)
        v = v / np.linalg.norm(v)
        with self.lock:
            if self.index.ntotal == 0:
                return []
            D, I = self.index.search(v.reshape(1, -1), topk)
            # D is inner product (cosine) score when normalized
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                dbid, name = self.metadata[idx]
                # Convert similarity to distance-ish metric: distance = 1 - sim
                distance = float(1.0 - score)
                results.append((dbid, distance, name))
            return results

faiss_index = FaissIndex(dim=EMBEDDING_DIM)
faiss_index.rebuild_from_db()

# ========== Simple in-memory per-camera seen history to avoid spamming logs ==========
class SeenCache:
    def __init__(self):
        self.cache = {}  # camera -> dict(track_id -> last_ts)
        self.ttl = 10.0  # seconds to consider seen
        self.lock = threading.Lock()

    def is_recent(self, camera: str, identity_key: str) -> bool:
        now = time.time()
        with self.lock:
            cammap = self.cache.setdefault(camera, {})
            last = cammap.get(identity_key, 0)
            if now - last < self.ttl:
                return True
            cammap[identity_key] = now
            # prune old
            for k, v in list(cammap.items()):
                if now - v > (self.ttl * 4):
                    del cammap[k]
            return False

seen_cache = SeenCache()

# ========== Camera worker ==========
class CameraWorker(threading.Thread):
    def __init__(self, cam_name: str, rtsp_url: str):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.rtsp_url = rtsp_url
        self.vcap = None
        self.last_frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        print(f"Starting camera worker for {self.cam_name}")
        # OpenCV VideoCapture with RTSP
        self.vcap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.vcap.isOpened():
            print(f"Failed to open stream for {self.cam_name}: {self.rtsp_url}")
            return
        fps_wait = max(1.0 / CAM_PROCESS_FPS, 0.001)
        while self.running:
            ret, frame = self.vcap.read()
            if not ret or frame is None:
                # Try reconnect
                print(f"{self.cam_name}: frame read failed, reconnecting...")
                time.sleep(2.0)
                try:
                    self.vcap.release()
                except:
                    pass
                self.vcap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                time.sleep(1.0)
                continue
            # store last frame for streaming
            with self.lock:
                self.last_frame = frame.copy()
            # process at configured fps
            self.process_frame(frame)
            time.sleep(fps_wait)

    def get_frame(self):
        with self.lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def stop(self):
        self.running = False
        try:
            self.vcap.release()
        except:
            pass

    def process_frame(self, frame: np.ndarray):
        # Detection + embedding
        faces = face_engine.detect_and_embed(frame)
        for i, f in enumerate(faces):
            x1, y1, x2, y2 = f['bbox']
            emb = f['embedding']
            # Search FAISS
            results = faiss_index.search(emb, topk=3)
            name = "Unknown"
            distance = 1.0
            if results:
                best_dbid, best_dist, best_name = results[0]
                distance = best_dist
                if distance < MATCH_THRESHOLD:
                    name = best_name
            # Deduplicate by seen cache
            identity_key = f"{name}:{int(x1)}:{int(y1)}:{int(x2)}:{int(y2)}"
            if not seen_cache.is_recent(self.cam_name, identity_key):
                # Save thumbnail
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                filename = f"{self.cam_name}_{ts}_{i}.jpg"
                path = os.path.join(SAVE_CROPS_DIR, filename)
                try:
                    cv2.imwrite(path, crop)
                except Exception as e:
                    print("Failed to save crop:", e)
                    path = ""
                db.add_log(self.cam_name, name, distance, path)
            # Optionally: place overlays on the last_frame for streaming
            with self.lock:
                draw = self.last_frame
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw, f"{name} ({distance:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)

# ========== Start camera workers for configured streams ==========
camera_workers: Dict[str, CameraWorker] = {}
for idx, url in enumerate(RTSP_STREAMS):
    cam_name = f"cam{idx}"
    w = CameraWorker(cam_name, url)
    camera_workers[cam_name] = w
    w.start()

# ========== Flask app (simple UI + API) ==========
app = Flask(__name__)

INDEX_HTML = '''
<!doctype html>
<title>Face Recognition Dashboard - MVP</title>
<h1>Face Recognition Dashboard</h1>
<div>
  <h2>Live Streams</h2>
  {% for cam in cams %}
    <div style="display:inline-block;margin:10px;">
      <h3>{{cam}}</h3>
      <img src="/video_feed/{{cam}}" width="640" />
    </div>
  {% endfor %}
</div>
<div>
  <h2>Enroll a face</h2>
  <form action="/enroll" method="post" enctype="multipart/form-data">
    Name: <input type="text" name="name" required />
    Image: <input type="file" name="image" accept="image/*" required />
    <input type="submit" value="Enroll" />
  </form>
</div>
<div>
  <h2>Recent logs</h2>
  <iframe src="/logs" width="100%" height="300"></iframe>
</div>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, cams=list(camera_workers.keys()))

def gen_mjpeg(camera_worker: CameraWorker):
    while True:
        frame = camera_worker.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        # encode as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/video_feed/<cam>')
def video_feed(cam):
    if cam not in camera_workers:
        return "Not found", 404
    return Response(gen_mjpeg(camera_workers[cam]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/enroll', methods=['POST'])
def enroll():
    name = request.form.get('name')
    img = request.files.get('image')
    if not name or not img:
        return "Missing", 400
    img_bytes = img.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    faces = face_engine.detect_and_embed(bgr)
    if not faces:
        return "No face found", 400
    # pick largest face
    faces_sorted = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
    emb = faces_sorted[0]['embedding']
    dbid = db.add_enrollment(name, emb)
    faiss_index.add(emb, dbid, name)
    return jsonify({'status': 'ok', 'id': dbid, 'name': name})

@app.route('/logs')
def logs():
    cur = db.conn.cursor()
    cur.execute('SELECT camera, name, distance, thumbnail_path, ts FROM logs ORDER BY id DESC LIMIT 200')
    rows = cur.fetchall()
    html = '<html><body><table border=1><tr><th>ts</th><th>camera</th><th>name</th><th>distance</th><th>thumb</th></tr>'
    for r in rows:
        cam, name, dist, thumb, ts = r
        thumb_html = f'<a href="/thumb/{os.path.basename(thumb)}">{os.path.basename(thumb)}</a>' if thumb else ''
        html += f'<tr><td>{ts}</td><td>{cam}</td><td>{name}</td><td>{dist:.3f}</td><td>{thumb_html}</td></tr>'
    html += '</table></body></html>'
    return html

@app.route('/thumb/<filename>')
def thumb(filename):
    path = os.path.join(SAVE_CROPS_DIR, filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/rebuild_index')
def rebuild_index():
    faiss_index.rebuild_from_db()
    return jsonify({'status': 'ok'})

# graceful shutdown
import atexit

def shutdown():
    for w in camera_workers.values():
        w.stop()
    print("Shutdown complete")

atexit.register(shutdown)

if __name__ == '__main__':
    # Start Flask
    print("Starting Flask app on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
