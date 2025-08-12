"""
rtsp_insightface_attendance.py
Optimized RTSP face recognition prototype using InsightFace (FaceAnalysis).
- Tries to use ONNX GPU provider if available, otherwise CPU.
- Capture thread + Inference thread (process latest frame only).
- Resize + skip frames + simple tracking to reduce load.
- Attendance stored in SQLite; unknown faces saved to 'unknown_faces/'.
"""

import os
import time
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from threading import Thread, Lock
import queue
import argparse

# Try to import insightface and check ONNX providers
import insightface
import onnxruntime as ort

# ---------- CONFIG ----------
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/102?tcp"
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_SAVE_DIR = "unknown_faces"
DB_PATH = "attendance_insightface.db"

# Performance tunables
PROCESS_EVERY_N_FRAMES = 2       # run detection every N captured frames (1 = every frame)
RESIZE_SCALE = 0.5               # scale incoming frame before detection (0.25 - fastest, 1.0 = no resize)
DETECTION_SIZE = (640, 640)      # SCRFD det_size
RECOGNITION_THRESHOLD = 0.50     # cosine / L2 threshold depends on model; 0.5 is typical for ArcFace
TRACKER_TYPE = "CSRT"            # simple per-face tracker between detections
MAX_TRACKER_MISSES = 10          # frames to keep tracker without detection

# --------------------------------------------------

os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

# ---------- Database helpers ----------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp TEXT
)
''')
conn.commit()

db_lock = Lock()

def mark_attendance(name: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_lock:
        # simple duplicate prevention: last record for name within last 60 sec ignored
        c.execute("SELECT timestamp FROM attendance WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        row = c.fetchone()
        if row:
            last_ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_ts).total_seconds() < 60:
                return
        c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, ts))
        conn.commit()
    print(f"[{ts}] Attendance -> {name}")

# ---------- Helper: choose ONNX provider ----------
providers = ort.get_available_providers()
use_cuda = False
if "CUDAExecutionProvider" in providers:
    use_cuda = True
    provider_to_use = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    provider_to_use = ["CPUExecutionProvider"]

print("ONNX Runtime providers:", providers)
print("Using GPU (CUDA)?" , use_cuda)

# ---------- Prepare InsightFace FaceAnalysis ----------
# allowed_modules='detection','recognition' is enough
face_app = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
# ctx_id = 0 for GPU, -1 for CPU
ctx_id = 0 if use_cuda else -1

print("Preparing InsightFace model (this may download models on first run)...")
face_app.prepare(ctx_id=ctx_id, det_size=DETECTION_SIZE)
print("FaceAnalysis ready. Models loaded.")

# ---------- Build known face embeddings from known_faces folder ----------
known_embeddings = []
known_names = []

def build_known_database():
    global known_embeddings, known_names
    known_embeddings = []
    known_names = []
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Please create the folder '{KNOWN_FACES_DIR}' and add authorized person images.")
        return
    for f in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, f)
        if not os.path.isfile(path):
            continue
        name, ext = os.path.splitext(f)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            bgr = cv2.imread(path)
            if bgr is None:
                print("Could not read", path)
                continue
            # InsightFace expects BGR images
            faces = face_app.get(bgr)  # returns list of Face objects with .normed_embedding
            if len(faces) == 0:
                print(f"No face found in {f}, skipping.")
                continue
            # If multiple faces, take the largest (first)
            emb = faces[0].embedding  # numpy array (512-d)
            known_embeddings.append(emb)
            known_names.append(name)
            print(f"Loaded {name} from {f}")
        except Exception as e:
            print("Error loading", f, e)

build_known_database()

# ---------- Video capture thread ----------
class VideoCaptureThread(Thread):
    def __init__(self, src, queue_size=1):
        super().__init__(daemon=True)
        self.src = src
        self.cap = None
        self.stopped = False
        self.frame = None
        self.lock = Lock()
        self.connected = False

    def connect(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.cap = cv2.VideoCapture(self.src)
        # small timeout retry logic if needed
        self.connected = self.cap.isOpened()
        print("VideoCapture opened?:", self.connected)

    def run(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self.connect()
                if not self.connected:
                    print("Unable to open stream, retry in 2s...")
                    time.sleep(2)
                    continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # sometimes the capture returns false transiently
                print("Frame grab failed, reconnecting...")
                self.cap.release()
                self.connected = False
                time.sleep(0.5)
                continue
            with self.lock:
                self.frame = frame
            # tiny sleep to yield
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

# ---------- Simple tracker container ----------
class TrackedFace:
    def __init__(self, bbox, name=None):
        # bbox in (x1,y1,x2,y2) absolute coords on the displayed (original) frame
        self.bbox = bbox
        self.name = name
        self.misses = 0
        self.tracker = cv2.TrackerCSRT_create() if TRACKER_TYPE == "CSRT" else cv2.TrackerKCF_create()
        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1
        # initialize tracker with ROI in format (x,y,w,h)
        self.tracker.init(last_display_frame, (int(x1), int(y1), int(w), int(h)))

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if not ok:
            self.misses += 1
            return False
        x,y,w,h = box
        self.bbox = (int(x), int(y), int(x+w), int(y+h))
        return True

# We'll need a global placeholder for tracker init; updated in main loop
last_display_frame = None

# ---------- Inference & main loop ----------
def l2_norm(a, b):
    # cosine similarity alternative or L2; insightface embeddings are L2-normalized typically.
    # We compute cosine similarity here:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def find_best_match(embedding):
    best_idx = -1
    best_score = -1.0
    for idx, known in enumerate(known_embeddings):
        score = l2_norm(embedding, known)  # cosine in [-1,1], higher = better
        if score > best_score:
            best_score = score
            best_idx = idx
    # convert threshold: for cosine, typical good match > 0.5-0.6 depending on model
    if best_score >= RECOGNITION_THRESHOLD:
        return best_idx, best_score
    return -1, best_score

def save_unknown(face_crop):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(UNKNOWN_SAVE_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(path, face_crop)
    print("Saved unknown face to", path)

def main_loop(rtsp_url):
    global last_display_frame
    cap_thread = VideoCaptureThread(rtsp_url)
    cap_thread.start()
    time.sleep(1.0)  # warm up

    frame_idx = 0
    trackers = []  # list of dict { bbox, name, tracker_obj, misses }
    fps_time = time.time()
    fps_count = 0

    try:
        while True:
            raw = cap_thread.read()
            if raw is None:
                time.sleep(0.01)
                continue

            # optionally convert or resize for display
            display_frame = raw.copy()
            last_display_frame = display_frame  # for tracker initialization
            h0, w0 = display_frame.shape[:2]

            frame_idx += 1
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                print(f"FPS ~ {fps_count}")
                fps_count = 0
                fps_time = time.time()

            # Update trackers first (to have quick bounding boxes)
            new_trackers = []
            for t in trackers:
                ok, box = t['tracker'].update(display_frame)
                if ok:
                    x,y,w,h = box
                    x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
                    t['bbox'] = (x1,y1,x2,y2)
                    t['misses'] = 0
                    new_trackers.append(t)
                else:
                    t['misses'] += 1
                    if t['misses'] < MAX_TRACKER_MISSES:
                        new_trackers.append(t)
                    else:
                        print("Tracker expired for", t.get('name'))
            trackers = new_trackers

            # Only run detection every N frames
            if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
                # Preprocess: resize for detection to speed up
                small = cv2.resize(display_frame, (0,0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
                # FaceAnalysis expects BGR images
                faces = face_app.get(small)  # each face has bbox and embedding
                detected_bboxes = []
                detected_embs = []
                for face in faces:
                    # face.bbox is in small image coordinates [x1, y1, x2, y2]
                    x1,y1,x2,y2 = map(int, face.bbox)
                    # scale back to display coords
                    x1 = int(x1 / RESIZE_SCALE); y1 = int(y1 / RESIZE_SCALE)
                    x2 = int(x2 / RESIZE_SCALE); y2 = int(y2 / RESIZE_SCALE)
                    detected_bboxes.append((x1,y1,x2,y2))
                    detected_embs.append(face.embedding)

                # match detected faces to known embeddings
                for bbox, emb in zip(detected_bboxes, detected_embs):
                    # Check if this detection overlaps an existing tracker -> skip creating a new tracker if so
                    overlap = False
                    for t in trackers:
                        tx1,ty1,tx2,ty2 = t['bbox']
                        # simple IoU / overlap check
                        ix1 = max(bbox[0], tx1); iy1 = max(bbox[1], ty1)
                        ix2 = min(bbox[2], tx2); iy2 = min(bbox[3], ty2)
                        iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                        if iw * ih > 0:
                            overlap = True
                            break
                    if overlap:
                        continue

                    best_idx, score = find_best_match(emb)
                    name = "Unknown"
                    if best_idx >= 0:
                        name = known_names[best_idx]
                        mark_attendance(name)
                    else:
                        # save unknown face crop
                        x1,y1,x2,y2 = bbox
                        # clamp coords
                        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w0-1, x2); y2 = min(h0-1, y2)
                        crop = display_frame[y1:y2, x1:x2]
                        if crop.size != 0:
                            save_unknown(crop)

                    # create a new tracker for this face region
                    tracker_obj = cv2.TrackerCSRT_create() if TRACKER_TYPE == "CSRT" else cv2.TrackerKCF_create()
                    x = bbox[0]; y = bbox[1]; w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
                    try:
                        tracker_obj.init(display_frame, (int(x), int(y), int(w), int(h)))
                        trackers.append({
                            'tracker': tracker_obj,
                            'bbox': bbox,
                            'name': name,
                            'misses': 0
                        })
                    except Exception as e:
                        print("Tracker init failed:", e)

            # draw trackers (labels + boxes)
            for t in trackers:
                x1,y1,x2,y2 = t['bbox']
                label = t.get('name', 'Unknown')
                color = (0,255,0) if label != "Unknown" else (0,0,255)
                cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display_frame, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # show FPS and counts
            cv2.putText(display_frame, f"Trackers:{len(trackers)}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # display
            cv2.imshow("RTSP Face Recognition (InsightFace)", display_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap_thread.stop()
        cv2.destroyAllWindows()
        conn.close()
        print("Shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", type=str, help="RTSP URL", default=RTSP_URL)
    args = parser.parse_args()
    RTSP_URL = args.rtsp
    main_loop(RTSP_URL)
