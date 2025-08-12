#pip install ultralytics insightface flask-socketio eventlet opencv-python-headless


import cv2
import numpy as np
import os
import base64
from datetime import datetime
import sqlite3
from ultralytics import YOLO
import insightface
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

# ----------------------------
# CONFIG
# ----------------------------
VIDEO_SOURCE = "rtsp://username:password@camera-ip:554/stream"
DETECTION_CONF = 0.35
IOU_THRESH = 0.45
RESIZE_FOR_DET = (960, 540)
EMBED_SIM_THRESHOLD = 0.6  # higher for fewer false positives
KNOWN_FACES_DIR = "known_faces"
SNAPSHOT_DIR = "snapshots"

# ----------------------------
# INIT FLASK + SOCKETIO
# ----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------------------
# INIT YOLOv8m-face
# ----------------------------
model = YOLO("yolov8m-face.pt")  # better recall than nano

# ----------------------------
# INIT InsightFace buffalo_l
# ----------------------------
face_analysis = insightface.app.FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_analysis.prepare(ctx_id=0, det_size=(640, 640), model='buffalo_l')

# ----------------------------
# LOAD KNOWN FACE EMBEDDINGS
# ----------------------------
known_embeddings = {}  # {person_name: [embedding1, embedding2, ...]}

def load_known_faces():
    global known_embeddings
    known_embeddings.clear()
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for file_name in os.listdir(person_folder):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_folder, file_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = face_analysis.get(img)
                if faces:
                    # take the biggest face
                    face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
                    embeddings.append(face.normed_embedding)
        if embeddings:
            known_embeddings[person_name] = embeddings
    print(f"[INFO] Loaded embeddings for {len(known_embeddings)} people.")

load_known_faces()

# ----------------------------
# INIT DATABASE
# ----------------------------
conn = sqlite3.connect("attendance.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                timestamp TEXT
            )''')
c.execute('''CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                confidence REAL,
                snapshot_path TEXT,
                timestamp TEXT
            )''')
conn.commit()

# ----------------------------
# HELPER: MATCH FACE
# ----------------------------
def match_face(embedding):
    best_match = None
    best_score = 0
    for name, embed_list in known_embeddings.items():
        for stored_emb in embed_list:
            sim = np.dot(embedding, stored_emb)
            if sim > best_score:
                best_score = sim
                best_match = name
    if best_score >= EMBED_SIM_THRESHOLD:
        return best_match, best_score
    return None, best_score

# ----------------------------
# VIDEO CAPTURE + DETECTION LOOP
# ----------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)

def process_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for detection
        det_frame = cv2.resize(frame, RESIZE_FOR_DET)

        # Detect faces
        results = model(det_frame, conf=DETECTION_CONF, iou=IOU_THRESH)
        annotated_frame = det_frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                face_crop = det_frame[y1:y2, x1:x2]
                faces = face_analysis.get(face_crop)
                if not faces:
                    continue

                face = faces[0]  # best face in crop
                embedding = face.normed_embedding

                name, score = match_face(embedding)
                display_name = name if name else "Unknown"

                # Draw on frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{display_name} {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                snapshot_path = os.path.join(SNAPSHOT_DIR, f"{display_name}_{timestamp}.jpg")
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                cv2.imwrite(snapshot_path, face_crop)

                # Log to DB
                c.execute(
                    "INSERT INTO logs (name, confidence, snapshot_path, timestamp) VALUES (?, ?, ?, ?)",
                    (display_name, score, snapshot_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()

                if name:
                    # Mark attendance only once per session
                    c.execute("SELECT COUNT(*) FROM attendance WHERE name=? AND timestamp LIKE ?",
                              (name, datetime.now().strftime("%Y-%m-%d") + "%"))
                    if c.fetchone()[0] == 0:
                        c.execute(
                            "INSERT INTO attendance (name, timestamp) VALUES (?, ?)",
                            (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        )
                        conn.commit()
                        socketio.emit("attendance", {"name": name})

        # Emit frame to dashboard
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit("frame", {"image": frame_b64})

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/files/<path:filename>")
def files(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    import threading
    t = threading.Thread(target=process_video)
    t.daemon = True
    t.start()
    socketio.run(app, host="0.0.0.0", port=5000)
