# app_combined.py
import os
import threading
import time
import csv
from datetime import datetime
import numpy as np
import cv2
import face_recognition
import imageio_ffmpeg
from playsound import playsound
from flask import Flask, Response, jsonify, request, render_template_string

# ========== CONFIG ==========
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.125:554/Streaming/channels/101"
KNOWN_FACES_DIR = "images"
UNKNOWN_DIR = "unknowns"
ZOOM_FACTOR = 1.4
ALERT_SOUND = "alert.wav"        # optional (place in project root)
ATTENDANCE_FILE = "attendance.csv"
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FRAME_PROCESS_EVERY = 5         # process every N-th frame for performance
FACE_DISTANCE_THRESHOLD = 0.55
ALERT_COOLDOWN = 5.0            # seconds between sounds

# ========== APP GLOBALS ==========
app = Flask(__name__)
output_frame = None             # latest jpeg bytes for web streaming
frame_lock = threading.Lock()
camera_thread = None
camera_running = False
camera_thread_lock = threading.Lock()

# Face data
known_encodings = []
known_names = []

# Attendance & logs (in-memory)
attendance_today = set()
detection_logs = []  # list of dicts: {"name":..., "time":..., "snapshot": optional}

# Alert throttling
_last_alert_time = 0.0

# Ensure folders & files exist
os.makedirs(UNKNOWN_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time"])

# ========== UTILITIES ==========
def load_known_faces():
    encs = []
    names = []
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"[WARN] Known faces dir '{KNOWN_FACES_DIR}' not found.")
        return encs, names
    for fn in os.listdir(KNOWN_FACES_DIR):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, fn)
            try:
                img = face_recognition.load_image_file(path)
                e = face_recognition.face_encodings(img)
                if e:
                    encs.append(e[0])
                    names.append(os.path.splitext(fn)[0])
                else:
                    print(f"[WARN] No face found in: {fn}")
            except Exception as ex:
                print(f"[WARN] Failed to load {fn}: {ex}")
    return encs, names

def mark_attendance(name):
    global attendance_today
    if name in attendance_today:
        return False
    now = datetime.now()
    ds = now.strftime("%Y-%m-%d")
    ts = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, ds, ts])
    attendance_today.add(name)
    entry = {"name": name, "date": ds, "time": ts}
    detection_logs.insert(0, {"name": name, "time": f"{ds} {ts}"})
    print(f"[ATTEND] {name} @ {ts}")
    return True

def play_alert_nonblocking():
    global _last_alert_time
    try:
        now = time.time()
        if now - _last_alert_time < ALERT_COOLDOWN:
            return
        _last_alert_time = now
        # playsound may block on some platforms; run in separate thread if needed
        threading.Thread(target=_play_sound, daemon=True).start()
    except Exception as e:
        print(f"[WARN] play_alert_nonblocking error: {e}")

def _play_sound():
    try:
        if os.path.exists(ALERT_SOUND):
            playsound(ALERT_SOUND)
        else:
            # fallback: macOS 'afplay', Linux 'aplay' (best-effort)
            if os.name == "posix":
                if os.system("which afplay > /dev/null 2>&1") == 0:
                    os.system(f"afplay {ALERT_SOUND} >/dev/null 2>&1")
                elif os.system("which aplay > /dev/null 2>&1") == 0:
                    os.system(f"aplay {ALERT_SOUND} >/dev/null 2>&1")
            # else nothing
    except Exception as e:
        print(f"[WARN] _play_sound failed: {e}")

def save_unknown_snapshot(frame_bgr):
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fn = f"unknown_{now}.jpg"
    path = os.path.join(UNKNOWN_DIR, fn)
    try:
        cv2.imwrite(path, frame_bgr)
        return path
    except Exception as e:
        print(f"[WARN] Saving unknown snapshot failed: {e}")
        return None

# ========== CAMERA & PROCESSING ==========
def get_stream():
    try:
        return imageio_ffmpeg.read_frames(
            RTSP_URL,
            input_params=["-rtsp_transport", "tcp", "-probesize", "32", "-analyzeduration", "0"],
            output_params=["-pix_fmt", "bgr24"]
        )
    except Exception as e:
        print(f"[ERROR] Failed to open RTSP stream: {e}")
        return None

def camera_worker():
    global camera_running, output_frame, known_encodings, known_names
    print("[INFO] Camera worker starting...")
    known_encodings, known_names = load_known_faces()
    print(f"[INFO] Loaded {len(known_encodings)} known faces.")

    while camera_running:
        stream = get_stream()
        if stream is None:
            print("[WARN] Stream open failed, retry in 2s...")
            time.sleep(2)
            continue

        try:
            meta = next(stream)
            width, height = meta["size"]
            frame_size = width * height * 3
            print(f"[INFO] Stream resolution: {width}x{height}")
            for i, frame_bytes in enumerate(stream):
                if not camera_running:
                    break
                if len(frame_bytes) != frame_size:
                    # corrupted frame
                    continue

                # decode & resize
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # zoom crop
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                zw, zh = int(w / ZOOM_FACTOR), int(h / ZOOM_FACTOR)
                x1, y1 = max(0, cx - zw // 2), max(0, cy - zh // 2)
                x2, y2 = min(w, cx + zw // 2), min(h, cy + zh // 2)
                zoomed = frame[y1:y2, x1:x2]
                if zoomed.size == 0:
                    zoomed = frame
                frame = cv2.resize(zoomed, (FRAME_WIDTH, FRAME_HEIGHT))

                # face processing every Nth frame
                if i % FRAME_PROCESS_EVERY == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

                    face_locs = face_recognition.face_locations(small)
                    face_encs = face_recognition.face_encodings(small, face_locs)

                    face_names = []
                    scaled_locs = []

                    for enc, loc in zip(face_encs, face_locs):
                        matches = face_recognition.compare_faces(known_encodings, enc)
                        name = "Unknown"
                        if True in matches:
                            dists = face_recognition.face_distance(known_encodings, enc)
                            best_idx = np.argmin(dists)
                            if matches[best_idx] and dists[best_idx] < FACE_DISTANCE_THRESHOLD:
                                name = known_names[best_idx]

                        # scale back to full frame size (we halved earlier)
                        top, right, bottom, left = loc
                        scaled_locs.append((top * 2, right * 2, bottom * 2, left * 2))
                        face_names.append(name)

                        # attendance or alert + log
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if name != "Unknown":
                            newly_marked = mark_attendance(name)
                            detection_logs.insert(0, {"name": name, "time": ts})
                        else:
                            # save snapshot and play alert
                            snap = save_unknown_snapshot(frame.copy())
                            detection_logs.insert(0, {"name": name, "time": ts, "snapshot": snap})
                            play_alert_nonblocking()

                    # draw boxes + names
                    for (top, right, bottom, left), name in zip(scaled_locs, face_names):
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, name, (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # encode frame to jpeg bytes for web streaming
                ret, buf = cv2.imencode(".jpg", frame)
                if not ret:
                    continue
                jpeg_bytes = buf.tobytes()
                with frame_lock:
                    output_frame = jpeg_bytes

        except Exception as e:
            print(f"[ERROR] Camera loop runtime error: {e}")
            time.sleep(2)

    print("[INFO] Camera worker stopped.")

# ========== FLASK ROUTES ==========
# Simple HTML UI (no external files). It polls attendance/logs every 2s.
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Face Recognition — Dashboard</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;margin:0;padding:20px}
    .card{background:#fff;border-radius:8px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:16px}
    .controls button{margin-right:8px;padding:8px 12px;border-radius:6px;border:0;cursor:pointer}
    .btn-start{background:#16a34a;color:#fff} .btn-stop{background:#dc2626;color:#fff}
    img#video{width:100%;max-width:920px;border-radius:6px;border:1px solid #ddd}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;border-bottom:1px solid #eee;text-align:left}
    th{background:#fafafa}
    .grid{display:grid;grid-template-columns:1fr;gap:16px}
    @media(min-width:900px){ .grid{grid-template-columns: 1fr 420px} }
    .small{font-size:0.9rem;color:#444}
  </style>
</head>
<body>
  <div class="card">
    <h2>Face Recognition Dashboard</h2>
    <div class="controls">
      <button class="btn-start" onclick="startCamera()">Start Camera</button>
      <button class="btn-stop" onclick="stopCamera()">Stop Camera</button>
      <span id="status" class="small">Camera stopped</span>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Live Feed</h3>
      <img id="video" src="/video_feed" alt="Live video feed (MJPEG)">
    </div>

    <div>
      <div class="card">
        <h3>Attendance (today)</h3>
        <table id="attendance_table">
          <thead><tr><th>Name</th><th>Date</th><th>Time</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="card">
        <h3>Detection Logs (recent)</h3>
        <table id="logs_table">
          <thead><tr><th>Name</th><th>Time</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

<script>
async function startCamera(){
  await fetch('/start', {method:'POST'});
  document.getElementById('status').innerText = 'Camera running';
}
async function stopCamera(){
  await fetch('/stop', {method:'POST'});
  document.getElementById('status').innerText = 'Camera stopped';
}
async function fetchAttendance(){
  try{
    const res = await fetch('/attendance');
    const rows = await res.json();
    const tbody = document.querySelector('#attendance_table tbody');
    tbody.innerHTML = '';
    rows.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${r.name}</td><td>${r.date}</td><td>${r.time}</td>`;
      tbody.appendChild(tr);
    });
  }catch(e){ console.warn(e) }
}
async function fetchLogs(){
  try{
    const res = await fetch('/logs');
    const logs = await res.json();
    const tbody = document.querySelector('#logs_table tbody');
    tbody.innerHTML = '';
    logs.slice(0,200).forEach(l => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${l.name}</td><td>${l.time}</td>`;
      tbody.appendChild(tr);
    });
  }catch(e){ console.warn(e) }
}
setInterval(fetchAttendance, 2000);
setInterval(fetchLogs, 2000);
fetchAttendance();
fetchLogs();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/video_feed")
def video_feed():
    def generate():
        global output_frame
        while True:
            with frame_lock:
                if output_frame is None:
                    # small sleep to avoid busy loop
                    time.sleep(0.05)
                    continue
                frame = output_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start", methods=["POST"])
def start_route():
    global camera_running, camera_thread
    with camera_thread_lock:
        if camera_running:
            return jsonify({"status": "already_running"})
        camera_running = True
        camera_thread = threading.Thread(target=camera_worker, daemon=True)
        camera_thread.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop_route():
    global camera_running
    camera_running = False
    return jsonify({"status": "stopped"})

@app.route("/attendance")
def attendance_route():
    rows = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                rows.append({"name": r[0], "date": r[1], "time": r[2]})
    rows = list(reversed(rows))  # newest first
    return jsonify(rows)

@app.route("/logs")
def logs_route():
    # return a copy to avoid race conditions
    return jsonify(detection_logs.copy())



# ========== START APP ==========
if __name__ == "__main__":
    print("[INFO] Starting Flask server on http://0.0.0.0:5000")
    # preload known faces (optional)
    known_encodings, known_names = load_known_faces()
    app.run(host="0.0.0.0", port=5000, debug=False)
