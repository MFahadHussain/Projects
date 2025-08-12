# app.py
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
from flask import Flask, Response, jsonify, request, render_template, send_from_directory

# ========== CONFIG ==========
RTSP_URL = 0 #"rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/101"
KNOWN_FACES_DIR = "images"
SNAPSHOT_DIR_KNOWN = os.path.join("static", "snapshots", "known")
SNAPSHOT_DIR_UNKNOWN = os.path.join("static", "snapshots", "unknown")
ZOOM_FACTOR = 1.3 #1.4
ALERT_SOUND = "alert.wav"        # optional
ATTENDANCE_FILE = "attendance.csv"
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FRAME_PROCESS_EVERY = 5
FACE_DISTANCE_THRESHOLD = 0.55
ALERT_COOLDOWN = 5.0

# ========== SETUP ==========
app = Flask(__name__, static_folder="static")
os.makedirs(SNAPSHOT_DIR_KNOWN, exist_ok=True)
os.makedirs(SNAPSHOT_DIR_UNKNOWN, exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time"])

# Globals
output_frame = None
frame_lock = threading.Lock()
camera_running = False
camera_thread = None
camera_thread_lock = threading.Lock()

# Face DB + logs
known_encodings = []
known_names = []
attendance_today = set()
detection_logs = []  # newest first

# Alert throttle
_last_alert_time = 0.0

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
    if name in attendance_today:
        return False
    now = datetime.now()
    ds = now.strftime("%Y-%m-%d")
    ts = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, ds, ts])
    attendance_today.add(name)
    detection_logs.insert(0, {"name": name, "time": f"{ds} {ts}"})
    print(f"[ATTEND] {name} @ {ts}")
    return True

def _play_sound_blocking():
    try:
        if os.path.exists(ALERT_SOUND):
            playsound(ALERT_SOUND)
    except Exception as e:
        print(f"[WARN] playsound failed: {e}")

def play_alert_nonblocking():
    global _last_alert_time
    try:
        now = time.time()
        if now - _last_alert_time < ALERT_COOLDOWN:
            return
        _last_alert_time = now
        threading.Thread(target=_play_sound_blocking, daemon=True).start()
    except Exception as e:
        print(f"[WARN] play_alert_nonblocking error: {e}")

def save_snapshot(frame_bgr, known=True, name=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if known:
        fn = f"{name}_{ts}.jpg" if name else f"known_{ts}.jpg"
        path = os.path.join(SNAPSHOT_DIR_KNOWN, fn)
    else:
        fn = f"unknown_{ts}.jpg"
        path = os.path.join(SNAPSHOT_DIR_UNKNOWN, fn)
    try:
        cv2.imwrite(path, frame_bgr)
        return fn
    except Exception as e:
        print(f"[WARN] failed saving snapshot: {e}")
        return None

# ========== CAMERA STREAMING & PROCESSING ==========
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
                    continue

                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # Zoom region
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                zw, zh = int(w / ZOOM_FACTOR), int(h / ZOOM_FACTOR)
                x1, y1 = max(0, cx - zw // 2), max(0, cy - zh // 2)
                x2, y2 = min(w, cx + zw // 2), min(h, cy + zw // 2)
                zoomed = frame[y1:y2, x1:x2]
                if zoomed.size == 0:
                    zoomed = frame
                frame = cv2.resize(zoomed, (FRAME_WIDTH, FRAME_HEIGHT))

                # Process faces every N frames
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

                        top, right, bottom, left = loc
                        scaled_locs.append((top * 2, right * 2, bottom * 2, left * 2))
                        face_names.append(name)

                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if name != "Unknown":
                            mark_attendance(name)
                            save_snapshot(frame.copy(), known=True, name=name)
                            detection_logs.insert(0, {"name": name, "time": ts})
                        else:
                            saved = save_snapshot(frame.copy(), known=False)
                            detection_logs.insert(0, {"name": name, "time": ts, "snapshot": saved})
                            play_alert_nonblocking()

                    # Draw boxes and put names
                    for (top, right, bottom, left), name in zip(scaled_locs, face_names):
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, name, (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # encode for MJPEG streaming
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



INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Face Recognition — Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap 5 CDN -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body{background:#f3f4f6;padding:20px;font-family:Inter,Arial,Helvetica,sans-serif}
    .card{border-radius:8px}
    img#video{width:100%;height:auto;border-radius:6px;border:1px solid #ddd}
    .snapshot-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
    .snapshot-grid img{width:100%;border-radius:6px;cursor:pointer;object-fit:cover;height:120px}
    .tab-pane{margin-top:12px}
    .small-muted{font-size:0.85rem;color:#666}
    .search-input{max-width:320px}
  </style>
</head>
<body>
  <div class="container">
    <div class="card p-3 mb-3">
      <div class="d-flex align-items-center justify-content-between">
        <h4 class="mb-0">Face Recognition Dashboard</h4>
        <div>
          <button id="btnStart" class="btn btn-success btn-sm me-2" onclick="startCamera()">Start Camera</button>
          <button id="btnStop" class="btn btn-danger btn-sm me-3" onclick="stopCamera()">Stop Camera</button>
          <span id="status" class="small-muted">Camera stopped</span>
        </div>
      </div>

      <ul class="nav nav-tabs mt-3" role="tablist">
        <li class="nav-item" role="presentation">
          <button class="nav-link active" id="feed-tab" data-bs-toggle="tab" data-bs-target="#feed" type="button" role="tab">Live Feed</button>
        </li>
        <li class="nav-item" role="presentation">
          <button class="nav-link" id="att-tab" data-bs-toggle="tab" data-bs-target="#attendance" type="button" role="tab">Attendance & Logs</button>
        </li>
        <li class="nav-item" role="presentation">
          <button class="nav-link" id="snap-tab" data-bs-toggle="tab" data-bs-target="#snapshots" type="button" role="tab">Snapshots</button>
        </li>
        <li class="nav-item" role="presentation">
          <button class="nav-link" id="unk-tab" data-bs-toggle="tab" data-bs-target="#unknowns" type="button" role="tab">Unknowns</button>
        </li>
      </ul>
    </div>

    <div class="tab-content">
      <!-- Live Feed tab -->
      <div class="tab-pane fade show active" id="feed" role="tabpanel">
        <div class="row g-3">
          <div class="col-lg-8">
            <div class="card p-3">
              <h6>Live Feed</h6>
              <img id="video" src="/video_feed" alt="Live feed (MJPEG)">
            </div>
          </div>

          <div class="col-lg-4">
            <div class="card p-3 mb-3">
              <h6>Attendance (latest)</h6>
              <div style="max-height:260px;overflow:auto">
                <table class="table table-sm">
                  <thead><tr><th>Name</th><th>Time</th></tr></thead>
                  <tbody id="attendance_small"></tbody>
                </table>
              </div>
            </div>

            <div class="card p-3">
              <h6>Logs (recent)</h6>
              <div style="max-height:260px;overflow:auto">
                <table class="table table-sm">
                  <thead><tr><th>Name</th><th>Time</th></tr></thead>
                  <tbody id="logs_small"></tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Attendance & Logs tab -->
      <div class="tab-pane fade" id="attendance" role="tabpanel">
        <div class="card p-3">
          <div class="d-flex justify-content-between mb-2">
            <h6 class="mb-0">Attendance (all)</h6>
            <input id="searchAttendance" class="form-control form-control-sm search-input" placeholder="Search name..." oninput="filterAttendance()">
          </div>
          <div style="max-height:420px;overflow:auto">
            <table class="table table-striped">
              <thead><tr><th>Name</th><th>Date</th><th>Time</th></tr></thead>
              <tbody id="attendance_full"></tbody>
            </table>
          </div>
        </div>

        <div class="card p-3 mt-3">
          <h6 class="mb-2">Detection Logs (full)</h6>
          <div style="max-height:320px;overflow:auto">
            <table class="table table-striped">
              <thead><tr><th>Name</th><th>Time</th></tr></thead>
              <tbody id="logs_full"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Snapshots tab -->
      <div class="tab-pane fade" id="snapshots" role="tabpanel">
        <div class="card p-3">
          <h6>All Snapshots (known + unknown)</h6>
          <div id="snapshot_grid_all" class="snapshot-grid mt-3"></div>
        </div>
      </div>

      <!-- Unknowns tab -->
      <div class="tab-pane fade" id="unknowns" role="tabpanel">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <h6 class="mb-0">Unknown Snapshots</h6>
            <button class="btn btn-sm btn-outline-secondary" onclick="fetchUnknowns()">Refresh</button>
          </div>
          <div id="snapshot_grid_unknown" class="snapshot-grid"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Modal for click-to-enlarge -->
  <div class="modal fade" id="imgModal" tabindex="-1">
    <div class="modal-dialog modal-fullscreen-sm-down modal-lg modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-body p-0">
          <img id="modalImg" src="" style="width:100%;height:auto;display:block">
        </div>
      </div>
    </div>
  </div>

  <!-- Bootstrap JS -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

  <script>
    // Basic functions
    async function startCamera(){
      await fetch('/start',{method:'POST'});
      document.getElementById('status').innerText = 'Camera running';
    }
    async function stopCamera(){
      await fetch('/stop',{method:'POST'});
      document.getElementById('status').innerText = 'Camera stopped';
    }

    // Fetch attendance & logs
    async function fetchAttendanceSmall(){
      try{
        const res = await fetch('/attendance_data');
        const rows = await res.json();
        const tbody = document.getElementById('attendance_small');
        tbody.innerHTML = '';
        rows.slice(0,15).forEach(r => {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${r.name}</td><td>${r.time}</td>`;
          tbody.appendChild(tr);
        });
      }catch(e){ console.warn(e) }
    }
    async function fetchLogsSmall(){
      try{
        const res = await fetch('/logs_data');
        const logs = await res.json();
        const tbody = document.getElementById('logs_small');
        tbody.innerHTML = '';
        logs.slice(0,20).forEach(l => {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${l.name}</td><td>${l.time}</td>`;
          tbody.appendChild(tr);
        });
      }catch(e){ console.warn(e) }
    }

    async function fetchAttendanceFull(){
      try{
        const res = await fetch('/attendance_data');
        const rows = await res.json();
        const tbody = document.getElementById('attendance_full');
        tbody.innerHTML = '';
        rows.forEach(r => {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${r.name}</td><td>${r.date}</td><td>${r.time}</td>`;
          tbody.appendChild(tr);
        });
      }catch(e){ console.warn(e) }
    }

    async function fetchLogsFull(){
      try{
        const res = await fetch('/logs_data');
        const logs = await res.json();
        const tbody = document.getElementById('logs_full');
        tbody.innerHTML = '';
        logs.forEach(l => {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${l.name}</td><td>${l.time}</td>`;
          tbody.appendChild(tr);
        });
      }catch(e){ console.warn(e) }
    }

    // Snapshots
    async function fetchUnknowns(){
      try{
        const res = await fetch('/unknowns_data');
        const snaps = await res.json();
        const grid = document.getElementById('snapshot_grid_unknown');
        grid.innerHTML = '';
        snaps.forEach(s => {
          const img = document.createElement('img');
          img.src = `/snapshots/unknown/${s.filename}`;
          img.alt = s.filename;
          img.title = s.time;
          img.onclick = () => showModal(img.src);
          grid.appendChild(img);
        });
      }catch(e){ console.warn(e) }
    }

    async function fetchAllSnapshots(){
      try{
        // known
        const knownDir = '/static/snapshots/known';
        // unknown handled via /unknowns_data
        // We'll list known files by trying to request the directory listing via unknowns_data for unknowns,
        // and for known we simply fetch filenames by attempting a small trick: server doesn't have known list API --
        // but we can show unknowns + recent known saved in detection_logs (which reference saved filenames optionally).
        // To keep it simple and reliable, we'll request unknowns_data and also show known snapshots by reading detection logs (client-side).
        const res = await fetch('/unknowns_data');
        const unknowns = await res.json();
        const grid = document.getElementById('snapshot_grid_all');
        grid.innerHTML = '';
        // show unknowns first
        unknowns.forEach(s => {
          const img = document.createElement('img');
          img.src = `/snapshots/unknown/${s.filename}`;
          img.alt = s.filename;
          img.title = s.time;
          img.onclick = () => showModal(img.src);
          grid.appendChild(img);
        });
        // Add recent known snapshots from logs (best-effort)
        const logsRes = await fetch('/logs_data');
        const logs = await logsRes.json();
        // If logs entries include snapshot filename, show them (some entries do for unknowns)
        logs.slice(0,50).forEach(l => {
          if (l.snapshot) {
            const img = document.createElement('img');
            img.src = `/snapshots/unknown/${l.snapshot}`;
            img.alt = l.snapshot;
            img.title = l.time;
            img.onclick = () => showModal(img.src);
            grid.appendChild(img);
          }
        });
      }catch(e){ console.warn(e) }
    }

    function showModal(src){
      const modalImg = document.getElementById('modalImg');
      modalImg.src = src;
      const modal = new bootstrap.Modal(document.getElementById('imgModal'));
      modal.show();
    }

    function filterAttendance(){
      const q = document.getElementById('searchAttendance').value.toLowerCase();
      document.querySelectorAll('#attendance_full tbody tr').forEach(tr => {
        const name = tr.children[0].innerText.toLowerCase();
        tr.style.display = name.includes(q) ? '' : 'none';
      });
    }

    // Polling intervals
    setInterval(() => { fetchAttendanceSmall(); fetchLogsSmall(); }, 2500);
    setInterval(() => { fetchUnknowns(); fetchAllSnapshots(); }, 5000);

    // Initial load
    fetchAttendanceSmall();
    fetchLogsSmall();
    fetchAttendanceFull();
    fetchLogsFull();
    fetchUnknowns();
    fetchAllSnapshots();
  </script>
</body>
</html>

"""
# ========== FLASK ROUTES ==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        global output_frame
        while True:
            with frame_lock:
                if output_frame is None:
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

@app.route("/attendance_data")
def attendance_data():
    rows = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                rows.append({"name": r[0], "date": r[1], "time": r[2]})
    rows = list(reversed(rows))
    return jsonify(rows)

@app.route("/logs_data")
def logs_data():
    return jsonify(detection_logs.copy())

@app.route("/unknowns_data")
def unknowns_data():
    files = []
    for fn in os.listdir(SNAPSHOT_DIR_UNKNOWN):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(SNAPSHOT_DIR_UNKNOWN, fn)
            ts = os.path.getmtime(path)
            files.append({"filename": fn, "time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")})
    files.sort(key=lambda x: x["time"], reverse=True)
    return jsonify(files)

@app.route("/snapshots/known/<filename>")
def serve_known_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR_KNOWN, filename)

@app.route("/snapshots/unknown/<filename>")
def serve_unknown_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR_UNKNOWN, filename)

# ========== MAIN ==========
if __name__ == "__main__":
    print("[INFO] Loading known faces (startup)...")
    known_encodings, known_names = load_known_faces()
    print(f"[INFO] {len(known_encodings)} faces loaded.")
    print("[INFO] Starting Flask on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
