# face_attendance.py
import cv2
import os
import time
import sqlite3
import threading
from datetime import datetime
from flask import Flask, render_template_string, send_from_directory
import face_recognition

# ----------------------------
# CONFIG
# ----------------------------
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/101"
DB_PATH = "attendance.db"
UNKNOWN_DIR = "unknown_faces"
KNOWN_DIR = "known_faces"

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ----------------------------
# DATABASE SETUP
# ----------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp TEXT
)
""")
conn.commit()

# ----------------------------
# LOAD KNOWN FACES
# ----------------------------
known_face_encodings = []
known_face_names = []

for file in os.listdir(KNOWN_DIR):
    if file.endswith((".jpg", ".png")):
        img = face_recognition.load_image_file(os.path.join(KNOWN_DIR, file))
        enc = face_recognition.face_encodings(img)
        if enc:
            known_face_encodings.append(enc[0])
            known_face_names.append(os.path.splitext(file)[0])

print(f"[INFO] Loaded {len(known_face_names)} known faces.")

# ----------------------------
# CAMERA PROCESSING THREAD
# ----------------------------
def camera_loop():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame received.")
            time.sleep(0.5)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                # Log attendance
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
                conn.commit()
                print(f"[LOG] {name} at {timestamp}")

            else:
                # Save unknown face
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_img = frame[top:bottom, left:right]
                cv2.imwrite(os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg"), face_img)
                print("[INFO] Unknown face captured.")

        # Show preview (optional)
        cv2.imshow("RTSP Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# FLASK WEB DASHBOARD
# ----------------------------
app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Attendance Dashboard</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; }
        img { height: 100px; }
    </style>
</head>
<body>
    <h1>Attendance Logs</h1>
    <table>
        <tr><th>ID</th><th>Name</th><th>Timestamp</th></tr>
        {% for row in logs %}
        <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td></tr>
        {% endfor %}
    </table>

    <h1>Unknown Faces</h1>
    {% for img in unknowns %}
        <img src="/unknown/{{ img }}">
    {% endfor %}
</body>
</html>
"""

@app.route("/")
def dashboard():
    c.execute("SELECT * FROM attendance ORDER BY id DESC LIMIT 20")
    logs = c.fetchall()
    unknowns = sorted(os.listdir(UNKNOWN_DIR), reverse=True)
    return render_template_string(TEMPLATE, logs=logs, unknowns=unknowns)

@app.route("/unknown/<filename>")
def unknown_file(filename):
    return send_from_directory(UNKNOWN_DIR, filename)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
