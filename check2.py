import cv2
import numpy as np
import os
import time
import face_recognition
import imageio_ffmpeg
import csv
from datetime import datetime
from playsound import playsound  # pip install playsound==1.2.2

# === CONFIG ===
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.125:554/Streaming/channels/101"
KNOWN_FACES_DIR = "images"
ZOOM_FACTOR = 1.4
ALERT_SOUND = "alert.wav"  # Make sure this file exists
ATTENDANCE_FILE = "attendance.csv"

# === LOAD KNOWN FACES ===
print("[INFO] Loading known faces...")
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f"[WARN] No face found in: {filename}")

print(f"[INFO] Loaded {len(known_encodings)} known faces.")

# === ATTENDANCE SYSTEM ===
attendance_today = set()
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    if name not in attendance_today:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        with open(ATTENDANCE_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        attendance_today.add(name)
        print(f"[ATTENDANCE] {name} marked at {time_str}")

# === ALERT SYSTEM ===
last_alert_time = 0
ALERT_COOLDOWN = 5  # seconds

def play_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time > ALERT_COOLDOWN:
        last_alert_time = now
        try:
            playsound(ALERT_SOUND, block=False)
        except Exception as e:
            print(f"[WARN] Could not play alert sound: {e}")

# === GET VIDEO STREAM ===
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

# === MAIN LOOP ===
while True:
    print("[INFO] Connecting to RTSP stream...")
    stream = get_stream()
    if stream is None:
        time.sleep(2)
        continue

    try:
        meta = next(stream)
        width, height = meta["size"]
        frame_size = width * height * 3
        print(f"[INFO] Resolution: {width}x{height}")

        for i, frame_bytes in enumerate(stream):
            if len(frame_bytes) != frame_size:
                print("[WARN] Skipping corrupted frame")
                continue

            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
            frame = cv2.resize(frame, (640, 360))

            # === ZOOM ===
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            zw, zh = int(w / ZOOM_FACTOR), int(h / ZOOM_FACTOR)
            x1, y1 = cx - zw // 2, cy - zh // 2
            x2, y2 = cx + zw // 2, cy + zh // 2
            zoomed = frame[y1:y2, x1:x2]
            frame = cv2.resize(zoomed, (640, 360))

            # === FACE DETECTION === (every 5th frame)
            if i % 5 == 0:
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
                        face_distances = face_recognition.face_distance(known_encodings, enc)
                        best_idx = np.argmin(face_distances)
                        if matches[best_idx] and face_distances[best_idx] < 0.55:
                            name = known_names[best_idx]

                    face_names.append(name)
                    top, right, bottom, left = loc
                    scaled_locs.append((top * 2, right * 2, bottom * 2, left * 2))

                    # Attendance or Alert
                    if name != "Unknown":
                        mark_attendance(name)
                    else:
                        play_alert()

            # === DRAW BOXES ===
            if 'face_names' in locals():
                for (top, right, bottom, left), name in zip(scaled_locs, face_names):
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # === DISPLAY ===
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("[INFO] Exiting.")
        break
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        time.sleep(3)
        continue

cv2.destroyAllWindows()
