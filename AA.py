import cv2
import face_recognition
import numpy as np
import time
import os
from datetime import datetime
import sqlite3

RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/102?tcp"

conn = sqlite3.connect("attendance.db")
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp TEXT
)
''')
conn.commit()

known_face_encodings = []
known_face_names = []

for filename in os.listdir("known_faces"):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = face_recognition.load_image_file(os.path.join("known_faces", filename))
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

def mark_attendance(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("SELECT * FROM attendance WHERE name=? ORDER BY timestamp DESC LIMIT 1", (name,))
    last = c.fetchone()
    if last:
        last_time = datetime.strptime(last[2], "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - last_time).total_seconds() < 60:
            return
    c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
    conn.commit()
    print(f"[{timestamp}] Attendance marked for {name}")

def main():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error opening RTSP stream")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            time.sleep(0.5)
            continue

        frame_count += 1
        if frame_count % 3 != 0:
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                mark_attendance(name)
            else:
                print("Unauthorized detected!")

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
