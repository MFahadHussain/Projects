from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
logs = []

# Load known faces
known_face_encodings = []
known_face_names = []
for filename in os.listdir("images"):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = face_recognition.load_image_file(f"images/{filename}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

camera = cv2.VideoCapture(0)

def gen_frames():
    global logs
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize and convert
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                        name = known_face_names[best_match_index]

                face_names.append(name)
                if name != "Unknown":
                    logs.append({"name": name, "time": datetime.now().strftime("%H:%M:%S")})
                    logs = logs[-10:]  # Limit logs

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    return jsonify(logs)

if __name__ == '__main__':
    app.run(debug=True)
