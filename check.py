import cv2
import numpy as np
import imageio_ffmpeg as ffmpeg
import face_recognition
import threading
import time
import os
from flask import Flask, render_template, Response

# === CONFIGURATION ===
rtsp_url = "rtsp://admin:afaqkhan-1@192.168.18.125:554/Streaming/Channels/102"  # use substream for stability
known_faces_dir = "images"
zoom_factor = 1.5

# === GLOBAL VARIABLES ===
output_frame = None
lock = threading.Lock()

# === FLASK APP ===
app = Flask(__name__)

# === LOAD KNOWN FACES ===
def load_known_faces(folder):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"[WARN] No face found in {filename}")
    return known_face_encodings, known_face_names

# === GET STREAM USING FFMPEG ===
def get_stream(url):
    try:
        return ffmpeg.read_frames(
            url,
            input_params=["-rtsp_transport", "tcp", "-analyzeduration", "0", "-probesize", "32"],
            output_params=["-pix_fmt", "bgr24"]
        )
    except Exception as e:
        print(f"[ERROR] Failed to open RTSP stream: {e}")
        return None

# === FACE RECOGNITION THREAD ===
def face_recognition_thread():
    global output_frame

    print("[INFO] Loading known faces...")
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    print(f"[INFO] Loaded {len(known_face_encodings)} known faces.")

    while True:
        print("[INFO] Connecting to Hikvision camera...")
        process = get_stream(rtsp_url)
        if process is None:
            time.sleep(2)
            continue

        try:
            meta = next(process)  # Stream info
            width, height = meta["size"]
            frame_size = width * height * 3
            print(f"[INFO] Stream resolution: {width}x{height}")

            frame_count = 0
            for frame_bytes in process:
                if len(frame_bytes) != frame_size:
                    continue

                # Decode frame
                frame = np.frombuffer(frame_bytes, np.uint8).reshape((height, width, 3))
                frame = cv2.resize(frame, (640, 360))

                # Zoom
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                zoom_w, zoom_h = int(w / zoom_factor), int(h / zoom_factor)
                x1, y1 = cx - zoom_w // 2, cy - zoom_h // 2
                x2, y2 = cx + zoom_w // 2, cy + zoom_h // 2
                cropped_frame = frame[y1:y2, x1:x2]
                frame = cv2.resize(cropped_frame, (640, 360))

                # Process every 5th frame for performance
                if frame_count % 5 == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_small)
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                        face_locations = [(top * 2, right * 2, bottom * 2, left * 2)
                                          for (top, right, bottom, left) in face_locations]

                        face_names = []
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            if matches:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index] and face_distances[best_match_index] < 0.55:
                                    name = known_face_names[best_match_index]
                            face_names.append(name)

                        # Draw boxes
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
                            cv2.putText(frame, name, (left + 6, bottom - 6),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                # Save frame for web streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                with lock:
                    output_frame = buffer.tobytes()

                frame_count += 1

        except Exception as e:
            print(f"[ERROR] Runtime error: {e}")
            time.sleep(3)

# === FLASK ROUTES ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       output_frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === START APP ===
if __name__ == '__main__':
    threading.Thread(target=face_recognition_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
