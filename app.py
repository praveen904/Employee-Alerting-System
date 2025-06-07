from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Constants for drowsiness detection
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False
ATTENTION_ALARM_ON = False
NO_FACE_COUNTER = 0
NO_FACE_CONSEC_FRAMES = 20
NO_FACE_ALARM_ON = False

def detect_drowsiness(frame):
    global COUNTER, ALARM_ON, ATTENTION_ALARM_ON, NO_FACE_COUNTER, NO_FACE_ALARM_ON
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        NO_FACE_COUNTER += 1
        if NO_FACE_COUNTER == NO_FACE_CONSEC_FRAMES and not NO_FACE_ALARM_ON:
            NO_FACE_ALARM_ON = True
            socketio.emit('drowsiness_alert', {'status': 'out_of_frame', 'type': 'face'})
    else:
        if NO_FACE_ALARM_ON:
            NO_FACE_ALARM_ON = False
            socketio.emit('drowsiness_alert', {'status': 'in_frame', 'type': 'face'})
        NO_FACE_COUNTER = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        # Attention detection (face position)
        face_center_x = x + w/2
        frame_center_x = frame.shape[1]/2
        face_center_y = y + h/2
        frame_center_y = frame.shape[0]/2
        x_offset = abs(face_center_x - frame_center_x) / frame_center_x
        y_offset = abs(face_center_y - frame_center_y) / frame_center_y
        if x_offset > 0.3 or y_offset > 0.3:
            if not ATTENTION_ALARM_ON:
                ATTENTION_ALARM_ON = True
                socketio.emit('drowsiness_alert', {'status': 'inattentive', 'type': 'attention'})
        else:
            if ATTENTION_ALARM_ON:
                ATTENTION_ALARM_ON = False
                socketio.emit('drowsiness_alert', {'status': 'attentive', 'type': 'attention'})
        # Drowsiness detection: if fewer than 2 eyes detected for 20 frames
        if len(eyes) >= 2:
            COUNTER = 0
            if ALARM_ON:
                ALARM_ON = False
                socketio.emit('drowsiness_alert', {'status': 'awake', 'type': 'eyes'})
        else:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    socketio.emit('drowsiness_alert', {'status': 'drowsy', 'type': 'eyes'})
        # Draw rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_drowsiness(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True) 