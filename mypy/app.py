from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import urllib.request
import collections

LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_CLOSED_THRESH   = 0.20
BLINK_SCORE_THRESH  = 0.55
EAR_SMOOTH_FRAMES   = 5
SLEEP_CLOSED_FRAMES = 15
AWAY_THRESHOLD      = 15


def _ear(lms, indices, w, h):
    pts = [np.array([lms[i].x * w, lms[i].y * h]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0


def _blendshape_score(blendshapes, name):
    if not blendshapes:
        return None
    for bs in blendshapes[0]:
        if bs.category_name == name:
            return bs.score
    return None


def _draw_eye_landmarks(frame, lms, indices, w, h, color):
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1)
    hull = cv2.convexHull(np.array(pts))
    cv2.polylines(frame, [hull], True, color, 1)


class PhoneDetector:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, conf=0.3, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 67:
                    return True, box.xyxy[0]
        return False, None


class FaceAnalyzer:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "models", "face_landmarker.task")
        if not os.path.exists(model_path):
            os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
            print("얼굴 랜드마크 모델 다운로드 중... (약 30MB)")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                model_path
            )
            print("다운로드 완료!")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)
        self.no_face_counter = 0
        self.last_status = "STUDYING"
        self.left_ear_buf  = collections.deque(maxlen=EAR_SMOOTH_FRAMES)
        self.right_ear_buf = collections.deque(maxlen=EAR_SMOOTH_FRAMES)
        self.closed_frames = 0

    def _is_eye_closed(self, smooth_ear, blink_score):
        closed_by_ear = smooth_ear < EAR_CLOSED_THRESH
        if blink_score is not None:
            return closed_by_ear or (blink_score > BLINK_SCORE_THRESH)
        return closed_by_ear

    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            self.no_face_counter += 1
            if self.no_face_counter < AWAY_THRESHOLD:
                return self.last_status, frame
            self.closed_frames = 0
            self.last_status = "AWAY"
            return "AWAY", frame

        self.no_face_counter = 0
        lms = results.face_landmarks[0]

        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        ox, oy = max(0, int(min(xs))), max(0, int(min(ys)))
        ow = min(w - ox, int(max(xs)) - ox)
        oh = min(h - oy, int(max(ys)) - oy)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 255, 0), 2)
        cv2.putText(frame, "Face", (ox, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        left_ear  = _ear(lms, LEFT_EYE_IDX,  w, h)
        right_ear = _ear(lms, RIGHT_EYE_IDX, w, h)
        self.left_ear_buf.append(left_ear)
        self.right_ear_buf.append(right_ear)
        smooth_left  = sum(self.left_ear_buf)  / len(self.left_ear_buf)
        smooth_right = sum(self.right_ear_buf) / len(self.right_ear_buf)

        blink_l = _blendshape_score(results.face_blendshapes, "eyeBlinkLeft")
        blink_r = _blendshape_score(results.face_blendshapes, "eyeBlinkRight")

        left_closed  = self._is_eye_closed(smooth_left,  blink_l)
        right_closed = self._is_eye_closed(smooth_right, blink_r)

        _draw_eye_landmarks(frame, lms, LEFT_EYE_IDX,  w, h,
                            (0, 0, 255) if left_closed  else (0, 255, 0))
        _draw_eye_landmarks(frame, lms, RIGHT_EYE_IDX, w, h,
                            (0, 0, 255) if right_closed else (0, 255, 0))

        cv2.putText(frame,
                    f"EAR  L:{smooth_left:.3f}  R:{smooth_right:.3f}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if left_closed and right_closed:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        if self.closed_frames >= SLEEP_CLOSED_FRAMES:
            self.last_status = "SLEEPING"
            return "SLEEPING", frame

        self.last_status = "STUDYING"
        return "STUDYING", frame


app = Flask(__name__)
app.config['SECRET_KEY'] = 'focus_guardian_web_secret'
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
    async_mode='threading'
)

phone_engine = None
face_engine  = None


def get_engines():
    global phone_engine, face_engine
    if phone_engine is None:
        phone_engine = PhoneDetector()
    if face_engine is None:
        face_engine = FaceAnalyzer()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    get_engines()
    emit('connected', {'message': '모델 준비 완료'})


@socketio.on('frame')
def handle_frame(data):
    try:
        raw = data['image'].split(',', 1)
        img_bytes = base64.b64decode(raw[1] if len(raw) == 2 else raw[0])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        has_phone, phone_box = phone_engine.detect(frame)
        status, frame = face_engine.analyze(frame)

        if has_phone and phone_box is not None:
            x1, y1, x2, y2 = [int(v) for v in phone_box.tolist()]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "PHONE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        color_map = {
            "STUDYING": (0, 200, 80),
            "SLEEPING": (0, 140, 255),
            "AWAY":     (50, 50, 220),
        }
        cv2.putText(frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_map.get(status, (255, 255, 255)), 3)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        encoded = base64.b64encode(buf).decode('utf-8')

        emit('result', {
            'status': status,
            'has_phone': has_phone,
            'frame': f'data:image/jpeg;base64,{encoded}',
        })
    except Exception as exc:
        emit('error', {'message': str(exc)})


if __name__ == '__main__':
    print("=" * 50)
    print("  FocusGuardian Web 시작 중...")
    print("  브라우저에서 http://localhost:5000 을 여세요")
    print("=" * 50)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
