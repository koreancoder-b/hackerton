from ultralytics import YOLO
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import numpy as np
import collections

# EAR (Eye Aspect Ratio) 계산을 위한 6포인트 랜드마크 인덱스
# 순서: [외측, 상외, 상내, 내측, 하내, 하외]
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_CLOSED_THRESH   = 0.20   # EAR 이 이하면 눈 감음 (열린 눈: ~0.25~0.35)
BLINK_SCORE_THRESH  = 0.55   # 블렌드쉐이프 eyeBlink 점수 임계값
EAR_SMOOTH_FRAMES   = 5      # EAR 롤링 평균 프레임 수
SLEEP_CLOSED_FRAMES = 15     # 연속 N프레임 이상 감기면 SLEEPING (~0.5초 @30fps)
AWAY_THRESHOLD      = 15     # 얼굴 미검출 허용 프레임


def _ear(lms, indices, w, h):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
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
    # 눈 외곽선 그리기
    hull = cv2.convexHull(np.array(pts))
    cv2.polylines(frame, [hull], True, color, 1)


class PhoneDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, conf=0.3, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 67:
                    return True, box.xyxy[0]
        return False, None


class FaceAnalyzer:
    def __init__(self):
        model_path = "./models/face_landmarker.task"
        if not os.path.exists(model_path):
            os.makedirs("./models", exist_ok=True)
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
            output_face_blendshapes=True,  # eyeBlinkLeft / eyeBlinkRight 활성화
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

        self.no_face_counter = 0
        self.last_status = "STUDYING"

        # EAR 롤링 평균 버퍼
        self.left_ear_buf  = collections.deque(maxlen=EAR_SMOOTH_FRAMES)
        self.right_ear_buf = collections.deque(maxlen=EAR_SMOOTH_FRAMES)
        # 연속 눈 감김 카운터
        self.closed_frames = 0

    def _is_eye_closed(self, smooth_ear, blink_score):
        closed_by_ear = smooth_ear < EAR_CLOSED_THRESH
        if blink_score is not None:
            closed_by_blend = blink_score > BLINK_SCORE_THRESH
            return closed_by_ear or closed_by_blend
        return closed_by_ear

    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        # 얼굴 없음 처리
        if not results.face_landmarks:
            self.no_face_counter += 1
            if self.no_face_counter < AWAY_THRESHOLD:
                return self.last_status, self.last_status == "STUDYING", frame
            self.closed_frames = 0
            self.last_status = "AWAY"
            return "AWAY", False, frame

        self.no_face_counter = 0
        lms = results.face_landmarks[0]

        # 얼굴 바운딩 박스
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        ox, oy = max(0, int(min(xs))), max(0, int(min(ys)))
        ow = min(w - ox, int(max(xs)) - ox)
        oh = min(h - oy, int(max(ys)) - oy)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 255, 0), 2)
        cv2.putText(frame, "Face", (ox, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        # EAR 계산 + 롤링 평균
        left_ear  = _ear(lms, LEFT_EYE_IDX,  w, h)
        right_ear = _ear(lms, RIGHT_EYE_IDX, w, h)
        self.left_ear_buf.append(left_ear)
        self.right_ear_buf.append(right_ear)
        smooth_left  = sum(self.left_ear_buf)  / len(self.left_ear_buf)
        smooth_right = sum(self.right_ear_buf) / len(self.right_ear_buf)

        # 블렌드쉐이프 보조 신호
        blink_l = _blendshape_score(results.face_blendshapes, "eyeBlinkLeft")
        blink_r = _blendshape_score(results.face_blendshapes, "eyeBlinkRight")

        left_closed  = self._is_eye_closed(smooth_left,  blink_l)
        right_closed = self._is_eye_closed(smooth_right, blink_r)

        # 눈 랜드마크 시각화
        _draw_eye_landmarks(frame, lms, LEFT_EYE_IDX,  w, h,
                            (0, 0, 255) if left_closed  else (0, 255, 0))
        _draw_eye_landmarks(frame, lms, RIGHT_EYE_IDX, w, h,
                            (0, 0, 255) if right_closed else (0, 255, 0))

        # EAR 수치 표시
        cv2.putText(frame,
                    f"EAR  L:{smooth_left:.3f}  R:{smooth_right:.3f}  (th:{EAR_CLOSED_THRESH})",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 블렌드쉐이프 수치 표시
        if blink_l is not None and blink_r is not None:
            cv2.putText(frame,
                        f"Blink L:{blink_l:.2f}  R:{blink_r:.2f}",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 연속 감김 카운터
        if left_closed and right_closed:
            self.closed_frames += 1
        else:
            self.closed_frames = 0  # 한 쪽이라도 열리면 리셋 (깜빡임은 무시됨)

        if self.closed_frames >= SLEEP_CLOSED_FRAMES:
            self.last_status = "SLEEPING"
            return "SLEEPING", False, frame

        self.last_status = "STUDYING"
        return "STUDYING", True, frame


# ==================== 메인 루프 ====================

phone_engine = PhoneDetector()
face_engine  = FaceAnalyzer()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("FocusGuardian v2 시작... (종료: q 키)")

while True:
    success, frame = cap.read()
    if not success:
        print("프레임을 읽을 수 없습니다.")
        break

    has_phone, phone_box = phone_engine.detect(frame)
    status, is_studying, frame = face_engine.analyze(frame)

    if has_phone and phone_box is not None:
        x1, y1, x2, y2 = [int(v) for v in phone_box.tolist()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, "PHONE", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    color_map = {
        "STUDYING": (0, 255, 0),
        "SLEEPING": (0, 165, 255),
        "AWAY":     (0, 0, 255),
    }
    color = color_map.get(status, (255, 255, 255))
    cv2.putText(frame, f"Status: {status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("FocusGuardian v2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("종료되었습니다.")
