from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os


class PhoneDetector:
    def __init__(self, model_path="./models/best.pt"):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, conf=0.02, verbose=False)
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
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

        self.no_face_counter = 0
        self.AWAY_THRESHOLD = 15
        self.last_status = "STUDYING"

        # 👁 임계값 — 값이 클수록 "눈 떴다" 판정이 엄격해짐
        # 실제 출력값 보고 조정 필요 (눈 뜰 때 보통 0.02~0.04)
        self.EYE_OPEN_THRESHOLD = 0.0197

    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            self.no_face_counter += 1
            if self.no_face_counter < self.AWAY_THRESHOLD:
                return self.last_status, self.last_status == "STUDYING", frame
            self.last_status = "AWAY"
            return "AWAY", False, frame

        self.no_face_counter = 0
        landmarks = results.face_landmarks[0]

        # 얼굴 박스 드로잉
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        ox, oy = max(0, int(min(xs))), max(0, int(min(ys)))
        ow = min(w - ox, int(max(xs)) - ox)
        oh = min(h - oy, int(max(ys)) - oy)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 255, 0), 2)
        cv2.putText(frame, "Face", (ox, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 눈 랜드마크 좌표 (픽셀 단위)
        # 왼눈: 위 159, 아래 145 / 오른눈: 위 386, 아래 374
        left_top    = (int(landmarks[159].x * w), int(landmarks[159].y * h))
        left_bottom = (int(landmarks[145].x * w), int(landmarks[145].y * h))
        right_top    = (int(landmarks[386].x * w), int(landmarks[386].y * h))
        right_bottom = (int(landmarks[374].x * w), int(landmarks[374].y * h))

        # 왼눈 사각형 좌표
        left_eye_pts = [
            int(landmarks[33].x * w),   # 왼쪽 끝
            int(landmarks[159].y * h),  # 위
            int(landmarks[133].x * w),  # 오른쪽 끝
            int(landmarks[145].y * h),  # 아래
        ]
        # 오른눈 사각형 좌표
        right_eye_pts = [
            int(landmarks[362].x * w),  # 왼쪽 끝
            int(landmarks[386].y * h),  # 위
            int(landmarks[263].x * w),  # 오른쪽 끝
            int(landmarks[374].y * h),  # 아래
        ]

        left_eye_dist  = abs(landmarks[159].y - landmarks[145].y)
        right_eye_dist = abs(landmarks[386].y - landmarks[374].y)

        # 디버그 출력 (값 확인 후 주석 처리 가능)


        eyes_count = 0

        # 왼눈
        if left_eye_dist > self.EYE_OPEN_THRESHOLD:
            eyes_count += 1
            cv2.rectangle(frame,
                          (left_eye_pts[0], left_eye_pts[1]),
                          (left_eye_pts[2], left_eye_pts[3]),
                          (0, 255, 0), 2)  # 초록 = 눈 뜸
        else:
            cv2.rectangle(frame,
                          (left_eye_pts[0], left_eye_pts[1]),
                          (left_eye_pts[2], left_eye_pts[3]),
                          (0, 0, 255), 2)  # 빨강 = 눈 감음

        # 오른눈
        if right_eye_dist > self.EYE_OPEN_THRESHOLD:
            eyes_count += 1
            cv2.rectangle(frame,
                          (right_eye_pts[0], right_eye_pts[1]),
                          (right_eye_pts[2], right_eye_pts[3]),
                          (0, 255, 0), 2)  # 초록 = 눈 뜸
        else:
            cv2.rectangle(frame,
                          (right_eye_pts[0], right_eye_pts[1]),
                          (right_eye_pts[2], right_eye_pts[3]),
                          (0, 0, 255), 2)  # 빨강 = 눈 감음

        cv2.putText(frame, f"Eyes: {eyes_count}", (ox, oy + oh + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # print(f"left: {left_eye_dist:.4f}, right: {right_eye_dist:.4f}",eyes_count)

        if eyes_count < 2:
            self.last_status = "SLEEPING"
            return "SLEEPING", False, frame

        self.last_status = "STUDYING"
        return "STUDYING", True, frame
    

        
# ==================== 메인 루프 ====================

phone_engine = PhoneDetector("./models/best.pt")
face_engine = FaceAnalyzer()

cap = cv2.VideoCapture("test_video.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter("test_video_output.mp4", fourcc, fps, (width, height))

print("FocusGuardian started...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    has_phone, phone_box = phone_engine.detect(frame)
    status, is_studying, frame = face_engine.analyze(frame)

    # 핸드폰 감지 시 박스 드로잉
    if has_phone and phone_box is not None:
        x1, y1, x2, y2 = phone_box.tolist()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(frame, "PHONE", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 상태 표시
    color_map = {
        "STUDYING": (0, 255, 0),
        "SLEEPING": (0, 165, 255),
        "AWAY":     (0, 0, 255)
    }
    color = color_map.get(status, (255, 255, 255))
    cv2.putText(frame, f"Status: {status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    out.write(frame)

cap.release()
out.release()

print("Program exited successfully.")
print("저장 완료: test_video_output.mp4")