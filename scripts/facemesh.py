"""Preview MediaPipe Face Landmarker output from the webcam."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

MODEL_PATH = Path("ml/models/mediapipe/face_landmarker_v2_with_blendshapes.task")
HEAD_POSE_LANDMARKS = {1, 33, 61, 199, 263, 291}


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could_not_open_webcam")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )

    with FaceLandmarker.create_from_options(options) as face_landmarker:
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        while cap.isOpened():
            success, image_bgr = cap.read()
            if not success:
                break

            start = time.time()
            image_bgr = cv2.flip(image_bgr, 1)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(round((frame_index / fps) * 1000))
            frame_index += 1

            img_h, img_w, _ = image_bgr.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    face_2d = []
                    face_3d = []
                    nose_2d = None
                    nose_3d = None

                    for idx, lm in enumerate(face_landmarks):
                        if idx in HEAD_POSE_LANDMARKS:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    if len(face_2d) < 6 or nose_2d is None or nose_3d is None:
                        continue

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w
                    cam_matrix = np.array(
                        [
                            [focal_length, 0, img_w / 2],
                            [0, focal_length, img_h / 2],
                            [0, 0, 1],
                        ],
                    )
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(
                        face_3d,
                        face_2d,
                        cam_matrix,
                        dist_matrix,
                    )
                    if not success:
                        continue

                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, *_ = cv2.RQDecomp3x3(rmat)

                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    if y < -10:
                        text = "Looking Left"
                    elif y > 10:
                        text = "Looking Right"
                    elif x < -10:
                        text = "Looking Down"
                    elif x > 10:
                        text = "Looking Up"
                    else:
                        text = "Forward"

                    nose_3d_projection, _ = cv2.projectPoints(
                        np.array([nose_3d], dtype=np.float64),
                        rot_vec,
                        trans_vec,
                        cam_matrix,
                        dist_matrix,
                    )

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    cv2.line(image_bgr, p1, p2, (255, 0, 0), 3)
                    cv2.putText(
                        image_bgr,
                        text,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        image_bgr,
                        f"x: {np.round(x, 2)}",
                        (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        image_bgr,
                        f"y: {np.round(y, 2)}",
                        (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        image_bgr,
                        f"z: {np.round(z, 2)}",
                        (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    _draw_face_landmarks(image_bgr, face_landmarks)

            end = time.time()
            total_time = end - start
            preview_fps = 1 / total_time if total_time > 0 else 0
            cv2.putText(
                image_bgr,
                f"FPS: {int(preview_fps)}",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )
            _draw_blink_scores(image_bgr, results.face_blendshapes)

            cv2.imshow("Head Pose Estimation", image_bgr)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def _draw_face_landmarks(image_bgr, face_landmarks) -> None:
    height, width = image_bgr.shape[:2]
    for landmark in face_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image_bgr, (x, y), 2, (0, 255, 255), -1)


def _draw_blink_scores(image_bgr, face_blendshapes) -> None:
    if not face_blendshapes:
        return

    scores = {
        category.category_name: category.score
        for category in face_blendshapes[0]
        if category.category_name in {"eyeBlinkLeft", "eyeBlinkRight"}
    }
    left = scores.get("eyeBlinkLeft", 0.0)
    right = scores.get("eyeBlinkRight", 0.0)
    cv2.putText(
        image_bgr,
        f"blink L:{left:.2f} R:{right:.2f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


if __name__ == "__main__":
    main()
