"""Helpers for sampling frames from short webcam videos."""

from __future__ import annotations

import tempfile

import cv2


class VideoFrameDecodeError(ValueError):
    """Raised when a check-in video cannot be sampled."""


def sample_video_frames_as_jpegs(video: bytes, *, max_frames: int = 10) -> list[bytes]:
    if not video:
        raise VideoFrameDecodeError("empty_video")

    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
        tmp.write(video)
        tmp.flush()

        capture = cv2.VideoCapture(tmp.name)
        if not capture.isOpened():
            raise VideoFrameDecodeError("could_not_open_video")

        frames = []
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(frame_bgr)

        capture.release()

    if not frames:
        raise VideoFrameDecodeError("no_decodable_frames")

    sampled_frames = _evenly_sample(frames, max_items=max_frames)
    encoded_frames = []
    for frame_bgr in sampled_frames:
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise VideoFrameDecodeError("could_not_encode_frame")
        encoded_frames.append(encoded.tobytes())
    return encoded_frames


def _evenly_sample(items: list, *, max_items: int) -> list:
    if max_items <= 0:
        raise ValueError("max_items_must_be_positive")
    if len(items) <= max_items:
        return items

    last_index = len(items) - 1
    return [items[round(index * last_index / (max_items - 1))] for index in range(max_items)]
