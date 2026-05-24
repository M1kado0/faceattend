"""Shared liveness helpers for short webcam challenge videos."""

from __future__ import annotations

from dataclasses import dataclass

from backend.api.ml_client import MLServiceRejectedError, verify_passive_liveness
from backend.api.video_frames import VideoFrameDecodeError, sample_video_frames_as_jpegs


@dataclass(frozen=True)
class VideoLivenessSummary:
    sampled_frames: int
    visible_faces: int
    passive_passes: int
    passive_liveness_pass_ratio: float
    face_visible_ratio: float
    embedding_frame: bytes | None
    embedding_frames: tuple[bytes, ...] = ()


async def analyze_video_passive_liveness(
    *,
    ml_service_url: str,
    video: bytes,
    max_frames: int,
) -> VideoLivenessSummary:
    """Sample frames, run passive liveness, and choose a live frame for embedding."""
    frame_jpegs = sample_video_frames_as_jpegs(video, max_frames=max_frames)
    visible_faces = 0
    passive_passes = 0
    live_frame_candidates: list[tuple[float, bytes]] = []

    for index, frame_jpeg in enumerate(frame_jpegs):
        try:
            passive = await verify_passive_liveness(
                ml_service_url=ml_service_url,
                blob=frame_jpeg,
                filename=f"liveness-frame-{index}.jpg",
                content_type="image/jpeg",
            )
        except MLServiceRejectedError as exc:
            if exc.status_code == 422:
                continue
            raise

        visible_faces += 1
        if passive.passed:
            passive_passes += 1
            live_frame_candidates.append((passive.score, frame_jpeg))

    total_frames = len(frame_jpegs)
    if total_frames == 0:
        raise VideoFrameDecodeError("no_decodable_frames")

    embedding_frames = tuple(
        frame for _, frame in sorted(live_frame_candidates, key=lambda item: item[0], reverse=True)
    )

    return VideoLivenessSummary(
        sampled_frames=total_frames,
        visible_faces=visible_faces,
        passive_passes=passive_passes,
        passive_liveness_pass_ratio=passive_passes / total_frames,
        face_visible_ratio=visible_faces / total_frames,
        embedding_frame=embedding_frames[0] if embedding_frames else None,
        embedding_frames=tuple(embedding_frames),
    )
