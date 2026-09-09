"""Camera capture boundaries for the headless FaceAttend runtime."""

from faceattend.camera.worker import CameraCaptureError, CameraEvidenceWorker

__all__ = ["CameraCaptureError", "CameraEvidenceWorker"]
from faceattend.camera.latest_frame import LatestFrameBuffer

__all__ = ["LatestFrameBuffer"]
