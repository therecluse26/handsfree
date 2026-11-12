"""Face tracking: one camera frame in, landmarks + head pose + gesture scores out.

``FaceTracker`` is the interface. ``MediaPipeFaceTracker`` uses the MediaPipe
Face Landmarker model, which runs locally.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import FaceConfig
from .geometry import head_pose_from_matrix
from .models import require_face_landmarker
from .types import FaceObservation, Frame


class FaceTracker(Protocol):
    def process(self, frame: Frame) -> FaceObservation | None:
        """Return the face in the frame, or None if there is no face."""
        ...

    def close(self) -> None: ...


class MediaPipeFaceTracker:
    def __init__(self, cfg: FaceConfig) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._cv2 = cv2
        self._mp = mp
        delegate = BaseOptions.Delegate.GPU if cfg.delegate == "gpu" else BaseOptions.Delegate.CPU
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(require_face_landmarker(cfg.model_path)), delegate=delegate
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=cfg.min_detection_confidence,
                min_face_presence_confidence=cfg.min_presence_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self._last_ts_ms = -1

    def process(self, frame: Frame) -> FaceObservation | None:
        rgb = self._cv2.cvtColor(frame.image, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(self._mp.ImageFormat.SRGB, rgb)
        # MediaPipe needs strictly increasing integer timestamps.
        ts_ms = max(int(frame.t_capture * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks:
            return None
        landmarks = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        blendshapes = (
            {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
            if result.face_blendshapes
            else {}
        )
        head = (
            head_pose_from_matrix(np.asarray(result.facial_transformation_matrixes[0]))
            if result.facial_transformation_matrixes
            else None
        )
        h, w = frame.image.shape[:2]
        return FaceObservation(
            t_capture=frame.t_capture,
            image_size=(w, h),
            landmarks=landmarks,
            blendshapes=blendshapes,
            head=head,
        )

    def close(self) -> None:
        self._landmarker.close()
