"""Orchestrator that routes media to the right detector(s) and fuses results."""

import logging
import tempfile
from dataclasses import dataclass
from typing import Optional

from utils.media_processing import (
    get_file_type,
    extract_frames,
    extract_audio,
    cleanup_temp_files,
    validate_file,
)
from utils.aggregation import aggregate_frame_scores, combine_av_scores, score_to_verdict
from detectors.video_detector import ImageDeepfakeDetector
from detectors.audio_detector import AudioDeepfakeDetector

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for all detection outputs.

    Attributes:
        file_type: 'video', 'audio', or 'image'.
        video_score: Per-frame aggregated score (None if not applicable).
        audio_score: Audio classification score (None if not applicable).
        fused_score: Combined score used for the verdict.
        verdict: Human-readable verdict string.
        verdict_emoji: Emoji indicator for the verdict.
        frame_count: Number of frames analysed (0 for non-video input).
        had_audio: Whether the video contained an audio track.
        error: Error message if something went wrong, else None.
    """

    file_type: str = "unknown"
    video_score: Optional[float] = None
    audio_score: Optional[float] = None
    fused_score: float = 0.5
    verdict: str = "Inconcluso"
    verdict_emoji: str = "❓"
    frame_count: int = 0
    had_audio: bool = False
    error: Optional[str] = None


class Orchestrator:
    """High-level controller that coordinates all detectors.

    Models are injected so they can be loaded once at startup and
    shared across requests.

    Example:
        img_det = ImageDeepfakeDetector()
        aud_det = AudioDeepfakeDetector()
        orch = Orchestrator(img_det, aud_det)
        result = orch.analyse("my_video.mp4")
    """

    def __init__(
        self,
        image_detector: ImageDeepfakeDetector,
        audio_detector: AudioDeepfakeDetector,
        frame_fps: float = 1.0,
        max_frames: int = 30,
    ) -> None:
        """Initialise the orchestrator with pre-loaded detectors.

        Args:
            image_detector: Loaded image/video-frame detector.
            audio_detector: Loaded audio detector.
            frame_fps: Frames per second to sample from videos.
            max_frames: Maximum frames to analyse per video.
        """
        self._img = image_detector
        self._aud = audio_detector
        self._fps = frame_fps
        self._max_frames = max_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, file_path: str) -> DetectionResult:
        """Analyse a media file and return a DetectionResult.

        Automatically detects whether the file is a video, audio, or image
        and routes it through the appropriate pipeline.

        Args:
            file_path: Absolute or relative path to the media file.

        Returns:
            DetectionResult with all scores and the final verdict.
        """
        result = DetectionResult()
        try:
            validate_file(file_path)
            file_type = get_file_type(file_path)
            result.file_type = file_type

            if file_type == "video":
                self._analyse_video(file_path, result)
            elif file_type == "audio":
                self._analyse_audio(file_path, result)
            elif file_type == "image":
                self._analyse_image(file_path, result)
            else:
                result.error = "Formato de archivo no soportado."
                result.verdict = "Inconcluso"
                result.verdict_emoji = "❓"
                return result

            verdict, emoji = score_to_verdict(result.fused_score)
            result.verdict = verdict
            result.verdict_emoji = emoji

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error analysing %s", file_path)
            result.error = str(exc)
            result.verdict = "Error"
            result.verdict_emoji = "❌"

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse_video(self, file_path: str, result: DetectionResult) -> None:
        """Run video + audio analysis and populate *result* in place.

        Args:
            file_path: Path to the video file.
            result: DetectionResult to populate.
        """
        tmp_dir = tempfile.mkdtemp(prefix="deepfake_video_")
        try:
            # 1. Extract and score frames.
            frames, _ = extract_frames(
                file_path,
                fps=self._fps,
                max_frames=self._max_frames,
                tmp_dir=tmp_dir,
            )
            result.frame_count = len(frames)

            if frames:
                frame_scores = self._img.predict_batch(frames)
                result.video_score = aggregate_frame_scores(frame_scores)
                logger.info(
                    "Video score: %.4f from %d frames", result.video_score, len(frames)
                )
            else:
                logger.warning("No frames extracted from %s", file_path)

            # 2. Extract and score audio.
            audio_path = extract_audio(file_path, tmp_dir=tmp_dir)
            if audio_path:
                result.had_audio = True
                result.audio_score = self._aud.predict(audio_path)
                logger.info("Audio score: %.4f", result.audio_score)

            # 3. Fusionar — el audio pesa menos en videos porque el modelo de audio
            # está entrenado para voz sintética, no para música de fondo.
            # Un video con música (ej. deepfake de baile) no debe ser penalizado
            # por el audio, ya que la música no es voz humana ni voz sintética.
            result.fused_score = combine_av_scores(
                result.video_score,
                result.audio_score,
                video_weight=0.85,
                audio_weight=0.15,
            )

        finally:
            cleanup_temp_files(tmp_dir)

    def _analyse_audio(self, file_path: str, result: DetectionResult) -> None:
        """Run audio-only analysis and populate *result* in place.

        Args:
            file_path: Path to the audio file.
            result: DetectionResult to populate.
        """
        result.audio_score = self._aud.predict(file_path)
        result.fused_score = result.audio_score
        result.had_audio = True
        logger.info("Audio-only score: %.4f", result.audio_score)

    def _analyse_image(self, file_path: str, result: DetectionResult) -> None:
        """Run image-only analysis and populate *result* in place.

        Args:
            file_path: Path to the image file.
            result: DetectionResult to populate.
        """
        result.video_score = self._img.predict_path(file_path)
        result.fused_score = result.video_score
        result.frame_count = 1
        logger.info("Image score: %.4f", result.video_score)
