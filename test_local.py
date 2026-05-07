"""Local smoke-test script for the deepfake detector pipeline.

Creates minimal synthetic test fixtures (a solid-color video, a sine-wave
audio, and a JPEG image) so the pipeline can be exercised without real media.

Usage:
    python test_local.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure project root is on the path.
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("test_local")


# ---------------------------------------------------------------------------
# Fixture generators
# ---------------------------------------------------------------------------

def _make_test_image(tmp_dir: str) -> str:
    """Create a 224×224 RGB JPEG filled with a neutral grey tone."""
    import cv2

    path = os.path.join(tmp_dir, "test_image.jpg")
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    cv2.imwrite(path, img)
    logger.info("Created test image: %s", path)
    return path


def _make_test_audio(tmp_dir: str) -> str:
    """Create a 3-second 440 Hz sine wave as a 16-bit mono WAV."""
    import soundfile as sf

    path = os.path.join(tmp_dir, "test_audio.wav")
    sr = 16_000
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, signal, sr)
    logger.info("Created test audio: %s", path)
    return path


def _make_test_video(tmp_dir: str) -> str:
    """Create a 3-second solid-colour MP4 with a 440 Hz audio track."""
    import subprocess

    path = os.path.join(tmp_dir, "test_video.mp4")
    cmd = [
        "ffmpeg", "-y",
        # Video: 3 s of blue frames at 10 fps
        "-f", "lavfi", "-i", "color=c=blue:s=224x224:r=10:d=3",
        # Audio: 3 s of 440 Hz sine wave
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(
            "ffmpeg video generation failed: %s\nSkipping video test.",
            result.stderr.decode(errors="replace"),
        )
        return ""
    logger.info("Created test video: %s", path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_utils(tmp_dir: str) -> None:
    """Test media processing utilities."""
    from utils.media_processing import get_file_type, validate_file
    from utils.aggregation import aggregate_frame_scores, combine_av_scores, score_to_verdict

    logger.info("--- Testing utils ---")

    assert get_file_type("foo.mp4") == "video"
    assert get_file_type("foo.wav") == "audio"
    assert get_file_type("foo.png") == "image"
    assert get_file_type("foo.xyz") == "unknown"

    assert aggregate_frame_scores([]) == 0.0
    assert aggregate_frame_scores([1.0]) == 1.0
    assert 0 <= aggregate_frame_scores([0.3, 0.7, 0.5]) <= 1.0

    assert combine_av_scores(None, None) == 0.0
    assert combine_av_scores(0.8, None) == 0.8
    assert combine_av_scores(None, 0.4) == 0.4
    fused = combine_av_scores(0.8, 0.4)
    assert 0.4 <= fused <= 0.8

    verdict, emoji = score_to_verdict(0.1)
    assert "REAL" in verdict
    verdict, emoji = score_to_verdict(0.9)
    assert "FAKE" in verdict
    verdict, emoji = score_to_verdict(0.5)
    assert "Inconcluso" in verdict

    logger.info("✅ utils tests passed")


def test_image_detector(image_path: str) -> None:
    """Test the image deepfake detector on a synthetic image."""
    from detectors.video_detector import ImageDeepfakeDetector

    logger.info("--- Testing ImageDeepfakeDetector ---")
    det = ImageDeepfakeDetector()
    score = det.predict_path(image_path)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    logger.info("✅ Image detector score: %.4f", score)


def test_audio_detector(audio_path: str) -> None:
    """Test the audio deepfake detector on a synthetic audio file."""
    from detectors.audio_detector import AudioDeepfakeDetector

    logger.info("--- Testing AudioDeepfakeDetector ---")
    det = AudioDeepfakeDetector()
    score = det.predict(audio_path)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    logger.info("✅ Audio detector score: %.4f", score)


def test_orchestrator_image(image_path: str) -> None:
    """Test the orchestrator on an image file."""
    from detectors.video_detector import ImageDeepfakeDetector
    from detectors.audio_detector import AudioDeepfakeDetector
    from detectors.orchestrator import Orchestrator

    logger.info("--- Testing Orchestrator (image) ---")
    orch = Orchestrator(ImageDeepfakeDetector(), AudioDeepfakeDetector())
    result = orch.analyse(image_path)
    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.file_type == "image"
    assert result.verdict in {"Probablemente REAL", "Probablemente IA / FAKE", "Inconcluso"}
    logger.info("✅ Orchestrator image result: %s (%.4f)", result.verdict, result.fused_score)


def test_orchestrator_audio(audio_path: str) -> None:
    """Test the orchestrator on an audio file."""
    from detectors.video_detector import ImageDeepfakeDetector
    from detectors.audio_detector import AudioDeepfakeDetector
    from detectors.orchestrator import Orchestrator

    logger.info("--- Testing Orchestrator (audio) ---")
    orch = Orchestrator(ImageDeepfakeDetector(), AudioDeepfakeDetector())
    result = orch.analyse(audio_path)
    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.file_type == "audio"
    logger.info("✅ Orchestrator audio result: %s (%.4f)", result.verdict, result.fused_score)


def test_orchestrator_video(video_path: str) -> None:
    """Test the orchestrator on a video file."""
    from detectors.video_detector import ImageDeepfakeDetector
    from detectors.audio_detector import AudioDeepfakeDetector
    from detectors.orchestrator import Orchestrator

    logger.info("--- Testing Orchestrator (video) ---")
    orch = Orchestrator(ImageDeepfakeDetector(), AudioDeepfakeDetector())
    result = orch.analyse(video_path)
    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.file_type == "video"
    logger.info(
        "✅ Orchestrator video result: %s (video=%.4f, audio=%s, fused=%.4f)",
        result.verdict,
        result.video_score or 0.0,
        f"{result.audio_score:.4f}" if result.audio_score is not None else "N/A",
        result.fused_score,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all tests sequentially."""
    with tempfile.TemporaryDirectory(prefix="deepfake_test_") as tmp_dir:
        logger.info("Using temp dir: %s", tmp_dir)

        # Create fixtures
        image_path = _make_test_image(tmp_dir)
        audio_path = _make_test_audio(tmp_dir)
        video_path = _make_test_video(tmp_dir)

        # Run utility tests (no model required)
        test_utils(tmp_dir)

        # Run model-based tests
        test_image_detector(image_path)
        test_audio_detector(audio_path)

        # Orchestrator tests
        test_orchestrator_image(image_path)
        test_orchestrator_audio(audio_path)

        if video_path:
            test_orchestrator_video(video_path)
        else:
            logger.warning("Skipped video orchestrator test (ffmpeg not available).")

    logger.info("=== All tests completed ===")


if __name__ == "__main__":
    main()
