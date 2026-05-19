"""Utilities for extracting frames and audio from media files using ffmpeg."""

import os
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import ffmpeg
import cv2

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def get_file_type(file_path: str) -> str:
    """Determine media type from file extension.

    Args:
        file_path: Path to the media file.

    Returns:
        One of 'video', 'audio', 'image', or 'unknown'.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def validate_file(file_path: str) -> None:
    """Validate that a file exists, is readable, and is within size limits.

    Args:
        file_path: Path to validate.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exceeds the size limit.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.is_dir():
        raise ValueError(f"La ruta es un directorio, no un archivo: {file_path}")
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large ({size / 1024 / 1024:.1f} MB). Maximum is 100 MB."
        )


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 30,
    tmp_dir: Optional[str] = None,
) -> Tuple[List[str], str]:
    """Extract frames from a video file at the specified rate.

    Args:
        video_path: Path to the input video.
        fps: Frames per second to extract (default 1.0).
        max_frames: Maximum number of frames to extract (default 30).
        tmp_dir: Directory for temporary files; uses system temp if None.

    Returns:
        Tuple of (list of frame file paths, temp directory path used).

    Raises:
        RuntimeError: If ffmpeg fails to extract frames.
    """
    validate_file(video_path)
    work_dir = tmp_dir or tempfile.mkdtemp(prefix="deepfake_frames_")
    frame_pattern = os.path.join(work_dir, "frame_%04d.jpg")

    logger.info("Extracting frames from %s at %.1f fps (max %d)", video_path, fps, max_frames)
    try:
        (
            ffmpeg
            .input(video_path)
            .filter("fps", fps=fps)
            .output(frame_pattern, vframes=max_frames, q=2)
            .overwrite_output()
            .run(quiet=True)
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg no está instalado. Instálalo con:  brew install ffmpeg"
        ) from exc
    except ffmpeg.Error as exc:
        raise RuntimeError(f"ffmpeg falló al extraer fotogramas: {exc.stderr.decode(errors='replace')}") from exc

    frames = sorted(
        str(p) for p in Path(work_dir).glob("frame_*.jpg")
    )
    logger.info("Extracted %d frames", len(frames))
    return frames, work_dir


def extract_audio(
    video_path: str,
    tmp_dir: Optional[str] = None,
) -> Optional[str]:
    """Extract audio track from a video file as a WAV file.

    Args:
        video_path: Path to the input video.
        tmp_dir: Directory for temporary files; uses system temp if None.

    Returns:
        Path to the extracted WAV file, or None if the video has no audio.

    Raises:
        RuntimeError: If ffmpeg fails unexpectedly.
    """
    validate_file(video_path)
    work_dir = tmp_dir or tempfile.mkdtemp(prefix="deepfake_audio_")
    audio_out = os.path.join(work_dir, "audio.wav")

    logger.info("Extracting audio from %s", video_path)
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_out, acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        if "no audio" in stderr.lower() or "does not contain" in stderr.lower():
            logger.warning("No audio track found in %s", video_path)
            return None
        # Some videos silently produce an empty file; treat that as no audio.
        if not Path(audio_out).exists() or Path(audio_out).stat().st_size < 512:
            logger.warning("Audio extraction produced no usable output for %s", video_path)
            return None
        if "no such file" in stderr.lower() or not stderr:
            raise RuntimeError(
                "ffmpeg no está instalado. Instálalo con:  brew install ffmpeg"
            ) from exc
        raise RuntimeError(f"ffmpeg falló al extraer audio: {stderr}") from exc

    if not Path(audio_out).exists() or Path(audio_out).stat().st_size < 512:
        logger.warning("Audio file is empty for %s", video_path)
        return None

    logger.info("Audio extracted to %s", audio_out)
    return audio_out


def load_image(image_path: str):
    """Load an image file as an RGB numpy array via OpenCV.

    Args:
        image_path: Path to the image.

    Returns:
        RGB numpy array (H, W, 3).

    Raises:
        ValueError: If the image cannot be read.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def cleanup_temp_files(*paths: str) -> None:
    """Remove temporary files and/or directories.

    Args:
        *paths: Paths to files or directories to delete.
    """
    import shutil
    for path in paths:
        try:
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not remove temp path %s: %s", path, exc)
