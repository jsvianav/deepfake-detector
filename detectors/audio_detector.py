"""Audio deepfake detector using a Wav2Vec2 / XLSR-based classifier."""

import logging
from typing import Optional

import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

FAKE_LABELS = {"fake", "deepfake", "spoof", "ai", "artificial", "generated", "synthetic"}
REAL_LABELS = {"real", "genuine", "authentic", "bonafide", "human", "original"}


def _parse_audio_score(results: list) -> float:
    """Extract deepfake probability from audio-classification pipeline output.

    Args:
        results: Raw pipeline output list of dicts with 'label' and 'score'.

    Returns:
        Deepfake probability in [0, 1].
    """
    if not results:
        return 0.5

    best_fake: Optional[float] = None
    best_real: Optional[float] = None

    for item in results:
        label_lower = item["label"].lower()
        score = float(item["score"])
        if any(kw in label_lower for kw in FAKE_LABELS):
            best_fake = max(best_fake or 0.0, score)
        elif any(kw in label_lower for kw in REAL_LABELS):
            best_real = max(best_real or 0.0, score)

    if best_fake is not None:
        return best_fake
    if best_real is not None:
        return 1.0 - best_real

    logger.warning(
        "Could not map audio labels to fake/real: %s — defaulting to 0.5",
        [r["label"] for r in results],
    )
    return 0.5


class AudioDeepfakeDetector:
    """Wrapper around the audio deepfake detection pipeline.

    The model is loaded once on construction and reused for all calls.

    Example:
        detector = AudioDeepfakeDetector()
        score = detector.predict("path/to/audio.wav")
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Load the audio classification model into memory.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading audio deepfake model %s on %s …", MODEL_ID, device)
        self._pipe = pipeline(
            "audio-classification",
            model=MODEL_ID,
            device=0 if device == "cuda" else -1,
        )
        self._device = device
        logger.info("Audio model loaded.")

    def predict(self, audio_path: str) -> float:
        """Run inference on an audio file.

        The pipeline internally handles resampling to the model's expected
        sample rate, so any sample rate is acceptable on input.

        Args:
            audio_path: Path to a WAV, MP3, FLAC, or OGG file.

        Returns:
            Deepfake probability in [0, 1].
        """
        results = self._pipe(audio_path)
        score = _parse_audio_score(results)
        logger.debug("Audio score: %.4f (raw=%s)", score, results)
        return score
