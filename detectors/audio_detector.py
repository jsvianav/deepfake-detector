"""Detector de deepfakes en audio usando un clasificador Wav2Vec2 / XLSR."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

FAKE_LABELS = {"fake", "deepfake", "spoof", "ai", "artificial", "generated", "synthetic"}
REAL_LABELS = {"real", "genuine", "authentic", "bonafide", "human", "original"}

TARGET_SR = 16_000  # Hz que espera el modelo


def _load_audio(audio_path: str) -> dict:
    """Carga audio como dict {raw, sampling_rate} compatible con la pipeline.

    Intenta primero soundfile (sin ffmpeg) y luego librosa (requiere ffmpeg
    para formatos como mp3 / m4a).
    """
    import soundfile as sf

    try:
        data, sr = sf.read(audio_path, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return {"raw": data.astype(np.float32), "sampling_rate": int(sr)}
    except Exception:
        pass

    # soundfile no admite MP3/M4A — intentar con librosa (necesita ffmpeg)
    try:
        import librosa
        data, sr = librosa.load(audio_path, sr=None, mono=True)
        return {"raw": data, "sampling_rate": int(sr)}
    except Exception as exc:
        msg = str(exc).lower()
        if "ffmpeg" in msg or "audioread" in msg or "no such file" in msg:
            ext = Path(audio_path).suffix.upper()
            raise RuntimeError(
                f"No se puede decodificar {ext}: falta ffmpeg. "
                "Instálalo con:  brew install ffmpeg"
            ) from exc
        raise


def _parse_audio_score(results: list) -> float:
    """Extrae la probabilidad de deepfake de la salida de la pipeline."""
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
        "No se pudieron mapear las etiquetas de audio a fake/real: %s — usando 0.5",
        [r["label"] for r in results],
    )
    return 0.5


class AudioDeepfakeDetector:
    """Envoltorio de la pipeline de detección de audio deepfake."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Cargando modelo de audio %s en %s …", MODEL_ID, device)
        self._pipe = pipeline(
            "audio-classification",
            model=MODEL_ID,
            device=0 if device == "cuda" else -1,
        )
        self._device = device
        logger.info("Modelo de audio cargado.")

    def predict(self, audio_path: str) -> float:
        """Ejecuta inferencia sobre un archivo de audio.

        Returns:
            Probabilidad de deepfake en [0, 1].
        """
        audio_input = _load_audio(audio_path)
        results = self._pipe(audio_input)
        score = _parse_audio_score(results)
        logger.debug("Puntuación de audio: %.4f (raw=%s)", score, results)
        return score
