"""Detector de deepfakes en audio usando un clasificador Wav2Vec2 / XLSR.

Mejoras sobre la versión anterior:
- Remuestreo explícito a 16 kHz antes de la inferencia (en lugar de depender
  del pipeline, que puede ser inconsistente con .m4a de iPhone a 44.1 kHz).
- Normalización RMS del audio para reducir variaciones por nivel de grabación.
- Análisis por segmentos: para grabaciones largas, se analiza en ventanas de
  10 s con solapamiento y se promedia, evitando truncaciones silenciosas.
- Mejor manejo de errores con mensajes en español.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_ID   = "MelodyMachine/Deepfake-audio-detection-V2"
TARGET_SR  = 16_000   # Hz que espera el modelo Wav2Vec2
CHUNK_SEC  = 10       # segundos por segmento
HOP_SEC    = 5        # salto entre segmentos (50 % de solapamiento)
MIN_SEC    = 1        # mínimo de audio válido para analizar un segmento

FAKE_LABELS = {"fake", "deepfake", "spoof", "ai", "artificial", "generated", "synthetic"}
REAL_LABELS = {"real", "genuine", "authentic", "bonafide", "human", "original"}


# ---------------------------------------------------------------------------
# Carga y preprocesamiento de audio
# ---------------------------------------------------------------------------

def _load_and_preprocess(audio_path: str) -> np.ndarray:
    """Carga un archivo de audio, lo remuestrea a TARGET_SR y lo normaliza.

    Intenta primero soundfile (sin ffmpeg) y luego librosa para formatos
    como mp3 / m4a que requieren ffmpeg.

    Returns:
        Array float32 normalizado en [-1, 1] a TARGET_SR Hz (mono).
    """
    import soundfile as sf

    raw: Optional[np.ndarray] = None
    sr: int = TARGET_SR

    # Intentar con soundfile (WAV, FLAC, OGG — sin ffmpeg)
    try:
        raw, sr = sf.read(audio_path, always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        raw = raw.astype(np.float32)
    except Exception:
        pass

    # Fallback a librosa (MP3, M4A — necesita ffmpeg)
    if raw is None:
        try:
            import librosa
            raw, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as exc:
            msg = str(exc).lower()
            if "ffmpeg" in msg or "audioread" in msg or "no such file" in msg:
                ext = Path(audio_path).suffix.upper()
                raise RuntimeError(
                    f"No se puede decodificar {ext}: falta ffmpeg. "
                    "Instálalo con:  brew install ffmpeg"
                ) from exc
            raise

    # Remuestrear a TARGET_SR si es necesario (crítico para .m4a a 44.1 kHz)
    if sr != TARGET_SR:
        import librosa
        logger.info("Remuestreando audio de %d Hz a %d Hz …", sr, TARGET_SR)
        raw = librosa.resample(raw, orig_sr=int(sr), target_sr=TARGET_SR)
        sr = TARGET_SR

    # Normalización RMS: reduce variaciones por nivel de grabación
    rms = float(np.sqrt(np.mean(raw ** 2)))
    if rms > 1e-8:
        target_rms = 0.1
        raw = raw * (target_rms / rms)
        raw = np.clip(raw, -1.0, 1.0)

    logger.info(
        "Audio cargado: %.2f s | SR=%d Hz | RMS=%.4f", len(raw) / TARGET_SR, TARGET_SR, rms
    )
    return raw


# ---------------------------------------------------------------------------
# Análisis por segmentos
# ---------------------------------------------------------------------------

def _chunk_scores(pipe, raw: np.ndarray) -> float:
    """Divide el audio en ventanas solapadas y promedia las predicciones.

    Evita que el pipeline trunce silenciosamente grabaciones largas
    (el modelo Wav2Vec2 tiene límite ~30 s).
    """
    chunk_size = CHUNK_SEC * TARGET_SR
    hop_size   = HOP_SEC   * TARGET_SR
    min_size   = MIN_SEC   * TARGET_SR

    # Construir segmentos
    segments = []
    start = 0
    while start < len(raw):
        end   = min(start + chunk_size, len(raw))
        chunk = raw[start:end]
        if len(chunk) >= min_size:
            segments.append(chunk)
        start += hop_size

    if not segments:
        segments = [raw]  # audio muy corto → analizar tal cual

    logger.info("Analizando %d segmento(s) de audio …", len(segments))

    scores = []
    for i, chunk in enumerate(segments):
        result = pipe({"raw": chunk, "sampling_rate": TARGET_SR})
        score  = _parse_audio_score(result)
        logger.debug("  Segmento %d/%d → %.4f", i + 1, len(segments), score)
        scores.append(score)

    final = float(np.mean(scores))
    logger.info(
        "Puntuación final: %.4f (media de %d segmento(s): %s)",
        final, len(scores), [f"{s:.3f}" for s in scores],
    )
    return final


# ---------------------------------------------------------------------------
# Mapeo de etiquetas del modelo
# ---------------------------------------------------------------------------

def _parse_audio_score(results: list) -> float:
    """Extrae la probabilidad de deepfake de la salida del pipeline."""
    if not results:
        return 0.5

    best_fake: Optional[float] = None
    best_real: Optional[float] = None

    for item in results:
        label_lower = item["label"].lower()
        score       = float(item["score"])
        if any(kw in label_lower for kw in FAKE_LABELS):
            best_fake = max(best_fake or 0.0, score)
        elif any(kw in label_lower for kw in REAL_LABELS):
            best_real = max(best_real or 0.0, score)

    if best_fake is not None:
        return best_fake
    if best_real is not None:
        return 1.0 - best_real

    logger.warning(
        "No se pudieron mapear las etiquetas a fake/real: %s — usando 0.5",
        [r["label"] for r in results],
    )
    return 0.5


# ---------------------------------------------------------------------------
# Clase pública
# ---------------------------------------------------------------------------

class AudioDeepfakeDetector:
    """Envoltorio de la pipeline de detección de deepfakes en audio."""

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

        Carga, remuestrea a 16 kHz, normaliza por RMS y analiza por
        segmentos de 10 s para mayor precisión en grabaciones largas.

        Returns:
            Probabilidad de deepfake en [0, 1].
        """
        raw    = _load_and_preprocess(audio_path)
        score  = _chunk_scores(self._pipe, raw)
        return score
