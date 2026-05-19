"""Detector de deepfakes en imágenes / fotogramas — doble modelo con recorte facial.

Arquitectura de doble modelo:
- MODEL_FULL analiza el frame completo (contexto global, artefactos de compresión).
- MODEL_FACE analiza solo el recorte de la cara (donde ocurre la manipulación).
  Este segundo modelo es diferente al primero, entrenado en otro conjunto de datos,
  lo que reduce los puntos ciegos de un modelo único.
- Score final = 82 % cara + 18 % frame completo.

Detección de cara con OpenCV:
- Haar cascade frontal + lateral (perfil) como respaldo.
- Si no se detecta cara, ambos modelos analizan el frame completo.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

# Modelo 1: análisis del frame completo (contexto holístico)
MODEL_FULL = "prithivMLmods/Deep-Fake-Detector-Model"
# Modelo 2: análisis del recorte facial (artefactos de manipulación en la cara)
MODEL_FACE = "dima806/deepfake_vs_real_image_detection"

_FACE_WEIGHT = 0.82
_FULL_WEIGHT = 0.18

FAKE_LABELS = {"fake", "deepfake", "ai", "artificial", "generated", "manipulated", "forged"}
REAL_LABELS = {"real", "authentic", "genuine", "original", "realism"}


# ---------------------------------------------------------------------------
# Detección y recorte de cara con OpenCV
# ---------------------------------------------------------------------------

_cascade_frontal: Optional[cv2.CascadeClassifier] = None
_cascade_profile: Optional[cv2.CascadeClassifier] = None


def _get_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    global _cascade_frontal, _cascade_profile
    if _cascade_frontal is None:
        _cascade_frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    if _cascade_profile is None:
        _cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
    return _cascade_frontal, _cascade_profile


def _detect_faces(gray: np.ndarray) -> np.ndarray:
    """Detecta caras usando cascadas frontal y de perfil (izq. y der.)."""
    front_cas, prof_cas = _get_cascades()
    params = dict(scaleFactor=1.05, minNeighbors=4, minSize=(50, 50))

    frontal = front_cas.detectMultiScale(gray, **params)
    profile  = prof_cas.detectMultiScale(gray, **params)
    flipped  = prof_cas.detectMultiScale(cv2.flip(gray, 1), **params)

    all_faces = []
    if len(frontal) > 0:
        all_faces.extend(frontal.tolist())
    if len(profile) > 0:
        all_faces.extend(profile.tolist())
    if len(flipped) > 0:
        W = gray.shape[1]
        for (x, y, w, h) in flipped:
            all_faces.append([W - x - w, y, w, h])  # deshacer el flip

    return np.array(all_faces) if all_faces else np.array([])


def _crop_face(image: Image.Image) -> Optional[Image.Image]:
    """Devuelve el recorte de la cara más grande (con margen del 30 %)."""
    arr_rgb = np.array(image)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)

    faces = _detect_faces(gray)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces.tolist(), key=lambda f: f[2] * f[3])
    pad = int(0.30 * max(w, h))
    H, W = arr_rgb.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)

    crop = arr_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    logger.debug("Cara: (%d,%d) %dx%d → recorte (%d,%d)-(%d,%d)", x, y, w, h, x1, y1, x2, y2)
    return Image.fromarray(crop)


# ---------------------------------------------------------------------------
# Mapeo de labels — maneja tanto softmax como sigmoid
# ---------------------------------------------------------------------------

def _parse_score(results: list) -> float:
    """Normaliza la salida del modelo a probabilidad de deepfake [0,1].

    Funciona con salidas softmax (suman ≈ 1) y sigmoid (ambas pueden ser > 0.5).
    """
    if not results:
        return 0.5

    best_fake: Optional[float] = None
    best_real: Optional[float] = None

    for item in results:
        lbl = item["label"].lower()
        s   = float(item["score"])
        if any(k in lbl for k in FAKE_LABELS):
            best_fake = max(best_fake or 0.0, s)
        elif any(k in lbl for k in REAL_LABELS):
            best_real = max(best_real or 0.0, s)

    if best_fake is not None and best_real is not None:
        total = best_fake + best_real
        return best_fake / total if total > 0 else 0.5
    if best_fake is not None:
        return best_fake
    if best_real is not None:
        return 1.0 - best_real

    logger.warning("Etiquetas no mapeadas: %s → 0.5", [r["label"] for r in results])
    return 0.5


# ---------------------------------------------------------------------------
# Detector principal
# ---------------------------------------------------------------------------

class ImageDeepfakeDetector:
    """Detector de deepfakes con doble modelo y recorte facial automático."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev_idx = 0 if device == "cuda" else -1

        logger.info("Cargando modelo de frame completo (%s) en %s …", MODEL_FULL, device)
        self._pipe_full = pipeline("image-classification", model=MODEL_FULL, device=dev_idx)

        logger.info("Cargando modelo de cara (%s) en %s …", MODEL_FACE, device)
        self._pipe_face = pipeline("image-classification", model=MODEL_FACE, device=dev_idx)

        self._device = device
        logger.info("Ambos modelos de imagen cargados.")

    # ------------------------------------------------------------------

    def _score_full(self, image: Image.Image) -> float:
        return _parse_score(self._pipe_full(image))

    def _score_face(self, image: Image.Image) -> float:
        return _parse_score(self._pipe_face(image))

    def predict_pil(self, image: Image.Image) -> float:
        """Inferencia con doble modelo + recorte facial.

        Si se detecta cara: 82 % modelo-cara + 18 % modelo-frame-completo.
        Si no: ambos modelos analizan el frame completo y se promedian.
        """
        full_score = self._score_full(image)
        face_crop  = _crop_face(image)

        if face_crop is not None:
            face_score = self._score_face(face_crop)
            combined   = _FACE_WEIGHT * face_score + _FULL_WEIGHT * full_score
            logger.debug(
                "full=%.4f  cara=%.4f  combinado=%.4f",
                full_score, face_score, combined,
            )
            return combined

        # Sin cara detectada: usar ambos modelos en el frame completo
        face_score = self._score_face(image)
        combined   = 0.5 * full_score + 0.5 * face_score
        logger.debug("Sin cara — full=%.4f dima806=%.4f → %.4f", full_score, face_score, combined)
        return combined

    def predict_path(self, image_path: str) -> float:
        img = Image.open(image_path).convert("RGB")
        return self.predict_pil(img)

    def predict_batch(self, image_paths: List[str]) -> List[float]:
        scores: List[float] = []
        for path in image_paths:
            scores.append(self.predict_path(path))
        return scores
