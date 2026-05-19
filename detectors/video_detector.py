"""Detector de deepfakes en imágenes / fotogramas usando un clasificador ViT.

Mejoras principales:
- Detección de cara con OpenCV antes de la inferencia: el modelo solo ve la
  región de la cara (donde ocurre la manipulación), no el fondo.
- Análisis dual: si se detecta cara, se promedia el score del recorte facial
  con el score de la imagen completa (ponderado 75 % cara / 25 % full).
- Mapeo de labels con normalización sigmoid → probabilidad calibrada.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-Model"

FAKE_LABELS = {"fake", "deepfake", "ai", "artificial", "generated", "manipulated", "forged"}
REAL_LABELS = {"real", "authentic", "genuine", "original", "realism"}

# Pesos para combinar score del recorte facial y de la imagen completa
_FACE_WEIGHT = 0.75
_FULL_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Detección y recorte de cara con OpenCV
# ---------------------------------------------------------------------------

_face_cascade: Optional[cv2.CascadeClassifier] = None

def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def _crop_face(image: Image.Image) -> Optional[Image.Image]:
    """Detecta la cara más grande en la imagen y devuelve un recorte con margen.

    Usa el clasificador Haar de OpenCV (sin dependencias extra).
    Devuelve None si no se detecta ninguna cara.
    """
    arr_rgb = np.array(image)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)

    cascade = _get_cascade()
    faces   = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None

    # Usar la cara más grande
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Añadir margen del 30 % para incluir frente y barbilla
    pad   = int(0.30 * max(w, h))
    H, W  = arr_rgb.shape[:2]
    x1    = max(0, x - pad)
    y1    = max(0, y - pad)
    x2    = min(W, x + w + pad)
    y2    = min(H, y + h + pad)

    crop = arr_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    logger.debug("Cara detectada: (%d,%d) %dx%d → recorte (%d,%d)-(%d,%d)", x, y, w, h, x1, y1, x2, y2)
    return Image.fromarray(crop)


# ---------------------------------------------------------------------------
# Mapeo de labels del modelo
# ---------------------------------------------------------------------------

def _parse_score(results: list) -> float:
    """Extrae la probabilidad de deepfake normalizando las salidas sigmoid.

    El modelo usa sigmoid (no softmax) → Fake y Real son independientes y
    ambos pueden ser > 0.5. Se normaliza como fake/(fake+real).
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

    if best_fake is not None and best_real is not None:
        total = best_fake + best_real
        return best_fake / total if total > 0 else 0.5

    if best_fake is not None:
        return best_fake
    if best_real is not None:
        return 1.0 - best_real

    logger.warning("No se mapearon etiquetas: %s → usando 0.5", [r["label"] for r in results])
    return 0.5


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ImageDeepfakeDetector:
    """Detector de deepfakes en imágenes con recorte facial automático."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Cargando modelo de imagen %s en %s …", MODEL_ID, device)
        self._pipe = pipeline(
            "image-classification",
            model=MODEL_ID,
            device=0 if device == "cuda" else -1,
        )
        self._device = device
        logger.info("Modelo de imagen cargado.")

    # ------------------------------------------------------------------

    def _infer(self, image: Image.Image) -> float:
        """Pasa la imagen por el modelo y devuelve el score normalizado."""
        results = self._pipe(image)
        return _parse_score(results)

    def predict_pil(self, image: Image.Image) -> float:
        """Inferencia sobre una imagen PIL con detección de cara.

        Si se detecta cara: score = 75 % recorte facial + 25 % imagen completa.
        Si no hay cara: score = imagen completa (sin penalizar).
        """
        full_score = self._infer(image)

        face_crop = _crop_face(image)
        if face_crop is not None:
            face_score = self._infer(face_crop)
            combined   = _FACE_WEIGHT * face_score + _FULL_WEIGHT * full_score
            logger.debug(
                "Score imagen completa=%.4f  cara=%.4f  combinado=%.4f",
                full_score, face_score, combined,
            )
            return combined

        logger.debug("Sin cara detectada — usando score completo=%.4f", full_score)
        return full_score

    def predict_path(self, image_path: str) -> float:
        img = Image.open(image_path).convert("RGB")
        return self.predict_pil(img)

    def predict_batch(self, image_paths: List[str]) -> List[float]:
        scores: List[float] = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            scores.append(self.predict_pil(img))
        return scores
