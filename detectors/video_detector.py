"""Detector de deepfakes en imágenes / fotogramas — dos modelos complementarios.

Arquitectura dual especializada:
- MODEL_FACESWAP (prithivMLmods v1, ViT): detecta manipulación facial (face-swap).
  Se aplica al recorte de cara si se detecta una, si no al frame completo.
- MODEL_AICONTENT (haywoodsloan SwinV2, 0.2 B params): detecta contenido generado
  por IA (videos Sora/Runway/diffusion, imágenes sintéticas). Se aplica siempre
  al frame completo — fue diseñado para imágenes completas.
  Precisión: 98.15 % acc / 98.17 % precision / 99.35 % recall.

Score final = 45 % face-swap + 55 % AI-content.

Detección de cara con OpenCV:
- Haar cascade frontal + lateral (perfil) como respaldo.
- Si no se detecta cara, MODEL_FACESWAP también analiza el frame completo.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

# Especialista en face-swap y manipulación facial (ViT, ~85 M params)
MODEL_FACESWAP  = "prithivMLmods/Deep-Fake-Detector-Model"
# Especialista en contenido generado por IA (SwinV2, ~0.2 B params)
MODEL_AICONTENT = "haywoodsloan/ai-image-detector-deploy"

_FACESWAP_WEIGHT  = 0.45
_AICONTENT_WEIGHT = 0.55

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
    """Detector de deepfakes: especialista face-swap + detector de contenido IA."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev_idx = 0 if device == "cuda" else -1

        logger.info("Cargando modelo face-swap (%s) en %s …", MODEL_FACESWAP, device)
        self._pipe_faceswap = pipeline(
            "image-classification", model=MODEL_FACESWAP, device=dev_idx
        )

        logger.info("Cargando modelo AI-content (%s) en %s …", MODEL_AICONTENT, device)
        self._pipe_aicontent = pipeline(
            "image-classification", model=MODEL_AICONTENT, device=dev_idx
        )

        self._device = device
        logger.info("Ambos modelos de imagen cargados.")

    # ------------------------------------------------------------------

    def predict_pil(self, image: Image.Image) -> float:
        """Inferencia combinada: face-swap specialist + AI-content detector.

        - MODEL_AICONTENT siempre analiza el frame completo (fue diseñado así).
        - MODEL_FACESWAP analiza el recorte de cara si hay cara, si no el frame completo.
        Score = 45 % face-swap + 55 % AI-content.
        """
        # Detector de contenido IA — siempre sobre el frame completo
        ai_score = _parse_score(self._pipe_aicontent(image))

        # Detector de face-swap — preferiblemente sobre el recorte de cara
        face_crop = _crop_face(image)
        if face_crop is not None:
            fs_score = _parse_score(self._pipe_faceswap(face_crop))
            logger.debug("cara: fs=%.4f  ai=%.4f", fs_score, ai_score)
        else:
            fs_score = _parse_score(self._pipe_faceswap(image))
            logger.debug("sin cara: fs=%.4f  ai=%.4f", fs_score, ai_score)

        return _FACESWAP_WEIGHT * fs_score + _AICONTENT_WEIGHT * ai_score

    def temporal_consistency_score(self, image_paths: List[str]) -> float:
        """Detecta incoherencia de textura facial entre frames — señal clave de face-swap.

        Los deepfakes sintetizan cada frame de forma independiente, produciendo
        variaciones sutiles en color y textura de la cara que no ocurren en videos
        reales (donde la textura facial es temporalmente estable).

        Returns:
            Score [0, 1]. Mayor = más incoherencia temporal = más sospechoso.
        """
        _SZ = 64
        features: List[np.ndarray] = []

        for path in image_paths:
            try:
                img  = Image.open(path).convert("RGB")
                face = _crop_face(img)
                if face is None:
                    continue
                arr = np.array(face.resize((_SZ, _SZ))).astype(np.float32)
                # Normalizar luminosidad → aísla crominancia/textura
                lum = arr.mean()
                if lum > 5.0:
                    arr = np.clip(arr / lum * 128.0, 0.0, 255.0)
                # Vector 6-D: media RGB + std RGB de la región facial
                # (captura tanto tono de piel como rugosidad de textura)
                vec = np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))])
                features.append(vec)
            except Exception:
                continue

        if len(features) < 3:
            logger.debug("Temporal: pocos frames con cara (%d) — señal neutra", len(features))
            return 0.5

        feat_arr  = np.stack(features)         # (n, 6)
        time_var  = float(feat_arr.std(axis=0).mean())  # varianza temporal media
        # Calibración empírica: real ≈ 0.5-2.0, face-swap ≈ 2.5-8.0 (escala 128)
        score = float(min(1.0, time_var / 6.0))
        logger.info("Temporal: varianza=%.4f → score=%.4f (%d frames con cara)",
                    time_var, score, len(features))
        return score

    def predict_path(self, image_path: str) -> float:
        img = Image.open(image_path).convert("RGB")
        return self.predict_pil(img)

    def predict_batch(self, image_paths: List[str]) -> List[float]:
        scores: List[float] = []
        for path in image_paths:
            scores.append(self.predict_path(path))
        return scores
