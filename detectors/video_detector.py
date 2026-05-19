"""Image / video-frame deepfake detector using a ViT-based classifier."""

import logging
from typing import List, Optional

import numpy as np
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

# HuggingFace model identifier — loaded once at startup.
MODEL_ID = "prithivMLmods/Deep-Fake-Detector-Model"

# Labels the model may emit that indicate a FAKE result.
# We normalise to lowercase for matching.
FAKE_LABELS = {"fake", "deepfake", "ai", "artificial", "generated", "manipulated", "forged"}
REAL_LABELS = {"real", "authentic", "genuine", "original"}


def _parse_score(results: list) -> float:
    """Extrae la probabilidad de deepfake de la salida del pipeline.

    Este modelo usa salidas sigmoid (no softmax), por lo que los scores de
    'Fake' y 'Real' son independientes y no suman 1.0. Se normalizan
    explícitamente para obtener una probabilidad calibrada.

    Ejemplo real del modelo:
        Fake=0.859, Real=0.716  →  0.859 / (0.859 + 0.716) = 0.546 (inconcluso)
        Fake=0.12,  Real=0.95   →  0.12  / (0.12  + 0.95)  = 0.112 (auténtico)
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

    # Normalizar cuando ambos scores están presentes (caso sigmoid multi-label)
    if best_fake is not None and best_real is not None:
        total = best_fake + best_real
        if total > 0:
            return best_fake / total
        return 0.5

    if best_fake is not None:
        # Solo hay score de fake — usarlo directamente
        return best_fake
    if best_real is not None:
        return 1.0 - best_real

    logger.warning(
        "No se pudieron mapear etiquetas a fake/real: %s — usando 0.5",
        [r["label"] for r in results],
    )
    return 0.5


class ImageDeepfakeDetector:
    """Wrapper around the ViT-based image deepfake detection pipeline.

    The model is loaded once on construction and reused for all calls.

    Example:
        detector = ImageDeepfakeDetector()
        score = detector.predict_pil(pil_image)
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Load the model into memory.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading image deepfake model %s on %s …", MODEL_ID, device)
        self._pipe = pipeline(
            "image-classification",
            model=MODEL_ID,
            device=0 if device == "cuda" else -1,
        )
        self._device = device
        logger.info("Image model loaded.")

    def predict_pil(self, image: Image.Image) -> float:
        """Run inference on a PIL image.

        Args:
            image: RGB PIL image.

        Returns:
            Deepfake probability in [0, 1].
        """
        results = self._pipe(image)
        score = _parse_score(results)
        logger.debug("Frame score: %.4f (raw=%s)", score, results)
        return score

    def predict_path(self, image_path: str) -> float:
        """Run inference on an image file.

        Args:
            image_path: Path to image file.

        Returns:
            Deepfake probability in [0, 1].
        """
        img = Image.open(image_path).convert("RGB")
        return self.predict_pil(img)

    def predict_batch(self, image_paths: List[str]) -> List[float]:
        """Run inference on a list of image paths.

        Args:
            image_paths: List of paths to image files.

        Returns:
            List of deepfake probabilities, one per input image.
        """
        images = [Image.open(p).convert("RGB") for p in image_paths]
        scores: List[float] = []
        for img in images:
            scores.append(self.predict_pil(img))
        return scores
