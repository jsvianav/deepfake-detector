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
    """Extract deepfake probability from pipeline output.

    The pipeline returns a list of dicts: [{'label': '...', 'score': 0.9}, ...].
    We identify which label corresponds to 'fake' and return its score.
    If both labels are ambiguous we return 0.5.

    Args:
        results: Raw output from transformers image-classification pipeline.

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
        # Check fake keywords
        if any(kw in label_lower for kw in FAKE_LABELS):
            best_fake = max(best_fake or 0.0, score)
        # Check real keywords
        elif any(kw in label_lower for kw in REAL_LABELS):
            best_real = max(best_real or 0.0, score)

    if best_fake is not None:
        return best_fake
    if best_real is not None:
        # Invert: high real score → low fake score
        return 1.0 - best_real

    # Fallback: use the top-1 label.  Assume the first result is 'fake' if
    # its score is the highest, otherwise treat as unknown.
    logger.warning(
        "Could not map labels to fake/real: %s — defaulting to 0.5",
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
