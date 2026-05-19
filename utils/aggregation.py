"""Helpers de agregación de scores para multi-fotograma y fusión audio+video."""

from typing import List, Optional, Tuple
import statistics


def aggregate_frame_scores(scores: List[float]) -> float:
    """Agrega los scores por fotograma usando señales estadísticas del conjunto.

    Usa tres señales complementarias:
    - Mediana: indica el comportamiento "típico" del video
    - Máximo + stdev: detecta el patrón de face-swap (pocos frames muy sospechosos
      con alta varianza, el resto pasan — típico de deepfakes de cara)
    - Media total: ancla conservadora cuando la mayoría de frames son reales

    Args:
        scores: Lista de probabilidades de deepfake por fotograma, en [0, 1].

    Returns:
        Score agregado en [0, 1]. Devuelve 0.0 para lista vacía.
    """
    if not scores:
        return 0.0
    if len(scores) == 1:
        return scores[0]

    sorted_desc = sorted(scores, reverse=True)
    n            = len(scores)
    mean_all     = statistics.mean(scores)
    median_all   = statistics.median(scores)
    max_score    = sorted_desc[0]
    stdev_all    = statistics.stdev(scores) if n > 1 else 0.0

    top_n     = max(1, n // 3)
    mean_top  = statistics.mean(sorted_desc[:top_n])

    # FIRMA DE VIDEO REAL: la mayoría de frames son claramente reales y el frame más
    # sospechoso no es alarmante. Usar la media de todos los frames (no solo los peores).
    if median_all < 0.38 and max_score < 0.65:
        return mean_all

    # FIRMA DE VIDEO IA (diffusion / Sora / Runway): todos los frames son sintéticos
    # → la mediana es alta de forma consistente.
    if median_all > 0.52:
        return mean_top

    # FIRMA DE FACE-SWAP: alta varianza + pico alto
    # (algunos frames claramente falsos, otros pasan; es el patrón típico de face-swap).
    if max_score > 0.68 and stdev_all > 0.08:
        return 0.55 * mean_top + 0.45 * max_score

    # POR DEFECTO: zona gris — pesar hacia los frames más sospechosos pero con cautela.
    return 0.70 * mean_top + 0.30 * mean_all


def combine_av_scores(
    video_score: Optional[float],
    audio_score: Optional[float],
    video_weight: float = 0.65,
    audio_weight: float = 0.35,
) -> float:
    """Combina los scores de video y audio en un score fusionado.

    Si una modalidad no está disponible, la otra lleva todo el peso.

    Args:
        video_score: Probabilidad de deepfake del detector visual, o None.
        audio_score: Probabilidad de deepfake del detector de audio, o None.
        video_weight: Peso del video cuando ambos están presentes.
        audio_weight: Peso del audio cuando ambos están presentes.

    Returns:
        Score fusionado en [0, 1].
    """
    if video_score is None and audio_score is None:
        return 0.0
    if video_score is None:
        return audio_score  # type: ignore[return-value]
    if audio_score is None:
        return video_score

    total = video_weight + audio_weight
    return (video_score * video_weight + audio_score * audio_weight) / total


def score_to_verdict(score: float) -> Tuple[str, str]:
    """Convierte el score fusionado en un veredicto legible.

    Returns:
        Tupla (etiqueta, emoji).
    """
    if score < 0.40:
        return "Probablemente REAL", "✅"
    if score > 0.57:
        return "Probablemente IA / FAKE", "⚠️"
    return "Inconcluso", "❓"
