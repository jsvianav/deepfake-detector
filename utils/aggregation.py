"""Helpers de agregación de scores para multi-fotograma y fusión audio+video."""

from typing import List, Optional, Tuple
import statistics


def aggregate_frame_scores(scores: List[float]) -> float:
    """Agrega los scores por fotograma en un único score de video.

    Estrategia: promedio del 25 % más sospechoso de los fotogramas, con un
    boost adicional si el fotograma más sospechoso supera 0.65. Esto hace que
    unos pocos fotogramas muy comprometedores eleven el resultado aunque el
    resto del video parezca normal (patrón típico en deepfakes de cara).

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
    top_n       = max(1, len(sorted_desc) // 4)   # top 25 %
    top_scores  = sorted_desc[:top_n]
    mean_top    = statistics.mean(top_scores)
    max_score   = sorted_desc[0]

    # Si el fotograma más sospechoso es muy alto, darle más peso
    if max_score > 0.65:
        return 0.55 * mean_top + 0.45 * max_score

    return mean_top


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
    if score > 0.55:
        return "Probablemente IA / FAKE", "⚠️"
    return "Inconcluso", "❓"
