"""Score aggregation helpers for multi-frame and audio+video fusion."""

from typing import List, Optional, Tuple
import statistics


def aggregate_frame_scores(scores: List[float]) -> float:
    """Aggregate per-frame deepfake scores into a single video score.

    Uses the mean of the top-50 % of scores so that a few very suspicious
    frames raise the alarm without isolated noise dominating.

    Args:
        scores: List of per-frame deepfake probabilities in [0, 1].

    Returns:
        Aggregated score in [0, 1].  Returns 0.0 for an empty list.
    """
    if not scores:
        return 0.0
    if len(scores) == 1:
        return scores[0]
    sorted_desc = sorted(scores, reverse=True)
    top_half = sorted_desc[: max(1, len(sorted_desc) // 2)]
    return statistics.mean(top_half)


def combine_av_scores(
    video_score: Optional[float],
    audio_score: Optional[float],
    video_weight: float = 0.6,
    audio_weight: float = 0.4,
) -> float:
    """Combine video and audio deepfake scores into a single fused score.

    If one modality is unavailable the other carries full weight.

    Args:
        video_score: Deepfake probability from the video detector, or None.
        audio_score: Deepfake probability from the audio detector, or None.
        video_weight: Weight for video when both are present.
        audio_weight: Weight for audio when both are present.

    Returns:
        Fused score in [0, 1].
    """
    if video_score is None and audio_score is None:
        return 0.0
    if video_score is None:
        return audio_score  # type: ignore[return-value]
    if audio_score is None:
        return video_score

    total_weight = video_weight + audio_weight
    return (video_score * video_weight + audio_score * audio_weight) / total_weight


def score_to_verdict(score: float) -> Tuple[str, str]:
    """Convert a fused deepfake score to a human-readable verdict.

    Args:
        score: Deepfake probability in [0, 1].

    Returns:
        Tuple of (verdict_label, emoji_indicator).
        verdict_label is one of:
            'Probablemente REAL'
            'Probablemente IA / FAKE'
            'Inconcluso'
    """
    if score < 0.35:
        return "Probablemente REAL", "✅"
    if score > 0.65:
        return "Probablemente IA / FAKE", "⚠️"
    return "Inconcluso", "❓"
