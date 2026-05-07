"""Deepfake Detector — Gradio chat-style interface.

Loads both models once at startup and serves inference through a Blocks UI
that accepts drag-and-drop media uploads and returns a structured analysis.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import torch

# Ensure project root is on the path when running from any directory.
sys.path.insert(0, str(Path(__file__).parent))

from detectors.video_detector import ImageDeepfakeDetector
from detectors.audio_detector import AudioDeepfakeDetector
from detectors.orchestrator import Orchestrator, DetectionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time model loading
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logger.warning(
        "No GPU detected — running on CPU. Inference will be slower. "
        "Consider using a machine with CUDA for faster results."
    )

logger.info("=== Loading models (this may take a minute on first run) ===")
_image_detector = ImageDeepfakeDetector(device=device)
_audio_detector = AudioDeepfakeDetector(device=device)
_orchestrator = Orchestrator(_image_detector, _audio_detector, frame_fps=1.0, max_frames=30)
logger.info("=== Models ready ===")

# ---------------------------------------------------------------------------
# Supported extensions (shown in the UI)
# ---------------------------------------------------------------------------
ACCEPTED_EXTENSIONS = [
    ".mp4", ".mov", ".avi", ".webm",
    ".mp3", ".wav", ".m4a", ".ogg",
    ".jpg", ".jpeg", ".png",
]

# ---------------------------------------------------------------------------
# Chat history type alias
# ---------------------------------------------------------------------------
ChatHistory = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Core inference handler
# ---------------------------------------------------------------------------

def analyse_file(
    file_obj,
    history: ChatHistory,
) -> Tuple[ChatHistory, ChatHistory, str]:
    """Run deepfake detection on the uploaded file and append the result to chat.

    Args:
        file_obj: Gradio file object (has .name attribute with temp path).
        history: Current chat history as list of (user_msg, bot_msg) tuples.

    Returns:
        Tuple of (updated history, updated history, empty string to clear input).
    """
    if file_obj is None:
        msg = "⚠️ Por favor, sube un archivo antes de analizar."
        history = history + [("", msg)]
        return history, history, None

    # NamedString (Gradio 4.x) is a str subclass; handle both str and file objects
    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = getattr(file_obj, "name", str(file_obj))

    if not file_path or not Path(file_path).is_file():
        msg = "⚠️ Archivo no válido. Por favor sube un archivo de imagen, audio o video."
        history = history + [("", msg)]
        return history, history, None

    file_name = Path(file_path).name

    # User bubble
    user_msg = f"📎 Analizando: **{file_name}**"
    history = history + [(user_msg, None)]  # type: ignore[list-item]

    try:
        result: DetectionResult = _orchestrator.analyse(file_path)
        bot_msg = _format_result(result, file_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error analysing %s", file_path)
        bot_msg = f"❌ Error inesperado: {exc}"

    # Replace the pending None with the actual bot response
    history[-1] = (user_msg, bot_msg)
    return history, history, None


def _format_result(result: DetectionResult, file_name: str) -> str:
    """Render a DetectionResult as a Markdown-formatted chat message.

    Args:
        result: The detection output from the orchestrator.
        file_name: Original file name (for display only).

    Returns:
        Markdown string suitable for display in a Gradio chatbot.
    """
    if result.error:
        return (
            f"❌ **Error al procesar `{file_name}`**\n\n"
            f"> {result.error}\n\n"
            "_Verifica que el archivo no esté corrupto y sea menor de 100 MB._"
        )

    lines: List[str] = []
    lines.append(f"## {result.verdict_emoji} Resultado para `{file_name}`")
    lines.append("")

    # Verdict
    lines.append(f"**Veredicto:** {result.verdict}")
    fused_pct = result.fused_score * 100
    lines.append(f"**Confianza (probabilidad de ser FAKE):** {fused_pct:.1f}%")
    lines.append("")

    # Score bar (ASCII)
    bar = _score_bar(result.fused_score)
    lines.append(f"`{bar}`")
    lines.append("")

    # Detail breakdown
    if result.file_type == "video":
        lines.append("### Desglose por modalidad")
        if result.video_score is not None:
            lines.append(
                f"- 🎞️ **Video** ({result.frame_count} fotogramas): "
                f"{result.video_score * 100:.1f}% fake"
            )
        else:
            lines.append("- 🎞️ **Video:** no se pudieron extraer fotogramas")

        if result.had_audio and result.audio_score is not None:
            lines.append(
                f"- 🔊 **Audio:** {result.audio_score * 100:.1f}% fake"
            )
        else:
            lines.append("- 🔊 **Audio:** no disponible en este video")
        lines.append("")

    elif result.file_type == "audio":
        lines.append(
            f"- 🔊 **Score de audio:** {result.fused_score * 100:.1f}% fake"
        )
        lines.append("")

    elif result.file_type == "image":
        lines.append(
            f"- 🖼️ **Score de imagen:** {result.fused_score * 100:.1f}% fake"
        )
        lines.append("")

    # Disclaimer
    lines.append("---")
    lines.append(
        "> ⚠️ **Nota:** Esta detección no es infalible. Los modelos pueden fallar "
        "con técnicas de síntesis nuevas, video muy comprimido o audio con mucho ruido."
    )

    return "\n".join(lines)


def _score_bar(score: float, width: int = 30) -> str:
    """Generate an ASCII progress bar for a score in [0, 1].

    Args:
        score: Value between 0 and 1.
        width: Total bar width in characters.

    Returns:
        String like 'FAKE |████████░░░░░░░░░░░░░░░░░░| REAL  42%'.
    """
    filled = int(round(score * width))
    empty = width - filled
    bar = "█" * filled + "░" * empty
    pct = score * 100
    return f"FAKE |{bar}| REAL  {pct:.0f}%"


def clear_history():
    """Reset the chat history."""
    return [], [], None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

WELCOME_MESSAGE = (
    "👋 **Bienvenido al Detector de Deepfakes**\n\n"
    "Sube un archivo de **video**, **audio** o **imagen** y te diré si parece "
    "real o generado/manipulado por IA.\n\n"
    f"**Formatos aceptados:** {', '.join(ACCEPTED_EXTENSIONS)}\n"
    "**Tamaño máximo:** 100 MB\n\n"
    f"_Ejecutando en: **{device.upper()}**_"
)

with gr.Blocks(
    title="Deepfake Detector",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="""
    #chatbot { height: 520px; }
    .upload-area { border: 2px dashed #6366f1 !important; border-radius: 12px; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown("# 🔍 Deepfake Detector")
    gr.Markdown(
        "Analiza archivos de video, audio e imagen con modelos locales de IA "
        "para detectar contenido sintético o manipulado."
    )

    # State: chat history persisted across interactions
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[(None, WELCOME_MESSAGE)],
                elem_id="chatbot",
                label="Análisis",
                bubble_full_width=False,
                show_label=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### Subir archivo")
            file_input = gr.File(
                label="Arrastra o selecciona un archivo",
                file_types=ACCEPTED_EXTENSIONS,
                elem_classes=["upload-area"],
            )
            analyse_btn = gr.Button("🔍 Analizar", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Limpiar chat", variant="secondary")

            gr.Markdown(
                "---\n"
                "**Modelos usados:**\n"
                "- Imagen/Video: `prithivMLmods/Deep-Fake-Detector-Model`\n"
                "- Audio: `MelodyMachine/Deepfake-audio-detection-V2`\n\n"
                "_100% local · sin APIs externas_"
            )

    # Wire up events
    analyse_btn.click(
        fn=analyse_file,
        inputs=[file_input, state],
        outputs=[chatbot, state, file_input],
    )

    # Allow pressing Enter on the file component to trigger analysis
    file_input.upload(
        fn=analyse_file,
        inputs=[file_input, state],
        outputs=[chatbot, state, file_input],
    )

    clear_btn.click(
        fn=clear_history,
        outputs=[chatbot, state, file_input],
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
