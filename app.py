"""Detector de Deepfakes — Interfaz Gradio con tema oscuro premium.

Carga ambos modelos una vez al iniciar y sirve inferencia a través de una
interfaz Blocks que acepta archivos multimedia por arrastrar y soltar.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import torch

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
# Carga de modelos (una sola vez al iniciar)
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logger.warning("Sin GPU detectada — ejecutando en CPU. La inferencia será más lenta.")

logger.info("=== Cargando modelos (puede tomar un minuto la primera vez) ===")
_image_detector = ImageDeepfakeDetector(device=device)
_audio_detector = AudioDeepfakeDetector(device=device)
_orchestrator = Orchestrator(_image_detector, _audio_detector, frame_fps=1.0, max_frames=30)
logger.info("=== Modelos listos ===")

# ---------------------------------------------------------------------------
# Extensiones aceptadas
# ---------------------------------------------------------------------------
ACCEPTED_EXTENSIONS = [
    ".mp4", ".mov", ".avi", ".webm",
    ".mp3", ".wav", ".m4a", ".ogg",
    ".jpg", ".jpeg", ".png",
]

ChatHistory = List[Tuple[str, str]]

# ---------------------------------------------------------------------------
# Iconos SVG (atributos HTML, no camelCase de React)
# ---------------------------------------------------------------------------
def _svg(path_content: str, w: int = 15) -> str:
    s = f'style="width:{w}px;height:{w}px;display:inline-block;vertical-align:middle"'
    attrs = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
    return f'<svg {attrs} {s}>{path_content}</svg>'

_ICON_VIDEO  = _svg('<polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>')
_ICON_AUDIO  = _svg('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>')
_ICON_IMAGE  = _svg('<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>')
_ICON_SEARCH = _svg('<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/>')
_ICON_ALERT  = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" '
    'style="width:13px;height:13px;flex-shrink:0;margin-top:1px">'
    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
    '</svg>'
)

_TYPE_ICON = {"video": _ICON_VIDEO, "audio": _ICON_AUDIO, "image": _ICON_IMAGE}

# ---------------------------------------------------------------------------
# CSS — tema oscuro premium
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Keyframes ───────────────────────────────────────────── */
@keyframes fadeSlideUp  { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn       { from { opacity:0; } to { opacity:1; } }
@keyframes scaleIn      { from { opacity:0; transform:scale(0.95); } to { opacity:1; transform:scale(1); } }
@keyframes barFill      { from { width:0; } }
@keyframes needleSlide  { from { left:50%; opacity:0; } }
@keyframes pulseDot     { 0%,100% { opacity:.6; transform:scale(1); } 50% { opacity:1; transform:scale(1.3); } }
@keyframes pulseAccent  { 0%,100% { box-shadow:0 0 20px rgba(124,58,237,.25); } 50% { box-shadow:0 0 32px rgba(124,58,237,.55); } }

/* ── Global reset / fuente ───────────────────────────────── */
html, body, .gradio-container, #app, gradio-app {
  background: #0A0A0F !important;
  color: #F8F7FF !important;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
  letter-spacing: -0.011em !important;
}

/* ── Contenedor con gradiente atmosférico ────────────────── */
.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 0 28px 60px !important;
  min-height: 100vh !important;
  position: relative !important;
  animation: fadeSlideUp 0.4s ease both !important;
  background:
    radial-gradient(ellipse 700px 500px at -5% -10%, rgba(124,58,237,.14), transparent 60%),
    radial-gradient(ellipse 600px 400px at 110% 110%, rgba(91,33,182,.10), transparent 60%),
    #0A0A0F !important;
}

/* ── Scrollbars ──────────────────────────────────────────── */
*::-webkit-scrollbar             { width:4px; height:4px; }
*::-webkit-scrollbar-track       { background:transparent; }
*::-webkit-scrollbar-thumb       { background:rgba(124,58,237,.35); border-radius:2px; }
*::-webkit-scrollbar-thumb:hover { background:rgba(124,58,237,.6); }

/* ── Encabezado de página ────────────────────────────────── */
.gradio-container > .markdown:first-child {
  padding: 28px 0 20px 0 !important;
  border-bottom: 1px solid rgba(255,255,255,.05) !important;
  margin-bottom: 24px !important;
}

/* ── Panel del chatbot ───────────────────────────────────── */
#chatbot {
  height: 620px !important;
  background: rgba(12,11,20,.75) !important;
  border: 1px solid rgba(124,58,237,.15) !important;
  border-radius: 18px !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.04),
    0 24px 60px -20px rgba(0,0,0,.65),
    0 0 0 1px rgba(124,58,237,.04) !important;
  padding: 4px !important;
}
#chatbot .wrapper,
#chatbot .bubble-wrap { background:transparent !important; border:none !important; }

/* Burbuja del bot — transparente para que se vean las tarjetas HTML */
#chatbot .message.bot,
#chatbot .bot-row .message,
#chatbot div[data-testid="bot"],
#chatbot .message-row.bot .message-bubble {
  background: transparent !important;
  border: none !important;
  padding: 4px 0 !important;
  animation: fadeSlideUp 0.28s ease both !important;
}

/* Burbuja del usuario */
#chatbot .message.user,
#chatbot .user-row .message,
#chatbot div[data-testid="user"],
#chatbot .message-row.user .message-bubble {
  background: rgba(124,58,237,.08) !important;
  color: #C4B5FD !important;
  border: 1px solid rgba(124,58,237,.2) !important;
  border-radius: 14px 14px 4px 14px !important;
  padding: 10px 14px !important;
  font-size: 13.5px !important;
  animation: fadeSlideUp 0.25s ease both !important;
}

/* Tipografía dentro de burbujas */
#chatbot strong  { color:#F8F7FF !important; }
#chatbot code    {
  background: rgba(124,58,237,.12) !important;
  color: #A78BFA !important;
  border-radius: 4px !important;
  padding: 1px 5px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
}
#chatbot blockquote {
  border-left: 3px solid rgba(124,58,237,.4) !important;
  padding-left: 12px !important;
  margin-left: 0 !important;
  color: #9CA3AF !important;
  font-style: italic !important;
}
#chatbot h2 { font-size:16px !important; font-weight:600 !important; color:#F8F7FF !important; margin:0 0 8px !important; }
#chatbot h3 { font-size:11px !important; font-weight:600 !important; color:#6B6883 !important; text-transform:uppercase !important; letter-spacing:.07em !important; margin:12px 0 6px !important; }
#chatbot ul { padding-left:16px !important; margin:4px 0 !important; }
#chatbot li { margin:3px 0 !important; color:#B8B5C8 !important; }
#chatbot p  { color:#B8B5C8 !important; line-height:1.6 !important; }

/* ── Panel derecho (tarjeta de subida) ───────────────────── */
.right-panel {
  background: rgba(12,11,20,.75) !important;
  border: 1px solid rgba(124,58,237,.15) !important;
  border-radius: 18px !important;
  padding: 22px !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 14px !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.04),
    0 24px 60px -20px rgba(0,0,0,.55) !important;
  position: sticky !important;
  top: 20px !important;
}
.right-panel > * { background:transparent !important; }
.right-panel .markdown h3 {
  font-size: 10px !important;
  font-weight: 600 !important;
  color: #6B6883 !important;
  text-transform: uppercase !important;
  letter-spacing: .1em !important;
  margin: 0 0 10px !important;
}

/* ── Zona de arrastrar y soltar ──────────────────────────── */
.upload-area {
  border: 1.5px dashed rgba(124,58,237,.4) !important;
  background: rgba(124,58,237,.025) !important;
  border-radius: 12px !important;
  transition: all .2s ease !important;
  cursor: pointer !important;
}
.upload-area:hover {
  border-color: rgba(124,58,237,.75) !important;
  background: rgba(124,58,237,.07) !important;
  box-shadow: 0 0 24px rgba(124,58,237,.15) !important;
}
.upload-area svg, .upload-area .icon { color:#A78BFA !important; }
.upload-area label, .upload-area span, .upload-area button { color:#6B6883 !important; font-size:13px !important; }
.upload-area button { background:transparent !important; border:none !important; }
.upload-area .file-preview, .upload-area [data-testid="file-upload"] { background:transparent !important; border:none !important; }

/* ── Botón principal Analizar ────────────────────────────── */
.right-panel button.primary,
.right-panel .primary > button,
button[variant="primary"] {
  background: #7C3AED !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 13px 20px !important;
  width: 100% !important;
  letter-spacing: -.005em !important;
  box-shadow: 0 8px 22px -8px rgba(124,58,237,.6), inset 0 1px 0 rgba(255,255,255,.12) !important;
  transition: all .18s ease !important;
  animation: pulseAccent 3s ease-in-out infinite !important;
  position: relative !important;
  overflow: hidden !important;
}
.right-panel button.primary:hover,
.right-panel .primary > button:hover {
  background: #8B45F2 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 12px 28px -8px rgba(124,58,237,.7) !important;
}
.right-panel button.primary:active,
.right-panel .primary > button:active { transform:translateY(0) !important; }

/* ── Botón secundario Limpiar ────────────────────────────── */
.right-panel button.secondary,
.right-panel .secondary > button {
  background: transparent !important;
  border: 1px solid rgba(124,58,237,.25) !important;
  color: #6B6883 !important;
  border-radius: 12px !important;
  font-weight: 500 !important;
  font-size: 13.5px !important;
  padding: 11px 20px !important;
  width: 100% !important;
  transition: all .18s ease !important;
}
.right-panel button.secondary:hover,
.right-panel .secondary > button:hover {
  border-color: rgba(124,58,237,.6) !important;
  color: #C4B5FD !important;
  background: rgba(124,58,237,.05) !important;
}

/* ── Tarjeta de info de modelos ──────────────────────────── */
.right-panel .markdown:last-of-type {
  background: rgba(124,58,237,.04) !important;
  border: 1px solid rgba(124,58,237,.1) !important;
  border-radius: 10px !important;
  padding: 12px 14px !important;
}
.right-panel .markdown:last-of-type p,
.right-panel .markdown:last-of-type li {
  font-size: 11.5px !important;
  color: #6B6883 !important;
  line-height: 1.6 !important;
  margin: 4px 0 !important;
}
.right-panel .markdown:last-of-type strong {
  color: #9CA3AF !important;
  font-weight: 600 !important;
  font-size: 10.5px !important;
  letter-spacing: .04em !important;
  text-transform: uppercase !important;
}
.right-panel .markdown:last-of-type code {
  color: #A78BFA !important;
  background: rgba(124,58,237,.1) !important;
  border-radius: 3px !important;
  padding: 1px 5px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
}

/* ── Ocultar pie de página de Gradio ─────────────────────── */
footer { display:none !important; }

/* ── Responsivo ──────────────────────────────────────────── */
@media (max-width: 900px) {
  .right-panel { position:static !important; }
  #chatbot { height:500px !important; }
  .gradio-container { padding:0 16px 40px !important; }
}
"""

# ---------------------------------------------------------------------------
# Helpers HTML para tarjetas de resultados
# ---------------------------------------------------------------------------

def _chip(
    text: str,
    color: str = "#A78BFA",
    bg: str = "rgba(124,58,237,.1)",
    border: str = "rgba(124,58,237,.2)",
) -> str:
    return (
        f'<span style="padding:3px 9px;border-radius:6px;background:{bg};border:1px solid {border};'
        f'font-size:10px;font-weight:600;color:{color};letter-spacing:.06em;text-transform:uppercase;'
        f"font-family:'JetBrains Mono',monospace\">{text}</span>"
    )


def _mini_bar(score_val: Optional[float], label: str, icon: str) -> str:
    label_row = (
        f'<div style="display:flex;align-items:center;gap:6px;font-size:10px;color:#6B6883;'
        f'text-transform:uppercase;letter-spacing:.07em;font-weight:500;margin-bottom:7px">'
        f'<span style="color:#A78BFA">{icon}</span>{label}</div>'
    )
    if score_val is None:
        return (
            label_row
            + '<div style="font-size:20px;font-weight:600;color:#6B6883;margin-bottom:6px;letter-spacing:-.025em">—</div>'
            + '<div style="height:3px;background:#1B1A26;border-radius:1.5px;margin-bottom:6px"></div>'
            + "<div style=\"font-size:10px;color:#6B6883;font-family:'JetBrains Mono',monospace\">sin datos</div>"
        )
    c = "#EF4444" if score_val > 0.5 else "#10B981"
    p = score_val * 100
    return (
        label_row
        + f'<div style="font-size:20px;font-weight:600;letter-spacing:-.025em;margin-bottom:6px;color:{c}">'
        + f'{p:.1f}<span style="font-size:12px;color:#6B6883;margin-left:1px">%</span></div>'
        + f'<div style="height:3px;background:#1B1A26;border-radius:1.5px;overflow:hidden;margin-bottom:6px">'
        + f'<div style="height:100%;width:{p:.1f}%;background:{c};border-radius:inherit;'
        + f'animation:barFill .8s cubic-bezier(.2,.7,.2,1) .25s both"></div></div>'
    )


# ---------------------------------------------------------------------------
# Renderizador de tarjeta de resultado
# ---------------------------------------------------------------------------

def _format_result(result: DetectionResult, file_name: str) -> str:
    """Genera una tarjeta HTML animada con el resultado de detección."""

    if result.error:
        return (
            '<div style="border-radius:14px;overflow:hidden;background:rgba(18,17,26,.9);'
            'border:1px solid rgba(245,158,11,.3);font-family:Inter,system-ui,sans-serif;'
            'box-shadow:0 20px 40px -16px rgba(0,0,0,.6);animation:scaleIn .45s cubic-bezier(.2,.7,.2,1) both">'
            '<div style="display:flex;align-items:center;gap:10px;padding:13px 18px;border-bottom:1px solid rgba(255,255,255,.06)">'
            f'<div style="width:32px;height:32px;border-radius:9px;background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.2);display:flex;align-items:center;justify-content:center;color:#A78BFA;flex-shrink:0">{_ICON_IMAGE}</div>'
            f"<div style=\"flex:1;min-width:0;font-family:'JetBrains Mono',monospace;font-size:12px;color:#F8F7FF;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500\">{file_name}</div>"
            + _chip("ERROR", "#FCD34D", "rgba(245,158,11,.1)", "rgba(245,158,11,.3)") +
            '</div>'
            f'<div style="padding:20px;color:#FCA5A5;font-size:13px;line-height:1.55">{result.error}</div>'
            '</div>'
        )

    score = result.fused_score
    pct   = score * 100

    if score >= 0.6:
        kind, verdict_label     = "fake", "Contenido sintético"
        fill_c, fill_sh         = "#EF4444", "rgba(239,68,68,.5)"
        pill_bg, pill_c, pill_b = "rgba(239,68,68,.1)", "#FCA5A5", "rgba(239,68,68,.3)"
        card_b, card_sh         = "rgba(239,68,68,.25)", "rgba(239,68,68,.18)"
    elif score <= 0.4:
        kind, verdict_label     = "real", "Contenido auténtico"
        fill_c, fill_sh         = "#10B981", "rgba(16,185,129,.5)"
        pill_bg, pill_c, pill_b = "rgba(16,185,129,.1)", "#6EE7B7", "rgba(16,185,129,.3)"
        card_b, card_sh         = "rgba(16,185,129,.25)", "rgba(16,185,129,.15)"
    else:
        kind, verdict_label     = "amb", "Resultado inconcluso"
        fill_c, fill_sh         = "#F59E0B", "rgba(245,158,11,.4)"
        pill_bg, pill_c, pill_b = "rgba(245,158,11,.1)", "#FCD34D", "rgba(245,158,11,.3)"
        card_b, card_sh         = "rgba(245,158,11,.25)", "rgba(245,158,11,.12)"

    thumb = _TYPE_ICON.get(result.file_type, _ICON_IMAGE)
    type_label = result.file_type.upper()

    # ── Desglose (solo video) ──────────────────────────────────────────
    breakdown = ""
    if result.file_type == "video":
        audio_score = result.audio_score if result.had_audio else None
        audio_meta  = "wav2vec2 · 16 kHz" if result.had_audio else "sin pista de audio"
        breakdown = (
            '<div style="border-top:1px solid rgba(255,255,255,.06);padding:14px 22px 16px;'
            'display:grid;grid-template-columns:1fr 1fr;gap:0">'
            '<div style="padding-right:14px">'
            + _mini_bar(result.video_score, "Visual", _ICON_VIDEO)
            + f"<div style=\"font-size:10px;color:#6B6883;font-family:'JetBrains Mono',monospace\">{result.frame_count} fotogramas · 1 fps</div>"
            + '</div>'
            '<div style="padding-left:14px;border-left:1px solid rgba(255,255,255,.06)">'
            + _mini_bar(audio_score, "Audio", _ICON_AUDIO)
            + f"<div style=\"font-size:10px;color:#6B6883;font-family:'JetBrains Mono',monospace\">{audio_meta}</div>"
            + '</div>'
            '</div>'
        )

    # ── Filas de detalles forenses ─────────────────────────────────────
    def _tr(k: str, v: str) -> str:
        mono = "font-family:'JetBrains Mono',monospace;font-size:11px"
        return (
            f'<span style="color:#6B6883;{mono}">{k}</span>'
            f'<span style="color:#F8F7FF;font-weight:500;{mono}">{v}</span>'
        )

    tech = _tr("puntaje_fusionado", f"{result.fused_score:.4f}")
    if result.video_score is not None:
        tech += _tr("puntaje_video", f"{result.video_score:.4f}")
    if result.audio_score is not None:
        tech += _tr("puntaje_audio", f"{result.audio_score:.4f}")
    tech += _tr("fotogramas",   str(result.frame_count))
    tech += _tr("tiene_audio",  "sí" if result.had_audio else "no")
    tech += _tr("tipo_archivo", result.file_type)
    tech += _tr("dispositivo",  device)

    # ── Ensamblado ─────────────────────────────────────────────────────
    return (
        # Envoltorio de tarjeta
        f'<div style="border-radius:14px;overflow:hidden;background:rgba(18,17,26,.9);'
        f'border:1px solid {card_b};'
        f'box-shadow:0 20px 40px -16px rgba(0,0,0,.65),0 0 40px -8px {card_sh};'
        f'font-family:Inter,system-ui,sans-serif;'
        f'animation:scaleIn .45s cubic-bezier(.2,.7,.2,1) both">'

        # Barra superior: icono + nombre de archivo + chip de tipo
        f'<div style="display:flex;align-items:center;gap:10px;padding:13px 18px;border-bottom:1px solid rgba(255,255,255,.06)">'
        f'<div style="width:32px;height:32px;border-radius:9px;background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.2);display:flex;align-items:center;justify-content:center;color:#A78BFA;flex-shrink:0">{thumb}</div>'
        f"<div style=\"flex:1;min-width:0;font-family:'JetBrains Mono',monospace;font-size:12px;color:#F8F7FF;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500\">{file_name}</div>"
        + _chip(type_label) +
        '</div>'

        # Veredicto: pastilla + número de puntuación grande
        f'<div style="padding:28px 22px 20px;text-align:center">'
        f'<div style="display:inline-flex;align-items:center;gap:7px;padding:5px 12px 5px 10px;border-radius:999px;'
        f'font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:18px;'
        f'background:{pill_bg};color:{pill_c};border:1px solid {pill_b};animation:fadeIn .4s ease .2s both">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{fill_c};box-shadow:0 0 6px {fill_c};display:inline-block"></span>'
        f'{verdict_label}</div>'
        f'<div style="font-size:60px;font-weight:600;letter-spacing:-.04em;line-height:1;color:#F8F7FF;font-variant-numeric:tabular-nums">'
        f'{pct:.1f}<span style="font-size:24px;color:#6B6883;font-weight:500;margin-left:3px">%</span></div>'
        f'<div style="font-size:12px;color:#6B6883;margin-top:8px">probabilidad de contenido sintético</div>'
        f'</div>'

        # Barra de escala animada con aguja
        f'<div style="padding:0 22px 22px">'
        f'<div style="position:relative;height:6px;background:#1B1A26;border-radius:3px;overflow:visible">'
        f'<div style="position:absolute;left:0;top:0;bottom:0;width:{pct:.1f}%;border-radius:3px;'
        f'background:{fill_c};box-shadow:0 0 12px {fill_sh};animation:barFill .9s cubic-bezier(.2,.7,.2,1) both"></div>'
        f'<div style="position:absolute;top:-3px;bottom:-3px;left:{pct:.1f}%;width:2px;background:#fff;'
        f'border-radius:1px;transform:translateX(-50%);'
        f'box-shadow:0 0 0 2px rgba(10,10,15,.6),0 0 8px #fff;'
        f'animation:needleSlide .6s cubic-bezier(.2,.7,.2,1) .35s both"></div>'
        f'</div>'
        f"<div style=\"display:flex;justify-content:space-between;margin-top:12px;font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.04em\">"
        f'<span style="color:#6EE7B7">0% — Auténtico</span>'
        f'<span style="color:#6B6883">50</span>'
        f'<span style="color:#FCA5A5">100% — Sintético</span>'
        f'</div></div>'

        + breakdown +

        # Detalles forenses desplegables
        '<details style="border-top:1px solid rgba(255,255,255,.06)">'
        '<summary style="list-style:none;padding:13px 22px;cursor:pointer;font-size:12.5px;color:#B8B5C8;'
        'display:flex;align-items:center;justify-content:space-between;user-select:none">'
        '<span>Detalles forenses</span>'
        "<span style=\"font-size:10px;color:#6B6883;font-family:'JetBrains Mono',monospace\">▼</span></summary>"
        '<div style="padding:0 22px 16px;display:grid;grid-template-columns:max-content 1fr;gap:7px 20px">'
        + tech +
        '</div></details>'

        # Aviso legal
        '<div style="border-top:1px solid rgba(255,255,255,.06);padding:12px 22px;background:rgba(0,0,0,.2);'
        'font-size:11.5px;color:#B8B5C8;line-height:1.5;display:flex;gap:8px;align-items:flex-start">'
        + _ICON_ALERT +
        '<span>La detección es probabilística, no definitiva. El resultado es una señal de referencia — '
        'revisa los detalles forenses y el contexto antes de sacar conclusiones.</span>'
        '</div>'

        '</div>'
    )


# ---------------------------------------------------------------------------
# Manejador principal de inferencia
# ---------------------------------------------------------------------------

def analyse_file(file_obj, history: ChatHistory) -> Tuple[ChatHistory, ChatHistory, None]:
    if file_obj is None:
        history = history + [("", "⚠️ Por favor sube un archivo antes de analizar.")]
        return history, history, None

    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))

    if not file_path or not Path(file_path).is_file():
        history = history + [("", "⚠️ Archivo inválido. Por favor sube un video, audio o imagen.")]
        return history, history, None

    file_name = Path(file_path).name
    user_msg  = f"📎 **{file_name}**"
    history   = history + [(user_msg, None)]  # type: ignore[list-item]

    try:
        result: DetectionResult = _orchestrator.analyse(file_path)
        bot_msg = _format_result(result, file_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado al analizar %s", file_path)
        bot_msg = f"❌ Error inesperado: {exc}"

    history[-1] = (user_msg, bot_msg)
    return history, history, None


def clear_history():
    return [], [], None


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------

_status_dot = (
    '<span style="width:5px;height:5px;border-radius:50%;background:#10B981;'
    'box-shadow:0 0 6px #10B981;display:inline-block;animation:pulseDot 2s ease-in-out infinite"></span>'
)

HEADER_HTML = (
    '<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:4px 0">'
    '<div style="width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#7C3AED,#5B21B6);'
    'display:flex;align-items:center;justify-content:center;box-shadow:0 0 24px rgba(124,58,237,.5);flex-shrink:0">'
    '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:16px;height:16px">'
    '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>'
    '</div>'
    '<div>'
    '<div style="font-size:20px;font-weight:700;color:#F8F7FF;letter-spacing:-.02em;line-height:1.1">'
    'Detector de '
    '<span style="background:linear-gradient(135deg,#C4B5FD,#A78BFA);-webkit-background-clip:text;background-clip:text;color:transparent">'
    'Deepfakes</span></div>'
    '<div style="font-size:12px;color:#6B6883;margin-top:3px">Imagen ViT · Audio Wav2Vec2 · Inferencia 100% local</div>'
    '</div>'
    '<div style="margin-left:auto;display:inline-flex;align-items:center;gap:7px;font-size:11px;color:#6EE7B7;'
    'background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:999px;padding:4px 11px;'
    "font-family:'JetBrains Mono',monospace\">"
    + _status_dot
    + f' Listo · {device.upper()}</div>'
    '</div>'
)

WELCOME_HTML = (
    '<div style="border-radius:14px;overflow:hidden;background:rgba(124,58,237,.06);'
    'border:1px solid rgba(124,58,237,.15);padding:24px 22px;font-family:Inter,system-ui,sans-serif;'
    'animation:fadeIn .5s ease both">'
    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
    '<div style="width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,#7C3AED,#5B21B6);'
    'display:flex;align-items:center;justify-content:center;box-shadow:0 0 14px rgba(124,58,237,.4)">'
    + _ICON_SEARCH +
    '</div>'
    '<div style="font-size:14px;font-weight:600;color:#F8F7FF;letter-spacing:-.01em">Listo para analizar</div>'
    '</div>'
    '<p style="font-size:13px;color:#9CA3AF;line-height:1.65;margin:0 0 16px 0">'
    'Sube un archivo de <strong style="color:#C4B5FD">video</strong>, <strong style="color:#C4B5FD">audio</strong> o '
    '<strong style="color:#C4B5FD">imagen</strong> en el panel de la derecha y haz clic en '
    '<strong style="color:#C4B5FD">Analizar</strong>. Todo se procesa localmente — tus archivos nunca salen de este equipo.'
    '</p>'
    # Formatos aceptados
    '<div style="margin-bottom:10px;font-size:10px;font-weight:600;color:#6B6883;text-transform:uppercase;letter-spacing:.08em">'
    'Formatos aceptados</div>'
    '<div style="display:grid;grid-template-columns:auto 1fr;gap:8px 12px;align-items:center">'

    # Video row
    '<div style="display:flex;align-items:center;gap:5px;font-size:10px;font-weight:600;color:#A78BFA;'
    'text-transform:uppercase;letter-spacing:.05em">'
    + _svg('<polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>', 12) +
    ' Video</div>'
    '<div style="display:flex;flex-wrap:wrap;gap:5px">'
    + _chip("mp4") + _chip("mov") + _chip("avi") + _chip("webm") +
    '</div>'

    # Audio row
    '<div style="display:flex;align-items:center;gap:5px;font-size:10px;font-weight:600;color:#A78BFA;'
    'text-transform:uppercase;letter-spacing:.05em">'
    + _svg('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>', 12) +
    ' Audio</div>'
    '<div style="display:flex;flex-wrap:wrap;gap:5px">'
    + _chip("mp3") + _chip("wav") + _chip("m4a") + _chip("ogg") +
    '</div>'

    # Imagen row
    '<div style="display:flex;align-items:center;gap:5px;font-size:10px;font-weight:600;color:#A78BFA;'
    'text-transform:uppercase;letter-spacing:.05em">'
    + _svg('<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>', 12) +
    ' Imagen</div>'
    '<div style="display:flex;flex-wrap:wrap;gap:5px">'
    + _chip("jpg") + _chip("jpeg") + _chip("png") +
    '</div>'

    '</div>'  # fin grid

    '<div style="margin-top:12px">'
    + _chip("máx. 100 MB", "#6EE7B7", "rgba(16,185,129,.08)", "rgba(16,185,129,.2)") +
    '</div>'
    '</div>'
)

with gr.Blocks(
    title="Detector de Deepfakes",
    theme=gr.themes.Soft(
        primary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(HEADER_HTML)

    state = gr.State([])

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[(None, WELCOME_HTML)],
                elem_id="chatbot",
                label="Análisis",
                bubble_full_width=False,
                show_label=False,
                render_markdown=True,
            )

        with gr.Column(scale=1, elem_classes=["right-panel"]):
            gr.Markdown("### Subir archivo")
            file_input = gr.File(
                label="Arrastra aquí o haz clic para buscar",
                file_types=ACCEPTED_EXTENSIONS,
                elem_classes=["upload-area"],
            )
            analyse_btn = gr.Button("Analizar", variant="primary", size="lg")
            clear_btn   = gr.Button("Limpiar chat", variant="secondary")

            gr.Markdown(
                "**Modelos**\n\n"
                "Visual — `prithivMLmods/Deep-Fake-Detector-Model` (ViT)\n\n"
                "Audio — `garystafford/wav2vec2-deepfake-voice-detector` (Wav2Vec2)\n\n"
                "100% local · sin APIs externas · código abierto"
            )

    analyse_btn.click(
        fn=analyse_file,
        inputs=[file_input, state],
        outputs=[chatbot, state, file_input],
    )
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
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
