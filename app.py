"""Detector de Deepfakes — Interfaz Gradio con tema oscuro premium.

Carga ambos modelos una vez al iniciar y sirve inferencia a través de una
interfaz Blocks que acepta archivos multimedia por arrastrar y soltar.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# huggingface_hub>=0.26 removed HfFolder; gradio[oauth] still imports it at load time.
# Inject a stub BEFORE importing gradio so the import chain succeeds.
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
        @staticmethod
        def delete_token(): pass
    _hfhub.HfFolder = _HfFolder

import gradio as gr

# gradio_client bug: _json_schema_to_python_type() crashes when additionalProperties is a
# boolean (True) instead of a dict. Patch the module-level name so recursive calls also
# hit the guard (Python resolves LOAD_GLOBAL through the module __dict__ at call time).
import gradio_client.utils as _gc_utils
_orig_schema_to_py = _gc_utils._json_schema_to_python_type
def _patched_schema_to_py(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_schema_to_py(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_schema_to_py

import torch

sys.path.insert(0, str(Path(__file__).parent))

# Detectar si corremos en Hugging Face Spaces
_IS_SPACES = os.environ.get("SPACE_ID") is not None

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

# En Spaces de CPU gratuito, reducir frames para que el análisis sea más rápido
_max_frames = 8 if (_IS_SPACES and device == "cpu") else 30
logger.info("Frames por video: %d (%s)", _max_frames, "Spaces CPU" if _max_frames == 8 else "completo")

_orchestrator = Orchestrator(_image_detector, _audio_detector, frame_fps=1.0, max_frames=_max_frames)
logger.info("=== Modelos listos ===")

# ---------------------------------------------------------------------------
# Extensiones aceptadas
# ---------------------------------------------------------------------------
ACCEPTED_EXTENSIONS = [
    ".mp4", ".mov", ".avi", ".webm",
    ".mp3", ".wav", ".m4a", ".ogg",
    ".jpg", ".jpeg", ".png",
]

# ---------------------------------------------------------------------------
# Iconos SVG (atributos HTML, no camelCase de React)
# ---------------------------------------------------------------------------
def _svg(path_content: str, w: int = 15) -> str:
    attrs = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
    return f'<svg {attrs} width="{w}" height="{w}" class="df-svg-icon">{path_content}</svg>'

_ICON_VIDEO  = _svg('<polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>')
_ICON_AUDIO  = _svg('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>')
_ICON_IMAGE  = _svg('<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>')
_ICON_SEARCH = _svg('<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/>')
_ICON_ALERT  = (
    '<svg class="df-icon-alert" viewBox="0 0 24 24" fill="none" stroke="#F59E0B"'
    ' stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
    '</svg>'
)

_TYPE_ICON = {"video": _ICON_VIDEO, "audio": _ICON_AUDIO, "image": _ICON_IMAGE}

# ---------------------------------------------------------------------------
# CSS — tema oscuro premium
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

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
  font-family: 'Poppins', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
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
.page-hdr {
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  padding: 28px 0 20px; border-bottom: 1px solid rgba(255,255,255,.05);
  margin-bottom: 24px;
}
.page-hdr-icon {
  width:38px; height:38px; border-radius:10px;
  background: linear-gradient(135deg,#7C3AED,#5B21B6);
  display:flex; align-items:center; justify-content:center;
  box-shadow: 0 0 24px rgba(124,58,237,.5); flex-shrink:0;
}
.page-hdr-text { flex:1; min-width:0; }
.page-hdr-title { font-size:20px; font-weight:700; color:#F8F7FF; letter-spacing:-.02em; line-height:1.1; }
.page-hdr-grad  { background:linear-gradient(135deg,#C4B5FD,#A78BFA); -webkit-background-clip:text; background-clip:text; color:transparent; }
.page-hdr-sub   { font-size:12px; color:#6B6883; margin-top:3px; }
.status-dot {
  width:5px; height:5px; border-radius:50%; background:#10B981;
  box-shadow:0 0 6px #10B981; display:inline-block;
  animation: pulseDot 2s ease-in-out infinite;
}
.status-pill {
  margin-left:auto; display:inline-flex; align-items:center; gap:7px;
  font-size:11px; color:#6EE7B7; background:rgba(16,185,129,.08);
  border:1px solid rgba(16,185,129,.2); border-radius:999px; padding:4px 11px;
  font-family:'JetBrains Mono',monospace;
}

/* ── Panel de chat (gr.HTML) ─────────────────────────────── */
#chat-panel {
  background: rgba(12,11,20,.75) !important;
  border: 1px solid rgba(124,58,237,.15) !important;
  border-radius: 18px !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.04),
    0 24px 60px -20px rgba(0,0,0,.65) !important;
  overflow: hidden !important;
}
#chat-panel > .wrap { padding:0 !important; }

.chat-scroll {
  height: 620px;
  overflow-y: auto;
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-sizing: border-box;
}
.chat-scroll::-webkit-scrollbar { width:4px; }
.chat-scroll::-webkit-scrollbar-thumb { background:rgba(124,58,237,.35); border-radius:2px; }

.chat-bubble-user {
  align-self: flex-end;
  background: rgba(124,58,237,.08);
  color: #C4B5FD;
  border: 1px solid rgba(124,58,237,.2);
  border-radius: 14px 14px 4px 14px;
  padding: 10px 14px;
  font-size: 13.5px;
  max-width: 80%;
  animation: fadeSlideUp 0.25s ease both;
  font-family: 'Poppins', system-ui, sans-serif;
}

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
.upload-area label, .upload-area span, .upload-area button { color:#F8F7FF !important; font-size:13px !important; }
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

/* ══════════════════════════════════════════════════════════ */
/* ── Tarjetas de resultado (clases en lugar de style="") ── */
/* ══════════════════════════════════════════════════════════ */
.df-svg-icon { display:inline-block; vertical-align:middle; }
.df-icon-alert { width:13px; height:13px; flex-shrink:0; margin-top:1px; }
.df-bar-svg { display:block; }

.df-card { border-radius:14px; overflow:hidden; background:rgba(18,17,26,.92); border:1px solid rgba(124,58,237,.22); box-shadow:0 20px 48px -16px rgba(0,0,0,.7); font-family:'Poppins',system-ui,sans-serif; animation:scaleIn .45s cubic-bezier(.2,.7,.2,1) both; }
.df-card-real { border-color:rgba(16,185,129,.28) !important; box-shadow:0 20px 48px -16px rgba(0,0,0,.7),0 0 40px -8px rgba(16,185,129,.16) !important; }
.df-card-fake { border-color:rgba(239,68,68,.28) !important;  box-shadow:0 20px 48px -16px rgba(0,0,0,.7),0 0 40px -8px rgba(239,68,68,.20)  !important; }
.df-card-amb  { border-color:rgba(245,158,11,.28) !important; box-shadow:0 20px 48px -16px rgba(0,0,0,.7),0 0 40px -8px rgba(245,158,11,.14) !important; }
.df-card-err  { border-color:rgba(245,158,11,.30) !important; }

.df-hdr { display:flex; align-items:center; gap:10px; padding:13px 18px; border-bottom:1px solid rgba(255,255,255,.06); }
.df-hdr-icon { width:32px; height:32px; border-radius:9px; background:rgba(124,58,237,.1); border:1px solid rgba(124,58,237,.2); display:flex; align-items:center; justify-content:center; color:#A78BFA; flex-shrink:0; }
.df-hdr-name { flex:1; min-width:0; font-family:'JetBrains Mono',monospace; font-size:12px; color:#F8F7FF; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:500; }
.df-err-body { padding:20px; color:#FCA5A5; font-size:13px; line-height:1.55; }

.df-chip { display:inline-block; padding:3px 9px; border-radius:6px; background:rgba(124,58,237,.1); border:1px solid rgba(124,58,237,.2); font-size:10px; font-weight:600; color:#A78BFA; letter-spacing:.06em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; }
.df-chip-green { background:rgba(16,185,129,.08); border-color:rgba(16,185,129,.2); color:#6EE7B7; }

.df-verdict-wrap { padding:28px 22px 20px; text-align:center; }
.df-pill { display:inline-flex; align-items:center; gap:7px; padding:5px 12px 5px 10px; border-radius:999px; font-size:11px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin-bottom:18px; animation:fadeIn .4s ease .2s both; }
.df-pill-real { background:rgba(16,185,129,.1); color:#6EE7B7; border:1px solid rgba(16,185,129,.3); }
.df-pill-fake { background:rgba(239,68,68,.1);  color:#FCA5A5; border:1px solid rgba(239,68,68,.3); }
.df-pill-amb  { background:rgba(245,158,11,.1); color:#FCD34D; border:1px solid rgba(245,158,11,.3); }
.df-dot { width:6px; height:6px; border-radius:50%; display:inline-block; }
.df-dot-real { background:#10B981; box-shadow:0 0 6px #10B981; }
.df-dot-fake { background:#EF4444; box-shadow:0 0 6px #EF4444; }
.df-dot-amb  { background:#F59E0B; box-shadow:0 0 6px #F59E0B; }

.df-score-big { font-size:60px; font-weight:600; letter-spacing:-.04em; line-height:1; color:#F8F7FF; font-variant-numeric:tabular-nums; }
.df-score-pct { font-size:24px; color:#6B6883; font-weight:500; margin-left:3px; }
.df-score-sub { font-size:12px; color:#6B6883; margin-top:8px; }

.df-bar-wrap { padding:0 22px 22px; }
.df-bar-scale { display:flex; justify-content:space-between; margin-top:10px; font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:.04em; }
.df-bar-real { color:#6EE7B7; }
.df-bar-mid  { color:#6B6883; }
.df-bar-fake { color:#FCA5A5; }

.df-breakdown { border-top:1px solid rgba(255,255,255,.06); padding:14px 22px 16px; display:grid; grid-template-columns:1fr 1fr; }
.df-bd-left  { padding-right:14px; }
.df-bd-right { padding-left:14px; border-left:1px solid rgba(255,255,255,.06); }
.df-mini-lbl  { display:flex; align-items:center; gap:6px; font-size:10px; color:#6B6883; text-transform:uppercase; letter-spacing:.07em; font-weight:500; margin-bottom:7px; }
.df-mini-icon { color:#A78BFA; }
.df-mini-num  { font-size:20px; font-weight:600; letter-spacing:-.025em; margin-bottom:6px; }
.df-mini-real { color:#10B981; }
.df-mini-fake { color:#EF4444; }
.df-mini-na   { color:#6B6883; }
.df-mini-pct  { font-size:12px; color:#6B6883; margin-left:1px; font-family:'JetBrains Mono',monospace; }
.df-mini-meta { font-size:10px; color:#6B6883; font-family:'JetBrains Mono',monospace; margin-top:4px; }

.df-details { border-top:1px solid rgba(255,255,255,.06); }
.df-details > summary { list-style:none; padding:13px 22px; cursor:pointer; font-size:12.5px; color:#B8B5C8; display:flex; align-items:center; justify-content:space-between; user-select:none; }
.df-details-arrow { font-size:10px; color:#6B6883; font-family:'JetBrains Mono',monospace; }
.df-grid { padding:0 22px 16px; display:grid; grid-template-columns:max-content 1fr; gap:7px 20px; }
.df-k { color:#6B6883; font-family:'JetBrains Mono',monospace; font-size:11px; }
.df-v { color:#F8F7FF; font-weight:500; font-family:'JetBrains Mono',monospace; font-size:11px; }

.df-notice { border-top:1px solid rgba(255,255,255,.06); padding:12px 22px; background:rgba(0,0,0,.2); font-size:11.5px; color:#B8B5C8; line-height:1.5; display:flex; gap:8px; align-items:flex-start; }

/* ── Welcome card ─────────────────────────────────────────── */
.df-welcome { border-radius:14px; overflow:hidden; background:rgba(124,58,237,.06); border:1px solid rgba(124,58,237,.15); padding:24px 22px; font-family:'Poppins',system-ui,sans-serif; animation:fadeIn .5s ease both; }
.df-welcome-hdr { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
.df-welcome-ico { width:30px; height:30px; border-radius:8px; background:linear-gradient(135deg,#7C3AED,#5B21B6); display:flex; align-items:center; justify-content:center; box-shadow:0 0 14px rgba(124,58,237,.4); }
.df-welcome-ttl { font-size:14px; font-weight:600; color:#F8F7FF; letter-spacing:-.01em; }
.df-welcome-dsc { font-size:13px; color:#9CA3AF; line-height:1.65; margin:0 0 16px 0; }
.df-welcome-hl  { color:#C4B5FD; }
.df-fmt-lbl  { margin-bottom:10px; font-size:10px; font-weight:600; color:#6B6883; text-transform:uppercase; letter-spacing:.08em; }
.df-fmt-grid { display:grid; grid-template-columns:auto 1fr; gap:8px 12px; align-items:center; }
.df-fmt-type { display:flex; align-items:center; gap:5px; font-size:10px; font-weight:600; color:#A78BFA; text-transform:uppercase; letter-spacing:.05em; }
.df-chips-row { display:flex; flex-wrap:wrap; gap:5px; }
.df-mt12 { margin-top:12px; }
"""

# ---------------------------------------------------------------------------
# Helpers HTML para tarjetas de resultados
# ---------------------------------------------------------------------------

def _chip(text: str, kind: str = "") -> str:
    extra = " df-chip-green" if kind == "green" else ""
    return f'<span class="df-chip{extra}">{text}</span>'


def _mini_bar(score_val: Optional[float], label: str, icon: str) -> str:
    lbl = f'<div class="df-mini-lbl"><span class="df-mini-icon">{icon}</span>{label}</div>'
    if score_val is None:
        return (
            lbl
            + '<div class="df-mini-num df-mini-na">—</div>'
            + '<svg class="df-bar-svg" width="100%" height="3" viewBox="0 0 100 3" preserveAspectRatio="none">'
            + '<rect x="0" y="0" width="100" height="3" rx="1.5" fill="#1B1A26"/>'
            + '</svg>'
            + '<div class="df-mini-meta">sin datos</div>'
        )
    p = score_val * 100
    sc_cls = "df-mini-fake" if score_val > 0.5 else "df-mini-real"
    fill   = "#EF4444" if score_val > 0.5 else "#10B981"
    return (
        lbl
        + f'<div class="df-mini-num {sc_cls}">{p:.1f}<span class="df-mini-pct">%</span></div>'
        + f'<svg class="df-bar-svg" width="100%" height="3" viewBox="0 0 100 3" preserveAspectRatio="none">'
        + f'<rect x="0" y="0" width="100" height="3" rx="1.5" fill="#1B1A26"/>'
        + f'<rect x="0" y="0" width="{p:.2f}" height="3" rx="1.5" fill="{fill}"/>'
        + '</svg>'
    )


# ---------------------------------------------------------------------------
# Renderizador de tarjeta de resultado
# ---------------------------------------------------------------------------

def _format_result(result: DetectionResult, file_name: str) -> str:
    """Genera una tarjeta HTML animada con el resultado de detección."""

    if result.error:
        return (
            '<div class="df-card df-card-err">'
            '<div class="df-hdr">'
            f'<div class="df-hdr-icon">{_ICON_IMAGE}</div>'
            f'<div class="df-hdr-name">{file_name}</div>'
            + _chip("ERROR") +
            '</div>'
            f'<div class="df-err-body">{result.error}</div>'
            '</div>'
        )

    score = result.fused_score
    pct   = score * 100

    if score > 0.57:
        verdict_label = "Contenido sintético"
        fill_c = "#EF4444"
        card_cls, pill_cls, dot_cls = "df-card-fake", "df-pill-fake", "df-dot-fake"
    elif score <= 0.40:
        verdict_label = "Contenido auténtico"
        fill_c = "#10B981"
        card_cls, pill_cls, dot_cls = "df-card-real", "df-pill-real", "df-dot-real"
    else:
        verdict_label = "Resultado inconcluso"
        fill_c = "#F59E0B"
        card_cls, pill_cls, dot_cls = "df-card-amb", "df-pill-amb", "df-dot-amb"

    thumb      = _TYPE_ICON.get(result.file_type, _ICON_IMAGE)
    type_label = result.file_type.upper()

    # ── Desglose (solo video) ──────────────────────────────────────────
    breakdown = ""
    if result.file_type == "video":
        audio_score = result.audio_score if result.had_audio else None
        audio_meta  = "wav2vec2 · 16 kHz" if result.had_audio else "sin pista de audio"
        breakdown = (
            '<div class="df-breakdown">'
            '<div class="df-bd-left">'
            + _mini_bar(result.video_score, "Visual", _ICON_VIDEO)
            + f'<div class="df-mini-meta">{result.frame_count} fotogramas · 1 fps</div>'
            + '</div>'
            '<div class="df-bd-right">'
            + _mini_bar(audio_score, "Audio", _ICON_AUDIO)
            + f'<div class="df-mini-meta">{audio_meta}</div>'
            + '</div>'
            '</div>'
        )

    # ── Filas de detalles forenses ─────────────────────────────────────
    def _tr(k: str, v: str) -> str:
        return f'<span class="df-k">{k}</span><span class="df-v">{v}</span>'

    tech = _tr("puntaje_fusionado", f"{result.fused_score:.4f}")
    if result.video_score is not None:
        tech += _tr("puntaje_video", f"{result.video_score:.4f}")
    if result.audio_score is not None:
        tech += _tr("puntaje_audio", f"{result.audio_score:.4f}")
    tech += _tr("fotogramas",   str(result.frame_count))
    tech += _tr("tiene_audio",  "sí" if result.had_audio else "no")
    tech += _tr("tipo_archivo", result.file_type)
    tech += _tr("dispositivo",  device)

    # Barra principal con SVG (los atributos SVG no son sanitizados)
    nx = max(1.0, min(99.0, pct))
    bar_svg = (
        f'<svg class="df-bar-svg" width="100%" height="12" viewBox="0 0 100 12" preserveAspectRatio="none">'
        f'<rect x="0" y="3" width="100" height="6" rx="3" fill="#1B1A26"/>'
        f'<rect x="0" y="3" width="{pct:.2f}" height="6" rx="3" fill="{fill_c}"/>'
        f'<rect x="{nx:.2f}" y="0" width="2" height="12" rx="1" fill="white" transform="translate(-1,0)"/>'
        f'</svg>'
    )

    # ── Ensamblado ─────────────────────────────────────────────────────
    return (
        f'<div class="df-card {card_cls}">'

        f'<div class="df-hdr">'
        f'<div class="df-hdr-icon">{thumb}</div>'
        f'<div class="df-hdr-name">{file_name}</div>'
        + _chip(type_label) +
        '</div>'

        f'<div class="df-verdict-wrap">'
        f'<div class="df-pill {pill_cls}">'
        f'<span class="df-dot {dot_cls}"></span>'
        f'{verdict_label}</div>'
        f'<div class="df-score-big">{pct:.1f}<span class="df-score-pct">%</span></div>'
        f'<div class="df-score-sub">probabilidad de contenido sintético</div>'
        f'</div>'

        f'<div class="df-bar-wrap">'
        + bar_svg +
        f'<div class="df-bar-scale">'
        f'<span class="df-bar-real">0% — Auténtico</span>'
        f'<span class="df-bar-mid">50</span>'
        f'<span class="df-bar-fake">100% — Sintético</span>'
        f'</div></div>'

        + breakdown +

        '<details class="df-details">'
        '<summary>'
        '<span>Detalles forenses</span>'
        '<span class="df-details-arrow">▼</span>'
        '</summary>'
        '<div class="df-grid">'
        + tech +
        '</div></details>'

        '<div class="df-notice">'
        + _ICON_ALERT +
        '<span>La detección es probabilística, no definitiva. El resultado es una señal de referencia — '
        'revisa los detalles forenses y el contexto antes de sacar conclusiones.</span>'
        '</div>'

        '</div>'
    )


# ---------------------------------------------------------------------------
# Renderizador de historial de chat
# ---------------------------------------------------------------------------

def _render_chat(items: list) -> str:
    """Convierte la lista de mensajes en HTML completo para gr.HTML."""
    if not items:
        return f'<div class="chat-scroll">{WELCOME_HTML}</div>'
    parts = []
    for item in items:
        if item.get("user"):
            parts.append(
                f'<div class="chat-bubble-user">📎 <strong>{item["user"]}</strong></div>'
            )
        parts.append(item["bot"])
    return '<div class="chat-scroll">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# Manejador principal de inferencia
# ---------------------------------------------------------------------------

def analyse_file(file_obj, history: list):
    if file_obj is None:
        err = '<div class="df-notice" style="border-radius:10px;margin-top:4px">⚠️ Por favor sube un archivo antes de analizar.</div>'
        history = history + [{"user": "", "bot": err}]
        return _render_chat(history), history, None

    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))

    if not file_path or not Path(file_path).is_file():
        err = '<div class="df-notice" style="border-radius:10px;margin-top:4px">⚠️ Archivo inválido. Por favor sube un video, audio o imagen.</div>'
        history = history + [{"user": "", "bot": err}]
        return _render_chat(history), history, None

    file_name = Path(file_path).name

    try:
        result: DetectionResult = _orchestrator.analyse(file_path)
        bot_html = _format_result(result, file_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado al analizar %s", file_path)
        bot_html = f'<div class="df-notice" style="border-radius:10px;margin-top:4px">❌ Error inesperado: {exc}</div>'

    history = history + [{"user": file_name, "bot": bot_html}]
    return _render_chat(history), history, None


def clear_history():
    return _render_chat([]), [], None


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------

HEADER_HTML = (
    '<div class="page-hdr">'
    '<div class="page-hdr-icon">'
    '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
    '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>'
    '</div>'
    '<div class="page-hdr-text">'
    '<div class="page-hdr-title">Detector de <span class="page-hdr-grad">Deepfakes</span></div>'
    '<div class="page-hdr-sub">Imagen ViT · Audio Wav2Vec2 · Inferencia 100% local</div>'
    '</div>'
    f'<div class="status-pill"><span class="status-dot"></span> Listo · {device.upper()}</div>'
    '</div>'
)

WELCOME_HTML = (
    '<div class="df-welcome">'
    '<div class="df-welcome-hdr">'
    '<div class="df-welcome-ico">'
    + _svg('<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/>', 14) +
    '</div>'
    '<div class="df-welcome-ttl">Listo para analizar</div>'
    '</div>'
    '<p class="df-welcome-dsc">'
    'Sube un archivo de <strong class="df-welcome-hl">video</strong>, '
    '<strong class="df-welcome-hl">audio</strong> o '
    '<strong class="df-welcome-hl">imagen</strong> en el panel de la derecha y haz clic en '
    '<strong class="df-welcome-hl">Analizar</strong>. '
    'Todo se procesa localmente — tus archivos nunca salen de este equipo.'
    '</p>'
    '<div class="df-fmt-lbl">Formatos aceptados</div>'
    '<div class="df-fmt-grid">'

    '<div class="df-fmt-type">'
    + _svg('<polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>', 12) +
    ' Video</div>'
    '<div class="df-chips-row">'
    + _chip("mp4") + _chip("mov") + _chip("avi") + _chip("webm") +
    '</div>'

    '<div class="df-fmt-type">'
    + _svg('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>', 12) +
    ' Audio</div>'
    '<div class="df-chips-row">'
    + _chip("mp3") + _chip("wav") + _chip("m4a") + _chip("ogg") +
    '</div>'

    '<div class="df-fmt-type">'
    + _svg('<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>', 12) +
    ' Imagen</div>'
    '<div class="df-chips-row">'
    + _chip("jpg") + _chip("jpeg") + _chip("png") +
    '</div>'

    '</div>'
    '<div class="df-mt12">'
    + _chip("máx. 100 MB", "green") +
    '</div>'
    '</div>'
)

with gr.Blocks(
    title="Detector de Deepfakes",
    theme=gr.themes.Soft(
        primary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Poppins"), "system-ui", "sans-serif"],
    ),
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(HEADER_HTML)

    state = gr.State([])

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chat_panel = gr.HTML(
                value=_render_chat([]),
                elem_id="chat-panel",
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

    analyse_btn.click(
        fn=analyse_file,
        inputs=[file_input, state],
        outputs=[chat_panel, state, file_input],
    )
    file_input.upload(
        fn=analyse_file,
        inputs=[file_input, state],
        outputs=[chat_panel, state, file_input],
    )
    clear_btn.click(
        fn=clear_history,
        outputs=[chat_panel, state, file_input],
    )

# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0" if _IS_SPACES else "127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
