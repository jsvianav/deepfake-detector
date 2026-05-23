"""Análisis de deepfakes usando Gemini 2.5 Flash (google-genai SDK)."""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"

_IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
}
_AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".aac": "audio/aac",
}
_VIDEO_MIME = {
    ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".webm": "video/webm",
}

_EVIDENCE_RULES = (
    "\n\nReglas para el campo 'evidence':\n"
    "- Máximo 3 puntos.\n"
    "- Cada punto: frase corta, máximo 10 palabras, en español simple y directo.\n"
    "- Sin jerga técnica. Escribe como si se lo explicaras a alguien sin conocimientos de IA.\n"
    "- Ejemplos de estilo correcto: 'La piel se ve demasiado lisa y artificial', "
    "'Los labios no coinciden con lo que dice', 'La voz suena robótica y sin emoción'."
)

_JSON_FORMAT = (
    '\nResponde ÚNICAMENTE con un JSON válido sin texto adicional:\n'
    '{"verdict":"REAL"|"SOSPECHOSO"|"DEEPFAKE","confidence":<0-100>,"evidence":["frase corta 1",...]}\n'
    "- REAL: parece auténtico\n"
    "- SOSPECHOSO: hay dudas pero no es concluyente\n"
    "- DEEPFAKE: claramente falso o generado por IA"
)

_IMAGE_PROMPT = (
    "Analiza esta imagen y determina si es real o fue creada/modificada por IA.\n\n"
    "Revisa: iluminación inconsistente, bordes del rostro borrosos, "
    "piel demasiado uniforme, ojos con reflejos raros, cabello o fondo con detalles extraños."
    + _EVIDENCE_RULES + _JSON_FORMAT
)

_VIDEO_PROMPT = (
    "Analiza este video y determina si es real o fue creado/manipulado por IA.\n\n"
    "Revisa: si los labios coinciden con lo que se dice, movimientos bruscos entre fotogramas, "
    "parpadeo extraño, expresiones faciales poco naturales, iluminación que cambia sin razón."
    + _EVIDENCE_RULES + _JSON_FORMAT
)

_AUDIO_PROMPT = (
    "Analiza este audio y determina si la voz es real o fue generada/clonada por IA.\n\n"
    "Revisa: si suena robótica o monótona, si falta respiración natural, "
    "pausas en lugares raros, calidad demasiado perfecta o artificial."
    + _EVIDENCE_RULES + _JSON_FORMAT
)


class DeepfakeAnalyzer:
    """Analiza imágenes, audio y video usando Gemini 2.5 Flash."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY no encontrada. "
                "Configúrala como variable de entorno o en el archivo .env"
            )
        self._client = genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def analyze_image(self, file_path_or_bytes: Union[str, Path, bytes]) -> dict:
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            ext = path.suffix.lower()
            with open(path, "rb") as f:
                data = f.read()
        else:
            data = file_path_or_bytes
            ext = ".jpg"

        if len(data) > 20 * 1024 * 1024:
            return self._error("Imagen demasiado grande para análisis inline (máx. 20 MB).")

        mime_type = _IMAGE_MIME.get(ext, "image/jpeg")
        try:
            response = self._client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=data, mime_type=mime_type),
                    _IMAGE_PROMPT,
                ],
            )
            return self._parse(response.text)
        except Exception as exc:
            logger.exception("Error analizando imagen")
            return self._error(str(exc))

    def analyze_audio(self, file_path: Union[str, Path]) -> dict:
        path = Path(file_path)
        ext = path.suffix.lower()
        mime_type = _AUDIO_MIME.get(ext, "audio/mpeg")
        try:
            uploaded = self._upload_file(path, mime_type, poll=False)
            try:
                response = self._client.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type),
                        _AUDIO_PROMPT,
                    ],
                )
                return self._parse(response.text)
            finally:
                self._safe_delete(uploaded.name)
        except Exception as exc:
            logger.exception("Error analizando audio")
            return self._error(str(exc))

    def analyze_video(self, file_path: Union[str, Path]) -> dict:
        path = Path(file_path)
        ext = path.suffix.lower()
        mime_type = _VIDEO_MIME.get(ext, "video/mp4")
        try:
            uploaded = self._upload_file(path, mime_type, poll=True)
            try:
                response = self._client.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type),
                        _VIDEO_PROMPT,
                    ],
                )
                return self._parse(response.text)
            finally:
                self._safe_delete(uploaded.name)
        except Exception as exc:
            logger.exception("Error analizando video")
            return self._error(str(exc))

    def analyze(self, file_path: Union[str, Path], media_type: str | None = None) -> dict:
        """Detecta el tipo de archivo automáticamente y llama al método correcto."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if media_type is None:
            if ext in _IMAGE_MIME:
                media_type = "image"
            elif ext in _AUDIO_MIME:
                media_type = "audio"
            elif ext in _VIDEO_MIME:
                media_type = "video"
            else:
                return self._error(f"Formato no soportado: {ext}")

        if media_type == "image":
            result = self.analyze_image(path)
        elif media_type == "audio":
            result = self.analyze_audio(path)
        elif media_type == "video":
            result = self.analyze_video(path)
        else:
            return self._error(f"Tipo de medio desconocido: {media_type}")

        result["media_type"] = media_type
        return result

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _upload_file(self, path: Path, mime_type: str, poll: bool = False):
        uploaded = self._client.files.upload(
            file=str(path),
            config=types.UploadFileConfig(
                mime_type=mime_type,
                display_name=path.name,
            ),
        )
        if poll:
            deadline = time.time() + 180
            while uploaded.state.name == "PROCESSING":
                if time.time() > deadline:
                    raise TimeoutError(
                        "El archivo tardó demasiado en procesarse en Gemini (>3 min)."
                    )
                time.sleep(5)
                uploaded = self._client.files.get(name=uploaded.name)
            if uploaded.state.name != "ACTIVE":
                raise RuntimeError(
                    f"El archivo no pudo procesarse: estado={uploaded.state.name}"
                )
        return uploaded

    def _safe_delete(self, name: str) -> None:
        try:
            self._client.files.delete(name=name)
        except Exception:
            pass

    def _parse(self, text: str) -> dict:
        try:
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                verdict = str(data.get("verdict", "SOSPECHOSO")).upper()
                if verdict not in ("REAL", "SOSPECHOSO", "DEEPFAKE"):
                    verdict = "SOSPECHOSO"
                confidence = max(0, min(100, int(data.get("confidence", 50))))
                evidence = data.get("evidence", [])
                if not isinstance(evidence, list):
                    evidence = [str(evidence)]
                return {
                    "verdict": verdict,
                    "confidence": confidence,
                    "evidence": [str(e) for e in evidence],
                    "raw_response": text,
                }
        except Exception:
            logger.warning("No se pudo parsear JSON de respuesta Gemini: %s", text[:300])
        return {
            "verdict": "SOSPECHOSO",
            "confidence": 50,
            "evidence": ["No se pudo interpretar la respuesta del modelo."],
            "raw_response": text,
        }

    @staticmethod
    def _error(msg: str) -> dict:
        return {
            "verdict": "SOSPECHOSO",
            "confidence": 0,
            "evidence": [msg],
            "raw_response": "",
            "error": msg,
        }
