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

_IMAGE_PROMPT = (
    "Eres un experto forense en detección de deepfakes y contenido sintético generado por IA. "
    "Analiza esta imagen y determina si es real o generada/manipulada por IA.\n\n"
    "Busca específicamente:\n"
    "- Inconsistencias de iluminación (sombras que no coinciden con fuentes de luz)\n"
    "- Artefactos en los bordes del rostro (bordes borrosos, halos, fusión incompleta)\n"
    "- Textura de piel irreal (demasiado uniforme, poros ausentes, aspecto plástico)\n"
    "- Anomalías oculares (reflejos inconsistentes, pupila deformada)\n"
    "- Asimetría facial exagerada o simetría perfecta antinatural\n"
    "- Cabello, ropa o accesorios con detalles irreales o fusionados con el fondo\n"
    "- Artefactos de compresión o patrones repetitivos\n\n"
    'Responde ÚNICAMENTE con un JSON válido sin texto adicional:\n'
    '{"verdict":"REAL"|"SOSPECHOSO"|"DEEPFAKE","confidence":<0-100>,"evidence":["evidencia1",...]}\n\n'
    "- REAL: sin señales claras de manipulación\n"
    "- SOSPECHOSO: anomalías presentes pero no concluyente\n"
    "- DEEPFAKE: claramente generado o manipulado por IA"
)

_VIDEO_PROMPT = (
    "Eres un experto forense en detección de deepfakes y contenido sintético generado por IA. "
    "Analiza este video y determina si es real o generado/manipulado por IA.\n\n"
    "Busca específicamente:\n"
    "- Sincronización labial (¿los labios coinciden exactamente con el audio?)\n"
    "- Consistencia entre fotogramas (saltos, temblores, incoherencias entre frames)\n"
    "- Artefactos en movimiento (bordes borrosos cuando el sujeto se mueve)\n"
    "- Parpadeo antinatural (frecuencia, duración o ritmo mecánico)\n"
    "- Inconsistencias de iluminación entre frames consecutivos\n"
    "- Expresiones faciales robóticas o transiciones poco naturales\n"
    "- Movimiento del cabello o ropa inconsistente con la física real\n\n"
    'Responde ÚNICAMENTE con un JSON válido sin texto adicional:\n'
    '{"verdict":"REAL"|"SOSPECHOSO"|"DEEPFAKE","confidence":<0-100>,"evidence":["evidencia1",...]}'
)

_AUDIO_PROMPT = (
    "Eres un experto forense en detección de audio sintético generado por IA y voice cloning. "
    "Analiza este audio y determina si la voz es real o generada/clonada por IA.\n\n"
    "Busca específicamente:\n"
    "- Artefactos de síntesis de voz (clics, pop, discontinuidades, glitches)\n"
    "- Respiración antinatural (ausente, demasiado regular, o en momentos incorrectos)\n"
    "- Entonación robótica (monotonía, transiciones de tono mecánicas)\n"
    "- Prosodia mecánica (ritmo demasiado uniforme, pausas no naturales)\n"
    "- Ausencia de variaciones naturales de la voz humana\n"
    "- Ruido de fondo inconsistente o artificialmente limpio\n"
    "- Consonantes o vocales con calidad irreal\n\n"
    'Responde ÚNICAMENTE con un JSON válido sin texto adicional:\n'
    '{"verdict":"REAL"|"SOSPECHOSO"|"DEEPFAKE","confidence":<0-100>,"evidence":["evidencia1",...]}'
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
            path=str(path),
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
