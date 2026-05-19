---
title: Detector de Deepfakes
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Deepfake Detector

Detecta contenido sintético o manipulado por IA en archivos de **video**, **audio** e **imagen** usando modelos open-source de Hugging Face que corren 100 % de forma local, sin APIs externas ni costos adicionales.

---

## Modelos utilizados

| Modalidad | Modelo | Arquitectura |
|-----------|--------|--------------|
| Imagen / Video | [`prithivMLmods/Deep-Fake-Detector-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model) | Vision Transformer (ViT) |
| Audio | [`MelodyMachine/Deepfake-audio-detection-V2`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) | Wav2Vec2 / XLSR |

Los modelos se descargan automáticamente en `~/.cache/huggingface/hub/` la primera vez que se ejecuta la app.

---

## Instalación local

### Requisitos previos

- Python 3.10 o superior
- `ffmpeg` instalado en el sistema:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: [descargar desde ffmpeg.org](https://ffmpeg.org/download.html)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/deepfake-detector.git
cd deepfake-detector

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Verificar instalación con los tests
python test_local.py

# 5. Lanzar la aplicación
python app.py
```

La interfaz estará disponible en `http://localhost:7860`.

### Con GPU (recomendado)

Si tienes una GPU NVIDIA con CUDA, instala el índice correcto de PyTorch:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Uso

1. Abre `http://localhost:7860` en tu navegador.
2. Arrastra o selecciona un archivo desde el panel derecho.
3. Haz clic en **Analizar** (o espera a que el archivo termine de subirse).
4. El chat mostrará el resultado con:
   - Veredicto: **Probablemente REAL** / **Probablemente IA/FAKE** / **Inconcluso**
   - Score de confianza en porcentaje
   - Para video: scores separados de video y audio
   - Barra de progreso ASCII

### Formatos aceptados

| Tipo | Extensiones |
|------|------------|
| Video | `.mp4` `.mov` `.avi` `.webm` |
| Audio | `.mp3` `.wav` `.m4a` `.ogg` |
| Imagen | `.jpg` `.jpeg` `.png` |

**Límite de tamaño:** 100 MB por archivo.

---

## Deploy en Hugging Face Spaces

1. Crea un nuevo Space en [huggingface.co/spaces](https://huggingface.co/spaces) con SDK **Gradio**.

2. Asegúrate de que el Space tenga una `app.py` en la raíz (ya está estructurado así).

3. Agrega un archivo `packages.txt` en la raíz con:
   ```
   ffmpeg
   ```

4. Sube todos los archivos del proyecto al Space:
   ```bash
   git remote add space https://huggingface.co/spaces/TU-USUARIO/deepfake-detector
   git push space main
   ```

5. En la configuración del Space, si usas hardware gratuito (CPU Basic) añade:
   ```
   GRADIO_SERVER_NAME=0.0.0.0
   ```

> **Nota:** En Spaces gratuitos (CPU) la primera inferencia tarda ~60–90 s porque descarga los modelos. Los modelos se cachean entre reinicios si el Space tiene persistencia activa.

---

## Limitaciones

- **No es infalible.** Los modelos fueron entrenados en datasets específicos y pueden fallar ante:
  - Técnicas de síntesis más recientes que sus datos de entrenamiento.
  - Video con alta compresión (bitrate muy bajo).
  - Audio con ruido de fondo intenso o efectos de voz.
  - Imágenes con resolución muy baja o recortadas.
- **Sesgo de dataset.** Ambos modelos fueron entrenados principalmente con rostros. Contenido sin rostros puede dar resultados menos fiables.
- **Velocidad en CPU.** Un video de 30 s tarda ~2–4 min en CPU. Con GPU es ~10–20 s.
- **Resultado combinado.** Para video, el veredicto final pondera video (60 %) y audio (40 %). Si el video no tiene audio, solo se usa el score visual.

---

## Estructura del proyecto

```
deepfake-detector/
├── app.py                    # Interfaz Gradio (punto de entrada)
├── detectors/
│   ├── __init__.py
│   ├── video_detector.py     # Detector de imagen/frames con ViT
│   ├── audio_detector.py     # Detector de audio con Wav2Vec2
│   └── orchestrator.py       # Fusión de resultados
├── utils/
│   ├── __init__.py
│   ├── media_processing.py   # Extracción de frames y audio (ffmpeg + OpenCV)
│   └── aggregation.py        # Agregación y fusión de scores
├── test_local.py             # Tests con fixtures sintéticos
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tests

```bash
python test_local.py
```

El script genera fixtures sintéticos (imagen gris, audio sinusoidal, video de color sólido) sin necesidad de archivos reales, y ejecuta toda la cadena de detección.

---

## Licencia

MIT. Los modelos de Hugging Face tienen sus propias licencias; revisa cada repositorio antes de uso comercial.
