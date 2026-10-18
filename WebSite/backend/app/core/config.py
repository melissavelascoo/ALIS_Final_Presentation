"""Rutas del modelo y parámetros del reconocimiento."""
import os
from pathlib import Path

# El modelo permanece en la carpeta HandSignDetector, fuera de website.
ALIS_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = Path(os.environ.get('ALIS_MODEL_DIR', str(ALIS_ROOT / 'HandSignDetector' / 'Model'))).expanduser().resolve()
MODEL_FILE = MODEL_DIR / 'keras_model.h5'
LABELS_FILE = MODEL_DIR / 'labels.txt'

CONFIDENCE_THRESHOLD = float(os.environ.get('ALIS_CONFIDENCE_THRESHOLD', '0.85'))
if not 0.0 < CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError('ALIS_CONFIDENCE_THRESHOLD debe estar entre 0 y 1.')

ALLOWED_CLASSES = {'A', 'E', 'I', 'O', 'U'}
MAX_FRAME_BYTES = 5 * 1024 * 1024
HAND_IMAGE_SIZE = 300
HAND_OFFSET = 20
