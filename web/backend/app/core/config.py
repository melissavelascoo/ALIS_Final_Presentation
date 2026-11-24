"""Rutas de los modelos y parámetros del reconocimiento."""
import os
import sys

ALIS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if ALIS_ROOT not in sys.path:
    sys.path.insert(0, ALIS_ROOT)

from paths import (
    WEB_DIR,
    MLP_MODEL_PATH,
    MLP_LABELS_PATH,
    LSTM_MODEL_PATH,
    LSTM_LABELS_PATH,
)

FRONTEND_DIR = WEB_DIR / "frontend"

MLP_NUM_FEATURES = 63
LSTM_SEQUENCE_LENGTH = 30
LSTM_NUM_FEATURES = 454

CONFIDENCE_THRESHOLD = float(os.environ.get("ALIS_CONFIDENCE_THRESHOLD", "0.75"))
if not 0.0 < CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError("ALIS_CONFIDENCE_THRESHOLD debe estar entre 0 y 1.")

MLP_CONFIDENCE_THRESHOLD = float(os.environ.get("ALIS_MLP_CONFIDENCE_THRESHOLD", "0.70"))
if not 0.0 < MLP_CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError("ALIS_MLP_CONFIDENCE_THRESHOLD debe estar entre 0 y 1.")

MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
MAX_FRAME_BYTES = 5 * 1024 * 1024