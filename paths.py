from pathlib import Path


# ============================================================
# DIRECTORIO PRINCIPAL
# ============================================================

ALIS_DIR = Path(__file__).resolve().parent

SLR_DIR = ALIS_DIR / "slr"
WEB_DIR = ALIS_DIR / "web"


# ============================================================
# FASE 1: STATIC MLP
# ============================================================

# ----------------------------
# DATOS
# ----------------------------

MLP_DATA_DIR = SLR_DIR / "data" / "static_signs"

MLP_RAW_DIR = MLP_DATA_DIR / "raw"
MLP_IMAGES_DIR = MLP_RAW_DIR / "images"
MLP_RAW_CSV = MLP_RAW_DIR / "landmarks_raw.csv"

MLP_PROCESSED_DIR = MLP_DATA_DIR / "processed"
MLP_CLEAN_CSV = MLP_PROCESSED_DIR / "landmarks_clean.csv"

MLP_SPLITS_DIR = MLP_PROCESSED_DIR / "splits"
MLP_TRAIN_CSV = MLP_SPLITS_DIR / "train.csv"
MLP_VAL_CSV = MLP_SPLITS_DIR / "val.csv"
MLP_TEST_CSV = MLP_SPLITS_DIR / "test.csv"


# ----------------------------
# MODELO
# ----------------------------

MLP_MODEL_DIR = SLR_DIR / "models" / "static_signs"

MLP_MODEL_PATH = MLP_MODEL_DIR / "model.h5"
MLP_LABELS_PATH = MLP_MODEL_DIR / "labels.txt"


# ----------------------------
# REPORTES
# ----------------------------

MLP_REPORTS_DIR = SLR_DIR / "reports" / "static_signs"

MLP_CLEAN_CSV_REPORT = MLP_REPORTS_DIR / "cleaning_report.txt"
MLP_TRAINING_REPORT = MLP_REPORTS_DIR / "training_report.txt"
MLP_TRAINING_CURVES = MLP_REPORTS_DIR / "training_curves.png"
MLP_TEST_REPORT = MLP_REPORTS_DIR / "test_report.txt"


# ============================================================
# FASE 2: DYNAMIC LSTM
# ============================================================

# ----------------------------
# DATOS
# ----------------------------

LSTM_DATA_DIR = SLR_DIR / "data" / "dynamic_signs"

LSTM_VIDEOS_DIR = LSTM_DATA_DIR / "videos_raw"
LSTM_RAW_DIR = LSTM_DATA_DIR / "raw"

# Datos procesados
LSTM_PROCESSED_DIR = LSTM_DATA_DIR / "processed"

LSTM_X_PATH = LSTM_PROCESSED_DIR / "X.npy"
LSTM_Y_PATH = LSTM_PROCESSED_DIR / "y.npy"
LSTM_PROCESSED_LABELS_PATH = LSTM_PROCESSED_DIR / "labels.json"

# Separacion Train / Validation / Test
LSTM_SPLITS_DIR = LSTM_DATA_DIR / "splits"

LSTM_TRAIN_NPZ = LSTM_SPLITS_DIR / "train.npz"
LSTM_VAL_NPZ = LSTM_SPLITS_DIR / "val.npz"
LSTM_TEST_NPZ = LSTM_SPLITS_DIR / "test.npz"


# ----------------------------
# MODELO
# ----------------------------

LSTM_MODEL_DIR = SLR_DIR / "models" / "dynamic_signs"

LSTM_MODEL_PATH = LSTM_MODEL_DIR / "model.h5"
LSTM_LABELS_PATH = LSTM_MODEL_DIR / "labels.txt"


# ----------------------------
# REPORTES
# ----------------------------

LSTM_REPORTS_DIR = SLR_DIR / "reports" / "dynamic_signs"

LSTM_TRAINING_REPORT = LSTM_REPORTS_DIR / "training_report.txt"
LSTM_TRAINING_CURVES = LSTM_REPORTS_DIR / "training_curves.png"
LSTM_TEST_REPORT = LSTM_REPORTS_DIR / "test_report.txt"