import numpy as np
import pandas as pd

from paths import MLP_CLEAN_CSV, MLP_SPLITS_DIR

# ============================================================
# 1. CONFIGURACION (PARTICION CRONOLOGICA POR BLOQUES)
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

META_COLS = ["label", "session", "image_file", "handedness"]
FEATURE_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
EXPECTED_COLS = META_COLS + FEATURE_COLS  # 67 columnas totales


def extract_timestamp_from_filename(filename):
    """Extrae el timestamp numerico desde '<sesion>_<handedness>_<timestamp>.jpg'."""
    stem = str(filename).rsplit(".", 1)[0]
    last_token = stem.rsplit("_", 1)[-1]
    return int(last_token) if last_token.isdigit() else -1


# ============================================================
# 2. CARGA Y VALIDACION INICIAL
# ============================================================

if not MLP_CLEAN_CSV.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset limpio en: {MLP_CLEAN_CSV}\n"
        "Ejecuta primero preprocess.py."
    )

MLP_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(MLP_CLEAN_CSV)

if df.empty:
    raise ValueError("El dataset limpio esta vacio.")

missing_cols = [col for col in EXPECTED_COLS if col not in df.columns]
if missing_cols:
    raise ValueError(
        f"El dataset no tiene el esquema de 63 coordenadas (67 columnas).\n"
        f"Columnas faltantes: {missing_cols[:10]} (total: {len(missing_cols)})."
    )

if not np.all(np.isfinite(df[FEATURE_COLS].to_numpy(dtype=np.float32))):
    raise ValueError("Las caracteristicas contienen valores NaN o infinitos.")

if not abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6:
    raise ValueError("Las proporciones de Train, Val y Test deben sumar 1.0.")


# ============================================================
# 3. PARTICION POR BLOQUES TEMPORALES (TIME_BLOCKS)
# ============================================================

train_parts, val_parts, test_parts = [], [], []

for (_, _), group in df.groupby(["label", "session"], sort=False):
    group = group.copy()
    group["_ts"] = group["image_file"].map(extract_timestamp_from_filename)

    # Ordenar cronologicamente para no mezclar frames consecutivos
    if (group["_ts"] >= 0).all():
        group = group.sort_values("_ts", kind="stable")

    group = group.drop(columns=["_ts"]).reset_index(drop=True)
    n = len(group)

    if n < 5:
        train_parts.append(group)
        continue

    i_train = int(n * TRAIN_RATIO)
    i_val = int(n * (TRAIN_RATIO + VAL_RATIO))

    if i_train < 1 or i_val <= i_train or i_val >= n:
        train_parts.append(group)
        continue

    train_parts.append(group.iloc[:i_train])
    val_parts.append(group.iloc[i_train:i_val])
    test_parts.append(group.iloc[i_val:])

if not train_parts or not val_parts or not test_parts:
    raise ValueError(
        "No hay suficientes muestras por clase/sesion para generar las 3 particiones."
    )

train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)


# ============================================================
# 4. VALIDACION DE COBERTURA DE CLASES
# ============================================================

all_labels = set(df["label"].unique())
for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    missing_labels = all_labels - set(part["label"].unique())
    if missing_labels:
        raise ValueError(
            f"{name} no contiene las siguientes clases: {sorted(missing_labels)}."
        )


# ============================================================
# 5. GUARDAR PARTICIONES Y REPORTE
# ============================================================

train_path = MLP_SPLITS_DIR / "train.csv"
val_path = MLP_SPLITS_DIR / "val.csv"
test_path = MLP_SPLITS_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("=== PARTICION COMPLETADA (time_blocks) ===")
print(f"Total: {len(df)} filas | Caracteristicas: {len(FEATURE_COLS)}")
print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%) -> {train_path}")
print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.2f}%) -> {val_path}")
print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.2f}%) -> {test_path}")