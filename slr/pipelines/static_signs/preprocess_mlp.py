import numpy as np
import pandas as pd

from paths import (
    MLP_RAW_CSV,
    MLP_CLEAN_CSV,
    MLP_CLEAN_CSV_REPORT,
)

# ============================================================
# 1. CONFIGURACION Y ESQUEMA (63 CARACTERISTICAS)
# ============================================================

META_COLS = ["label", "session", "image_file", "handedness"]
FEATURE_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
EXPECTED_COLS = META_COLS + FEATURE_COLS

if not MLP_RAW_CSV.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset crudo en: {MLP_RAW_CSV}\n"
        "Ejecuta primero collect.py para capturar los datos."
    )

MLP_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
MLP_CLEAN_CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATA ACCESS (COMPATIBLE CON BACKUP DE 65 COLS Y NUEVO DE 67 COLS)
# ============================================================

# Leer primera fila para detectar si tiene encabezado
first_line = MLP_RAW_CSV.read_text(encoding="utf-8").splitlines()[0].strip()
has_header = first_line.lower().startswith("label,")

if has_header:
    df = pd.read_csv(MLP_RAW_CSV)
else:
    # Formato backup antiguo: 65 columnas sin encabezado (label, session + 63 coords)
    df = pd.read_csv(MLP_RAW_CSV, header=None)
    if df.shape[1] == 65:
        df.columns = ["label", "session"] + FEATURE_COLS
        # Generar un indice secuencial como timestamp para mantener el orden en split.py
        df["image_file"] = [f"backup_{i}.jpg" for i in range(len(df))]
        df["handedness"] = "Unknown"
    else:
        raise ValueError(
            f"Formato inesperado sin encabezado: {df.shape[1]} columnas (se esperaban 65)."
        )

missing_cols = [col for col in EXPECTED_COLS if col not in df.columns]
if missing_cols:
    raise ValueError(
        f"El archivo {MLP_RAW_CSV} no tiene el esquema esperado.\n"
        f"Columnas faltantes: {missing_cols[:10]} (total: {len(missing_cols)})."
    )

df = df[EXPECTED_COLS].copy()
initial_rows = len(df)

# ============================================================
# 3. DATA CLEANING
# ============================================================

# a) Limpiar metadatos (etiquetas y sesiones)
before_meta = len(df)
df["label"] = df["label"].astype(str).str.strip().str.upper()
df["session"] = df["session"].astype(str).str.strip()
df["handedness"] = df["handedness"].astype(str).str.strip()

df = df[
    (df["label"] != "")
    & (df["label"] != "NAN")
    & (df["session"] != "")
    & (df["session"] != "NAN")
].copy()
removed_meta = before_meta - len(df)

# b) Asegurar tipos numericos y eliminar nulos o infinitos
before_na = len(df)
df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df.dropna(subset=FEATURE_COLS, inplace=True)
removed_na = before_na - len(df)

# c) Eliminar filas duplicadas exactas dentro de la misma clase y sesion
before_dup = len(df)
df.drop_duplicates(subset=["label", "session"] + FEATURE_COLS, inplace=True)
removed_dup = before_dup - len(df)

# d) Filtrar manos colapsadas o lecturas corruptas en ceros
before_collapsed = len(df)
coords_matrix = df[FEATURE_COLS].to_numpy(dtype=np.float32).reshape(-1, 21, 3)
shifted = coords_matrix - coords_matrix[:, 0:1, :]  # Origen en la muneca (0)
scales = np.max(np.linalg.norm(shifted, axis=2), axis=1)

valid_mask = scales > 1e-4
df = df.loc[valid_mask].copy()
shifted = shifted[valid_mask]
scales = scales[valid_mask]
removed_collapsed = before_collapsed - len(df)

if df.empty:
    raise ValueError("No quedaron registros validos despues de la limpieza.")


# ============================================================
# 4. NORMALIZACION GEOMETRICA VECTORIZADA
# ============================================================

# Dividir cada muestra por su escala maxima (invariante a posicion y distancia)
normalized_coords = shifted / scales[:, np.newaxis, np.newaxis]
df[FEATURE_COLS] = normalized_coords.reshape(-1, 63).astype(np.float32)
df.reset_index(drop=True, inplace=True)


# ============================================================
# 5. AGGREGATION / REPORTE
# ============================================================

class_counts = df.groupby("label").size()
session_counts = df.groupby(["label", "session"]).size()
handedness_counts = df.groupby("handedness").size()

report_lines = [
    "=== REPORTE DE LIMPIEZA Y PREPROCESAMIENTO ALIS (63 COORDS) ===",
    f"Archivo origen:               {MLP_RAW_CSV}",
    f"Filas iniciales:              {initial_rows}",
    f"Eliminadas por metadatos:     {removed_meta}",
    f"Eliminadas por nulos/Inf:     {removed_na}",
    f"Eliminadas por duplicados:    {removed_dup}",
    f"Eliminadas manos colapsadas:  {removed_collapsed}",
    f"Filas limpias finales:        {len(df)}",
    "",
    "--- Distribucion por letra ---",
    class_counts.to_string(),
    "",
    "--- Distribucion por letra y sesion ---",
    session_counts.to_string(),
    "",
    "--- Lateralidad detectada ---",
    handedness_counts.to_string(),
    "",
    f"Minimo por clase: {class_counts.min()} | Maximo por clase: {class_counts.max()}",
]
report_text = "\n".join(report_lines)


# ============================================================
# 6. GUARDAR RESULTADOS
# ============================================================

df.to_csv(MLP_CLEAN_CSV, index=False)
MLP_CLEAN_CSV_REPORT.write_text(report_text, encoding="utf-8")

print(report_text)
print(f"\nDataset limpio guardado en: {MLP_CLEAN_CSV}")
print(f"Reporte guardado en: {MLP_CLEAN_CSV_REPORT}")