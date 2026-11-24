import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from paths import (
    LSTM_TEST_NPZ,
    LSTM_MODEL_PATH,
    LSTM_LABELS_PATH,
    LSTM_TEST_REPORT,
)


# ============================================================
# CONFIGURACION Y ESQUEMA ESPERADO
# ============================================================

EXPECTED_SEQ_LENGTH = 30
STATIC_FEATURES = 227
EXPECTED_FEATURES = STATIC_FEATURES * 2  # 454

# Umbral de referencia para simular filtrado en tiempo real
CONFIDENCE_THRESHOLD = 0.75

FACE_EXPR_SWAP_PAIRS = (
    (0, 3), (1, 4), (2, 5),
    (6, 8), (7, 9),
    (10, 11),
    (16, 17),
)

BODY_SWAP_PAIRS = (
    (0, 1),
    (2, 3),
    (4, 5),
)


# ============================================================
# DIRECTORIOS
# ============================================================

LSTM_TEST_REPORT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# REFLEXION HORIZONTAL PARA EVALUACION AMBIDIESTRA
# ============================================================

def recompute_velocity_seq(static_seq):
    vel = np.zeros_like(static_seq, dtype=np.float32)
    vel[1:] = static_seq[1:] - static_seq[:-1]

    curr_pres = static_seq[1:, 223:227] > 0.5
    prev_pres = static_seq[:-1, 223:227] > 0.5

    both_pose = curr_pres[:, 0] & prev_pres[:, 0]
    both_face = curr_pres[:, 1] & prev_pres[:, 1]
    both_left = curr_pres[:, 2] & prev_pres[:, 2]
    both_right = curr_pres[:, 3] & prev_pres[:, 3]
    both_hands = both_left & both_right

    vel[1:, 0:57][~both_face] = 0.0
    vel[1:, 57:81][~both_pose] = 0.0
    vel[1:, 81:150][~both_left] = 0.0
    vel[1:, 150:219][~both_right] = 0.0
    vel[1:, 219:223][~both_hands] = 0.0
    vel[:, 223:227] = 0.0
    return vel


def mirror_sequence(sequence):
    """Genera la secuencia equivalente ejecutada con la mano opuesta."""
    seq_len = sequence.shape[0]
    m = sequence[:, :STATIC_FEATURES].copy()

    # 1. Cara
    m[:, 0] = -m[:, 0]
    face_expr = m[:, 3:57].reshape(seq_len, 18, 3)
    for l_idx, r_idx in FACE_EXPR_SWAP_PAIRS:
        face_expr[:, [l_idx, r_idx]] = face_expr[:, [r_idx, l_idx]]
    face_expr[:, :, 0] = -face_expr[:, :, 0]
    m[:, 3:57] = face_expr.reshape(seq_len, 54)

    # 2. Pose superior
    body = m[:, 57:81].reshape(seq_len, 6, 4)
    for l_idx, r_idx in BODY_SWAP_PAIRS:
        body[:, [l_idx, r_idx]] = body[:, [r_idx, l_idx]]
    body[:, :, 0] = -body[:, :, 0]
    m[:, 57:81] = body.reshape(seq_len, 24)

    # 3. Manos (izquierda <-> derecha)
    left_hand = m[:, 81:150].copy()
    right_hand = m[:, 150:219].copy()
    left_hand[:, 0::3] = -left_hand[:, 0::3]
    right_hand[:, 0::3] = -right_hand[:, 0::3]
    m[:, 81:150] = right_hand
    m[:, 150:219] = left_hand

    # 4. Relacion inter-manos
    m[:, 220] = -m[:, 220]
    m[:, 221] = -m[:, 221]

    # 5. Flags de presencia
    m[:, [225, 226]] = m[:, [226, 225]]

    # 6. Velocidad temporal
    vel_m = recompute_velocity_seq(m)
    return np.concatenate([m, vel_m], axis=1).astype(np.float32)


# ============================================================
# CARGA Y VALIDACION DE DATOS
# ============================================================

if not LSTM_TEST_NPZ.exists():
    raise FileNotFoundError(f"No se encontro el conjunto de prueba: {LSTM_TEST_NPZ}")

if not LSTM_LABELS_PATH.exists():
    raise FileNotFoundError(f"No se encontro el archivo de etiquetas: {LSTM_LABELS_PATH}")

if not LSTM_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontro el modelo entrenado: {LSTM_MODEL_PATH}")

test_data = np.load(LSTM_TEST_NPZ, allow_pickle=False)

X_test = test_data["x"].astype(np.float32)
Y_true = test_data["y"].astype(np.int32)
sessions = test_data["sessions"].tolist() if "sessions" in test_data else None

actions = [
    line.strip()
    for line in LSTM_LABELS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

num_classes = len(actions)

if len(X_test) == 0:
    raise ValueError("El conjunto de prueba esta vacio.")

if X_test.ndim != 3:
    raise ValueError(
        "X_test debe tener tres dimensiones: (secuencias, frames, caracteristicas)."
    )

if X_test.shape[1:] != (EXPECTED_SEQ_LENGTH, EXPECTED_FEATURES):
    raise ValueError(
        f"Forma inesperada en X_test: {X_test.shape[1:]}. "
        f"Se esperaba ({EXPECTED_SEQ_LENGTH}, {EXPECTED_FEATURES}). "
        "Regenera las particiones con las secuencias de 454 caracteristicas."
    )

if "labels" in test_data:
    npz_labels = test_data["labels"].tolist()
    if npz_labels != actions:
        raise ValueError(
            "Las etiquetas de LSTM_TEST_NPZ no coinciden con LSTM_LABELS_PATH.\n"
            f"  NPZ:    {npz_labels}\n"
            f"  Modelo: {actions}"
        )

if len(Y_true) != len(X_test):
    raise ValueError("La cantidad de etiquetas no coincide con las muestras.")

if not np.all(np.isfinite(X_test)):
    raise ValueError("El conjunto de prueba contiene valores NaN o infinitos.")

if np.any(Y_true < 0) or np.any(Y_true >= num_classes):
    raise ValueError("Se encontraron etiquetas fuera del rango de clases.")


# ============================================================
# CARGA DEL MODELO
# ============================================================

model = tf.keras.models.load_model(LSTM_MODEL_PATH)

if model.output_shape[-1] != num_classes:
    raise ValueError(
        f"La cantidad de salidas del modelo ({model.output_shape[-1]}) "
        f"no coincide con el numero de clases ({num_classes})."
    )

if tuple(model.input_shape[1:]) != (EXPECTED_SEQ_LENGTH, EXPECTED_FEATURES):
    raise ValueError(
        f"La entrada del modelo {tuple(model.input_shape[1:])} no coincide "
        f"con el esquema esperado ({EXPECTED_SEQ_LENGTH}, {EXPECTED_FEATURES})."
    )

if tuple(model.input_shape[1:]) != tuple(X_test.shape[1:]):
    raise ValueError(
        "Las dimensiones de entrada del modelo y del conjunto de prueba no coinciden."
    )


# ============================================================
# EVALUACION ORIGINAL Y AMBIDIESTRA (MANO OPUESTA)
# ============================================================

def evaluate_split(X_eval, y_eval):
    y_cat = tf.keras.utils.to_categorical(y_eval, num_classes=num_classes)
    loss_val, cat_acc_val = model.evaluate(X_eval, y_cat, verbose=0)

    probs_val = model.predict(X_eval, verbose=0)
    y_pred_val = np.argmax(probs_val, axis=1)
    confs_val = np.max(probs_val, axis=1)

    acc_val = accuracy_score(y_eval, y_pred_val)
    macro_f1_val = f1_score(y_eval, y_pred_val, average="macro", zero_division=0)
    weighted_f1_val = f1_score(y_eval, y_pred_val, average="weighted", zero_division=0)

    correct_m = y_pred_val == y_eval
    incorrect_m = ~correct_m

    mean_all = float(np.mean(confs_val))
    mean_ok = float(np.mean(confs_val[correct_m])) if np.any(correct_m) else 0.0
    mean_err = float(np.mean(confs_val[incorrect_m])) if np.any(incorrect_m) else 0.0

    acc_mask = confs_val >= CONFIDENCE_THRESHOLD
    cov_val = float(np.mean(acc_mask))
    thresh_acc_val = (
        float(accuracy_score(y_eval[acc_mask], y_pred_val[acc_mask]))
        if np.any(acc_mask)
        else 0.0
    )

    return {
        "loss": float(loss_val),
        "cat_acc": float(cat_acc_val),
        "acc": float(acc_val),
        "macro_f1": float(macro_f1_val),
        "weighted_f1": float(weighted_f1_val),
        "probs": probs_val,
        "y_pred": y_pred_val,
        "confs": confs_val,
        "incorrect_mask": incorrect_m,
        "mean_conf_all": mean_all,
        "mean_conf_ok": mean_ok,
        "mean_conf_err": mean_err,
        "coverage": cov_val,
        "thresholded_acc": thresh_acc_val,
    }


X_test_mirrored = np.stack([mirror_sequence(seq) for seq in X_test], axis=0)
X_test_both = np.concatenate([X_test, X_test_mirrored], axis=0)
Y_true_both = np.concatenate([Y_true, Y_true], axis=0)

res_orig = evaluate_split(X_test, Y_true)
res_mirror = evaluate_split(X_test_mirrored, Y_true)
res_both = evaluate_split(X_test_both, Y_true_both)


# ============================================================
# METRICAS POR CLASE Y MATRIZ DE CONFUSION (CONJUNTO COMBINADO)
# ============================================================

cls_report = classification_report(
    Y_true_both,
    res_both["y_pred"],
    labels=np.arange(num_classes),
    target_names=actions,
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(
    Y_true_both,
    res_both["y_pred"],
    labels=np.arange(num_classes),
)


def format_confusion_matrix(matrix, labels):
    """Formatea la matriz de confusion con nombres de filas y columnas."""
    col_width = max(max(len(lbl) for lbl in labels), 6) + 2
    header = " " * col_width + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    rows = [header]
    for idx, lbl in enumerate(labels):
        row_vals = "".join(f"{val:>{col_width}d}" for val in matrix[idx])
        rows.append(f"{lbl:<{col_width}}{row_vals}")
    return "\n".join(rows)


cm_formatted = format_confusion_matrix(cm, actions)


# ============================================================
# DESGLOSE DE ERRORES EN TEST (ORIGINAL + MANO OPUESTA)
# ============================================================

error_lines = []
num_orig = len(X_test)
for idx in np.where(res_both["incorrect_mask"])[0]:
    is_mirrored = idx >= num_orig
    base_idx = idx - num_orig if is_mirrored else idx
    hand_tag = "Mano opuesta (espejo)" if is_mirrored else "Mano original"
    true_lbl = actions[Y_true_both[idx]]
    pred_lbl = actions[res_both["y_pred"][idx]]
    pred_conf = res_both["confs"][idx]
    true_prob = res_both["probs"][idx, Y_true_both[idx]]
    sess_info = f" | Sesion: {sessions[base_idx]}" if sessions is not None else ""
    error_lines.append(
        f"  - Muestra #{base_idx:03d} [{hand_tag}]{sess_info} | Real: {true_lbl} -> "
        f"Predicho: {pred_lbl} (conf: {pred_conf:.4f}, prob_real: {true_prob:.4f})"
    )

errors_section = (
    "\n".join(error_lines)
    if error_lines
    else "  Ningun error: 100% de aciertos tanto en mano original como en mano opuesta."
)


# ============================================================
# REPORTE DE EVALUACION
# ============================================================

report = (
    "=== REPORTE DE EVALUACION ALIS (LSTM AMBIDIESTRO) ===\n\n"
    f"Clases ({num_classes}): {actions}\n"
    f"Muestras Test (original / espejo / total): {len(X_test)} / {len(X_test_mirrored)} / {len(X_test_both)}\n"
    f"Forma de entrada: {X_test.shape[1:]}\n\n"
    "--- Desglose por lateralidad ---\n"
    f"Mano original -> Accuracy: {res_orig['acc']:.4f} | Loss: {res_orig['loss']:.4f} | Macro F1: {res_orig['macro_f1']:.4f}\n"
    f"Mano opuesta  -> Accuracy: {res_mirror['acc']:.4f} | Loss: {res_mirror['loss']:.4f} | Macro F1: {res_mirror['macro_f1']:.4f}\n\n"
    f"--- Metricas globales (Ambas manos: {len(X_test_both)} muestras) ---\n"
    f"Test Loss:                 {res_both['loss']:.4f}\n"
    f"Test Accuracy:             {res_both['acc']:.4f}\n"
    f"Test Categorical Accuracy: {res_both['cat_acc']:.4f}\n"
    f"Test Macro F1:             {res_both['macro_f1']:.4f}\n"
    f"Test Weighted F1:          {res_both['weighted_f1']:.4f}\n\n"
    "--- Analisis de confianza (Ambas manos) ---\n"
    f"Confianza media global:    {res_both['mean_conf_all']:.4f}\n"
    f"Confianza media aciertos:  {res_both['mean_conf_ok']:.4f}\n"
    f"Confianza media errores:   {res_both['mean_conf_err']:.4f}\n"
    f"Con umbral >= {CONFIDENCE_THRESHOLD:.2f}:        "
    f"Accuracy = {res_both['thresholded_acc']:.4f} (cobertura: {res_both['coverage']:.1%})\n\n"
    "--- Metricas por clase (Ambas manos) ---\n"
    f"{cls_report}\n"
    "--- Matriz de confusion (Filas: Real | Columnas: Predicho) ---\n"
    f"{cm_formatted}\n\n"
    "--- Detalle de muestras fallidas ---\n"
    f"{errors_section}\n"
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + report)

LSTM_TEST_REPORT.write_text(
    report,
    encoding="utf-8",
)

print(f"Reporte guardado en: {LSTM_TEST_REPORT}")