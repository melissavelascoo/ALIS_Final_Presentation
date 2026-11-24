import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    LSTM,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from paths import (
    LSTM_TRAIN_NPZ,
    LSTM_VAL_NPZ,
    LSTM_MODEL_PATH,
    LSTM_LABELS_PATH,
    LSTM_TRAINING_REPORT,
    LSTM_TRAINING_CURVES,
)


# ============================================================
# HIPERPARAMETROS Y ESQUEMA ESPERADO
# ============================================================

EXPECTED_SEQ_LENGTH = 30
STATIC_FEATURES = 227
EXPECTED_FEATURES = STATIC_FEATURES * 2  # 454

SEED = 42
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Duplica cada secuencia de entrenamiento reflejando izquierda <-> derecha
MIRROR_AUGMENT = True

# Copias cinematicas sinteticas por cada secuencia base (0 para desactivar)
AUGMENT_COPIES_PER_SAMPLE = 2

# Pares simetricos en los 18 puntos de expresion facial de extract_videos.py
FACE_EXPR_SWAP_PAIRS = (
    (0, 3), (1, 4), (2, 5),   # Cejas izquierda <-> derecha
    (6, 8), (7, 9),           # Ojos izquierdo <-> derecho
    (10, 11),                 # Comisuras externas de boca
    (16, 17),                 # Comisuras internas de boca
)

# Pares simetricos en los 6 puntos de pose (11, 12, 13, 14, 15, 16)
BODY_SWAP_PAIRS = (
    (0, 1),  # Hombros
    (2, 3),  # Codos
    (4, 5),  # Munecas
)

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# DIRECTORIOS
# ============================================================

LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
LSTM_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
LSTM_TRAINING_REPORT.parent.mkdir(parents=True, exist_ok=True)
LSTM_TRAINING_CURVES.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUMENTACION DE DATOS: ESPEJO AMBIDIESTRO + CINEMATICA
# ============================================================

def recompute_velocity_seq(static_seq):
    """
    Recalcula las 227 caracteristicas de velocidad a partir de las
    227 caracteristicas estaticas, respetando las mascaras de presencia.
    """
    vel = np.zeros_like(static_seq, dtype=np.float32)
    vel[1:] = static_seq[1:] - static_seq[:-1]

    # Indicadores de presencia en static_seq:
    # [223]: pose, [224]: cara, [225]: mano izquierda, [226]: mano derecha
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

    # No derivar flags binarios de presencia
    vel[:, 223:227] = 0.0
    return vel


def mirror_sequence(sequence):
    """
    Genera la version reflejada horizontalmente (zurda <-> diestra)
    de una secuencia (30, 454) para entrenamiento ambidiestro.
    """
    seq_len = sequence.shape[0]
    m = sequence[:, :STATIC_FEATURES].copy()

    # 1. Cara (0..57): reflejar X en posicion de cabeza y puntos de expresion
    m[:, 0] = -m[:, 0]
    face_expr = m[:, 3:57].reshape(seq_len, 18, 3)
    for l_idx, r_idx in FACE_EXPR_SWAP_PAIRS:
        face_expr[:, [l_idx, r_idx]] = face_expr[:, [r_idx, l_idx]]
    face_expr[:, :, 0] = -face_expr[:, :, 0]
    m[:, 3:57] = face_expr.reshape(seq_len, 54)

    # 2. Pose superior (57..81): intercambiar izquierda/derecha y reflejar X
    body = m[:, 57:81].reshape(seq_len, 6, 4)
    for l_idx, r_idx in BODY_SWAP_PAIRS:
        body[:, [l_idx, r_idx]] = body[:, [r_idx, l_idx]]
    body[:, :, 0] = -body[:, :, 0]
    m[:, 57:81] = body.reshape(seq_len, 24)

    # 3. Manos (81..150 izquierda, 150..219 derecha): reflejar X e intercambiar bloques
    left_hand = m[:, 81:150].copy()
    right_hand = m[:, 150:219].copy()
    left_hand[:, 0::3] = -left_hand[:, 0::3]
    right_hand[:, 0::3] = -right_hand[:, 0::3]
    m[:, 81:150] = right_hand
    m[:, 150:219] = left_hand

    # 4. Relacion inter-manos (219..223): [dx, dy, dz, dist] -> [+dx, -dy, -dz, +dist]
    m[:, 220] = -m[:, 220]
    m[:, 221] = -m[:, 221]

    # 5. Flags de presencia (223..227): intercambiar presencia mano izquierda <-> derecha
    m[:, [225, 226]] = m[:, [226, 225]]

    # 6. Recalcular velocidad temporal sobre la secuencia reflejada
    vel_m = recompute_velocity_seq(m)
    return np.concatenate([m, vel_m], axis=1).astype(np.float32)


def augment_sequence(sequence, rng):
    """
    Genera una variacion realista de una secuencia (30, 454):
    1. Variacion de velocidad de ejecucion (0.85x a 1.15x) y desplazamiento temporal.
    2. Ligera variacion de escala espacial y ruido gaussiano en articulaciones activas
       (preservando intactas las columnas de visibilidad corporal en [0, 1]).
    3. Recalculo consistente de la velocidad temporal.
    """
    seq_len = sequence.shape[0]
    static_orig = sequence[:, :STATIC_FEATURES]

    speed = rng.uniform(0.85, 1.15)
    shift = rng.uniform(-2.0, 2.0)

    center = (seq_len - 1) / 2.0
    orig_idx = np.arange(seq_len, dtype=np.float32)
    src_idx = np.clip((orig_idx - center) * speed + center + shift, 0.0, seq_len - 1.0)

    new_static = np.zeros_like(static_orig, dtype=np.float32)

    # Interpolar coordenadas continuas (0..222)
    for col in range(223):
        new_static[:, col] = np.interp(src_idx, orig_idx, static_orig[:, col])

    # Mantener binarios los flags de presencia (223..226)
    nearest_idx = np.rint(src_idx).astype(np.int32)
    new_static[:, 223:227] = static_orig[nearest_idx, 223:227]

    # Mascaras activas por bloque anatomico
    mask_pose = (new_static[:, 223:224] > 0.5).astype(np.float32)
    mask_face = (new_static[:, 224:225] > 0.5).astype(np.float32)
    mask_left = (new_static[:, 225:226] > 0.5).astype(np.float32)
    mask_right = (new_static[:, 226:227] > 0.5).astype(np.float32)
    mask_both = mask_left * mask_right

    active_mask = np.concatenate(
        [
            np.repeat(mask_face, 57, axis=1),
            np.repeat(mask_pose, 24, axis=1),
            np.repeat(mask_left, 69, axis=1),
            np.repeat(mask_right, 69, axis=1),
            np.repeat(mask_both, 4, axis=1),
        ],
        axis=1,
    )

    # Limpiar posibles residuos interpolados donde la parte no esta presente
    new_static[:, :223] *= active_mask

    # Respaldar las 6 columnas de visibilidad corporal (indices 60, 64, 68, 72, 76, 80)
    body_vis_backup = np.clip(new_static[:, 60:81:4], 0.0, 1.0).copy()

    # Perturbacion espacial leve (escala + jitter) solo en coordenadas espaciales
    scale_jitter = rng.uniform(0.95, 1.05)
    noise = rng.normal(0.0, 0.012, size=(seq_len, 223)).astype(np.float32)
    new_static[:, :223] = (new_static[:, :223] * scale_jitter + noise) * active_mask

    # Restaurar visibilidad corporal limpia dentro de [0.0, 1.0]
    new_static[:, 60:81:4] = body_vis_backup * mask_pose

    new_vel = recompute_velocity_seq(new_static)
    return np.concatenate([new_static, new_vel], axis=1).astype(np.float32)


def build_augmented_train_set(X, y_idx, mirror_augment, copies_per_sample, seed):
    rng = np.random.default_rng(seed)

    if mirror_augment:
        X_mirrored = np.stack([mirror_sequence(seq) for seq in X], axis=0)
        X_base = np.concatenate([X, X_mirrored], axis=0)
        y_base = np.concatenate([y_idx, y_idx], axis=0)
    else:
        X_base = X
        y_base = y_idx

    if copies_per_sample <= 0:
        perm = rng.permutation(len(X_base))
        return X_base[perm], y_base[perm]

    X_aug_list = [X_base]
    y_aug_list = [y_base]

    for _ in range(copies_per_sample):
        batch_aug = np.stack([augment_sequence(seq, rng) for seq in X_base], axis=0)
        X_aug_list.append(batch_aug)
        y_aug_list.append(y_base)

    X_combined = np.concatenate(X_aug_list, axis=0)
    y_combined = np.concatenate(y_aug_list, axis=0)

    perm = rng.permutation(len(X_combined))
    return X_combined[perm], y_combined[perm]


# ============================================================
# CARGA Y VALIDACION DE DATOS
# ============================================================

train_data = np.load(LSTM_TRAIN_NPZ, allow_pickle=False)
val_data = np.load(LSTM_VAL_NPZ, allow_pickle=False)

X_train = train_data["x"].astype(np.float32)
Y_train_idx = train_data["y"].astype(np.int32)

X_val = val_data["x"].astype(np.float32)
Y_val_idx = val_data["y"].astype(np.int32)

actions = train_data["labels"].tolist()
val_actions = val_data["labels"].tolist()

num_classes = len(actions)

if actions != val_actions:
    raise ValueError("Las clases de entrenamiento y validacion no coinciden.")

if X_train.ndim != 3 or X_val.ndim != 3:
    raise ValueError(
        "Los datos deben tener tres dimensiones: (secuencias, frames, caracteristicas)."
    )

if X_train.shape[1:] != (EXPECTED_SEQ_LENGTH, EXPECTED_FEATURES):
    raise ValueError(
        f"Forma inesperada en train: {X_train.shape[1:]}. "
        f"Se esperaba ({EXPECTED_SEQ_LENGTH}, {EXPECTED_FEATURES}). "
        "Ejecuta nuevamente la particion con las secuencias de 454 caracteristicas."
    )

if X_train.shape[1:] != X_val.shape[1:]:
    raise ValueError("Las dimensiones de entrenamiento y validacion no coinciden.")

if X_train.shape[0] == 0 or X_val.shape[0] == 0:
    raise ValueError("El conjunto de entrenamiento o validacion esta vacio.")

if len(Y_train_idx) != len(X_train) or len(Y_val_idx) != len(X_val):
    raise ValueError("La cantidad de etiquetas no coincide con las secuencias.")

if num_classes < 2:
    raise ValueError("Se necesitan al menos dos clases.")

if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_val)):
    raise ValueError("Los datos contienen valores NaN o infinitos.")

if (
    np.any(Y_train_idx < 0)
    or np.any(Y_train_idx >= num_classes)
    or np.any(Y_val_idx < 0)
    or np.any(Y_val_idx >= num_classes)
):
    raise ValueError("Se encontraron etiquetas fuera del rango de clases.")

# Aplicar espejo ambidiestro + aumentacion cinematica unicamente sobre Train
X_train_aug, Y_train_aug_idx = build_augmented_train_set(
    X_train,
    Y_train_idx,
    mirror_augment=MIRROR_AUGMENT,
    copies_per_sample=AUGMENT_COPIES_PER_SAMPLE,
    seed=SEED,
)

# Calcular pesos de clase para compensar ligeros desbalanceos
class_counts = np.bincount(Y_train_aug_idx, minlength=num_classes)
total_samples = len(Y_train_aug_idx)
class_weight = {
    cls_idx: float(total_samples / (num_classes * max(1, count)))
    for cls_idx, count in enumerate(class_counts)
}

print("\nDATOS DE ENTRENAMIENTO")
print(f"Clases ({num_classes}): {actions}")
print(f"Train original:   {X_train.shape}")
print(f"Train aumentado:  {X_train_aug.shape} (Espejo ambidiestro: {MIRROR_AUGMENT})")
print(f"Val (sin tocar):  {X_val.shape}")
print(f"Caracteristicas por frame: {X_train.shape[2]}")


# ============================================================
# CODIFICACION DE ETIQUETAS
# ============================================================

Y_train = to_categorical(Y_train_aug_idx, num_classes=num_classes).astype(np.float32)
Y_val = to_categorical(Y_val_idx, num_classes=num_classes).astype(np.float32)


# ============================================================
# ARQUITECTURA LSTM REGULARIZADA
# ============================================================

model = Sequential([
    Input(shape=(EXPECTED_SEQ_LENGTH, EXPECTED_FEATURES)),

    # 1. Proyeccion espacial: comprime 454 features -> 128 por frame
    SpatialDropout1D(0.15),
    Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
    LayerNormalization(),
    Dropout(0.3),

    # 2. Modelado temporal bidireccional
    Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))
    ),
    Dropout(0.35),

    Bidirectional(
        LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4))
    ),
    Dropout(0.35),

    # 3. Clasificador final
    Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
    Dropout(0.25),
    Dense(num_classes, activation="softmax"),
])


# ============================================================
# COMPILACION
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["categorical_accuracy"],
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

log_dir = LSTM_MODEL_PATH.parent / "logs"

callbacks = [
    ModelCheckpoint(
        filepath=str(LSTM_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=1,
    ),
    TensorBoard(
        log_dir=str(log_dir)
    ),
]


# ============================================================
# ENTRENAMIENTO
# ============================================================

history = model.fit(
    X_train_aug,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# GUARDAR ETIQUETAS (EL MEJOR MODELO YA FUE GUARDADO POR CHECKPOINT)
# ============================================================

if not LSTM_MODEL_PATH.exists():
    model.save(LSTM_MODEL_PATH)

LSTM_LABELS_PATH.write_text(
    "\n".join(actions) + "\n",
    encoding="utf-8",
)


# ============================================================
# REPORTE DE ENTRENAMIENTO
# ============================================================

best_epoch = int(np.argmin(history.history["val_loss"])) + 1
best_val_loss = float(np.min(history.history["val_loss"]))
best_val_acc = float(history.history["val_categorical_accuracy"][best_epoch - 1])
best_train_acc = float(history.history["categorical_accuracy"][best_epoch - 1])

report = (
    "=== REPORTE DE ENTRENAMIENTO ALIS (LSTM) ===\n"
    f"Clases ({num_classes}): {actions}\n"
    f"Forma de entrada: {X_train.shape[1:]}\n"
    f"Muestras Train (original / aumentado): {len(X_train)} / {len(X_train_aug)}\n"
    f"Muestras Val: {len(X_val)}\n"
    f"Epocas ejecutadas: {len(history.history['loss'])}\n"
    f"Mejor epoca: {best_epoch}\n"
    f"Mejor Val Loss: {best_val_loss:.4f}\n"
    f"Mejor Val Accuracy: {best_val_acc:.4f}\n"
    f"Train Accuracy (mejor epoca): {best_train_acc:.4f}\n"
)

print("\n" + report)

LSTM_TRAINING_REPORT.write_text(
    report,
    encoding="utf-8",
)


# ============================================================
# GRAFICAS DEL ENTRENAMIENTO
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["categorical_accuracy"], label="Train Accuracy")
plt.plot(history.history["val_categorical_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoca")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(LSTM_TRAINING_CURVES, dpi=150)
plt.close()

print("\nEntrenamiento finalizado.")
print(f"Modelo guardado en: {LSTM_MODEL_PATH}")
print(f"Etiquetas guardadas en: {LSTM_LABELS_PATH}")
print(f"Reporte guardado en: {LSTM_TRAINING_REPORT}")
print(f"Graficas guardadas en: {LSTM_TRAINING_CURVES}")