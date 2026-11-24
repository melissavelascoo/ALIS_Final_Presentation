import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers

from paths import (
    MLP_LABELS_PATH,
    MLP_MODEL_PATH,
    MLP_TRAIN_CSV,
    MLP_TRAINING_CURVES,
    MLP_TRAINING_REPORT,
    MLP_VAL_CSV,
)

# ============================================================
# 1. CONFIGURACION E HIPERPARAMETROS
# ============================================================

SEED = 42
tf.keras.utils.set_random_seed(SEED)

EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4

FEATURE_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
MLP_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
MLP_TRAINING_REPORT.parent.mkdir(parents=True, exist_ok=True)
MLP_TRAINING_CURVES.parent.mkdir(parents=True, exist_ok=True)

for path in (MLP_TRAIN_CSV, MLP_VAL_CSV):
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}.\nEjecuta primero preprocess.py y split.py."
        )


# ============================================================
# 2. CARGA DE PARTICIONES Y DATA AUGMENTATION (SOLO TRAIN)
# ============================================================

train_df = pd.read_csv(MLP_TRAIN_CSV)
val_df = pd.read_csv(MLP_VAL_CSV)

classes = sorted(train_df["label"].unique().tolist())
num_classes = len(classes)

if num_classes < 2:
    raise ValueError(
        f"Solo se encontro {num_classes} clase ({classes}). "
        "Necesitas al menos 2 letras distintas para entrenar."
    )

label_to_idx = {label: idx for idx, label in enumerate(classes)}

unknown_val = set(val_df["label"].unique()) - set(classes)
if unknown_val:
    raise ValueError(
        f"Validation contiene clases no presentes en Train: {sorted(unknown_val)}."
    )

x_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
y_train = tf.keras.utils.to_categorical(
    train_df["label"].map(label_to_idx).to_numpy(), num_classes
)

# Data Augmentation: espejo horizontal sobre eje X (origen en muñeca)
x_train_mirror = x_train.copy().reshape(-1, 21, 3)
x_train_mirror[:, :, 0] *= -1.0
x_train_mirror = x_train_mirror.reshape(-1, 63)

x_train = np.concatenate([x_train, x_train_mirror], axis=0)
y_train = np.concatenate([y_train, y_train], axis=0)

# Barajar explícitamente todo el conjunto de entrenamiento (original + espejo)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(x_train))
x_train = x_train[perm]
y_train = y_train[perm]

x_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
y_val = tf.keras.utils.to_categorical(
    val_df["label"].map(label_to_idx).to_numpy(), num_classes
)

print(f"Classes ({num_classes}): {classes}")
print(f"Training (con espejo): {len(x_train)} | Validation: {len(x_val)}")


# ============================================================
# 3. ARQUITECTURA MLP MULTICLASE
# ============================================================

model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2_LAMBDA)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2_LAMBDA)),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# ============================================================
# 4. CALLBACKS Y ENTRENAMIENTO
# ============================================================

cb_early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1,
)

cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-5,
    verbose=1,
)

cb_checkpoint = callbacks.ModelCheckpoint(
    filepath=str(MLP_MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb_early_stop, cb_reduce_lr, cb_checkpoint],
    verbose=1,
)

# Guardar etiquetas sincronizadas con los indices del modelo
MLP_LABELS_PATH.write_text("\n".join(classes) + "\n", encoding="utf-8")


# ============================================================
# 5. REPORTE Y CURVAS DE APRENDIZAJE
# ============================================================

val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
best_epoch = int(np.argmin(history.history["val_loss"])) + 1

report_lines = [
    "=== REPORTE DE ENTRENAMIENTO Y VALIDACION DE ALIS (MLP) ===",
    f"Clases ({num_classes}): {classes}",
    f"Muestras -> Train (con espejo): {len(x_train)} | Val: {len(x_val)}",
    f"Epocas ejecutadas: {len(history.history['loss'])} (mejor epoca: {best_epoch})",
    f"Mejor Val Loss: {val_loss:.4f}",
    f"Mejor Val Accuracy: {val_acc:.4f}",
    f"Modelo Guardado: {MLP_MODEL_PATH}",
    f"Etiquetas: {MLP_LABELS_PATH}",
]

report_text = "\n".join(report_lines)
MLP_TRAINING_REPORT.write_text(report_text, encoding="utf-8")
print("\n" + report_text)

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Perdida por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Precision por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(MLP_TRAINING_CURVES, dpi=150)
    plt.close()
    print(f"Grafica guardada en: {MLP_TRAINING_CURVES}")
except ImportError:
    print("NOTA: matplotlib no esta instalado; se omitio training_curves.png")