import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from paths import (
    MLP_LABELS_PATH,
    MLP_MODEL_PATH,
    MLP_TEST_CSV,
    MLP_TEST_REPORT,
)

for path in (MLP_TEST_CSV, MLP_MODEL_PATH, MLP_LABELS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Falta el archivo requerido: {path}\n"
            "Asegurate de haber ejecutado split.py y train.py."
        )

MLP_TEST_REPORT.parent.mkdir(parents=True, exist_ok=True)

# 1) Cargar modelo, etiquetas y conjunto de testing
model = tf.keras.models.load_model(str(MLP_MODEL_PATH))
classes = [
    line.strip()
    for line in MLP_LABELS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
num_classes = len(classes)
label_to_idx = {label: idx for idx, label in enumerate(classes)}

test_df = pd.read_csv(MLP_TEST_CSV)
if test_df.empty:
    raise ValueError("test.csv esta vacio.")

feature_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

# Verificar que todas las etiquetas de test existan en el modelo
unknown_labels = set(test_df["label"].unique()) - set(classes)
if unknown_labels:
    raise ValueError(
        f"test.csv contiene clases no vistas al entrenar: {sorted(unknown_labels)}"
    )

x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
y_test_idx = test_df["label"].map(label_to_idx).to_numpy()
y_test_cat = tf.keras.utils.to_categorical(y_test_idx, num_classes)

# 2) Evaluacion Final
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
y_pred_probs = model.predict(x_test, verbose=0)
y_pred_idx = np.argmax(y_pred_probs, axis=1)

all_indices = list(range(num_classes))
macro_f1 = f1_score(y_test_idx, y_pred_idx, average="macro", zero_division=0)
cls_report = classification_report(
    y_test_idx,
    y_pred_idx,
    labels=all_indices,
    target_names=classes,
    digits=4,
    zero_division=0,
)
cm = confusion_matrix(y_test_idx, y_pred_idx, labels=all_indices)

# 3) Reporte final
report_lines = [
    "=== REPORTE DE EVALUACION FINAL ALIS (MLP) ===",
    "",
    f"Modelo evaluado: {MLP_MODEL_PATH.name}",
    f"Muestras Test:   {len(x_test)}",
    f"Test Loss:       {test_loss:.4f}",
    f"Test Accuracy:   {test_acc:.4f}",
    f"Test Macro F1:   {macro_f1:.4f}",
    "",
    "--- Metricas por Letra ---",
    cls_report,
    "--- Matriz de Confusion (filas=real, cols=predicho) ---",
    pd.DataFrame(cm, index=classes, columns=classes).to_string(),
]

report_text = "\n".join(report_lines)
MLP_TEST_REPORT.write_text(report_text, encoding="utf-8")

print(report_text)
print(f"\nReporte de Testing guardado en: {MLP_TEST_REPORT}")