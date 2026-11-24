from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

from paths import MLP_LABELS_PATH, MLP_MODEL_PATH

# ============================================================
# 1. CONFIGURACION DE INFERENCIA
# ============================================================

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 5
OFFSET = 20
SHOW_SKELETON = True  # Cambia a False si prefieres la imagen sin esqueleto

for path in (MLP_MODEL_PATH, MLP_LABELS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Falta el archivo requerido: {path}\n"
            "Ejecuta primero train.py para generar el modelo y las etiquetas."
        )

model = tf.keras.models.load_model(str(MLP_MODEL_PATH))
labels = [
    line.strip()
    for line in MLP_LABELS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def normalize_landmarks(lm_list):
    """
    Convierte los 21 puntos [x, y, z] en un vector de 63 valores
    relativos a la muneca (punto 0) y escalados por la distancia maxima.
    """
    pts = np.array(lm_list, dtype=np.float32)

    if pts.shape == (21, 2):
        pts = np.hstack([pts, np.zeros((21, 1), dtype=np.float32)])
    elif pts.shape != (21, 3):
        return None

    pts = pts - pts[0]  # Origen en la muneca
    max_dist = float(np.max(np.linalg.norm(pts, axis=1)))
    if max_dist < 1e-4:
        return None

    pts = pts / max_dist
    return pts.flatten().reshape(1, 63)


# ============================================================
# 2. BUCLE DE PREDICCION EN TIEMPO REAL
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara web.")

detector = HandDetector(maxHands=1, detectionCon=0.7)
prob_buffer = deque(maxlen=SMOOTHING_WINDOW)

print("ALIS en vivo iniciado. Presiona 'q' para salir.")

while True:
    success, img = cap.read()
    if not success:
        print("ERROR: No se pudo leer fotograma de la camara.")
        break

    hands, img = detector.findHands(img, draw=SHOW_SKELETON)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        lm_list = hand["lmList"]

        features = normalize_landmarks(lm_list)
        if features is not None:
            # Inferencia directa sin sobrecarga de model.predict()
            probs = model(features, training=False).numpy()[0]
            prob_buffer.append(probs)

            # Suavizado temporal sobre los ultimos fotogramas
            avg_probs = np.mean(prob_buffer, axis=0)
            idx = int(np.argmax(avg_probs))
            confidence = float(avg_probs[idx])

            if confidence >= CONFIDENCE_THRESHOLD:
                text = f"{labels[idx]} ({confidence * 100:.0f}%)"
                box_color = (0, 200, 0)  # Verde si supera el umbral
            else:
                text = "..."
                box_color = (0, 165, 255)  # Naranja si duda

            y_top = max(0, y - OFFSET - 50)
            x_left = max(0, x - OFFSET)
            cv2.rectangle(
                img, (x_left, y_top), (x_left + 190, y_top + 45), box_color, cv2.FILLED
            )
            cv2.putText(
                img,
                text,
                (x_left + 10, y_top + 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                img,
                (x_left, max(0, y - OFFSET)),
                (x + w + OFFSET, y + h + OFFSET),
                box_color,
                3,
            )
    else:
        prob_buffer.clear()

    cv2.imshow("ALIS - Reconocimiento LSB (Presiona 'q' para salir)", img)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()