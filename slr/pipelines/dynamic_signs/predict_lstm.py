import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from paths import LSTM_MODEL_PATH, LSTM_LABELS_PATH


# ============================================================
# 1. CONFIGURACION
# ============================================================

SEQUENCE_LENGTH = 30
STATIC_FEATURES = 227
NUM_FEATURES = STATIC_FEATURES * 2  # 454 (227 estaticas + 227 velocidad)

THRESHOLD = 0.85
STABILITY_WINDOW = 8
MIN_CONSISTENT_FRAMES = 6

# Pausa tras confirmar una sena para ignorar el movimiento de retorno a reposo
COOLDOWN_SECONDS = 1.2

# Movimiento minimo normalizado (promedio de velocidad de munecas/dedos)
# para evitar disparos falsos con las manos quietas en reposo.
MIN_NORM_MOTION = 0.015

# Tolerancia de frames consecutivos sin deteccion de manos (por desenfoque rapido)
# antes de reiniciar la ventana temporal.
MAX_NO_HAND_FRAMES = 4

MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

# Mantener la misma resolucion y complejidad que en la captura
# para conservar la escala temporal (30 frames ~ 1 segundo).
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# Mismos indices exactos utilizados en la captura
NOSE_INDEX = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

FACE_EXPR_INDICES = (
    70, 105, 107,
    300, 334, 336,
    159, 145,
    386, 374,
    61, 291, 13, 14, 0, 17, 78, 308,
)

FACE_FEATURES = 3 + len(FACE_EXPR_INDICES) * 3  # 57
BODY_INDICES = (11, 12, 13, 14, 15, 16)
BODY_FEATURES = len(BODY_INDICES) * 4          # 24
HAND_FEATURES = 21 * 3 + 3 + 3                 # 69
INTER_HAND_FEATURES = 4                        # 4
PRESENCE_FEATURES = 4                          # 4

UPPER_BODY_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# 2. CARGA DEL MODELO Y ETIQUETAS
# ============================================================

if not LSTM_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontro el modelo: {LSTM_MODEL_PATH}")

if not LSTM_LABELS_PATH.exists():
    raise FileNotFoundError(f"No se encontraron las etiquetas: {LSTM_LABELS_PATH}")

model = tf.keras.models.load_model(LSTM_MODEL_PATH)

actions = [
    line.strip()
    for line in LSTM_LABELS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if not actions:
    raise ValueError("No se encontraron etiquetas validas.")

if tuple(model.input_shape[1:]) != (SEQUENCE_LENGTH, NUM_FEATURES):
    raise ValueError(
        f"Entrada del modelo incompatible: {model.input_shape}. "
        f"Se esperaba (None, {SEQUENCE_LENGTH}, {NUM_FEATURES})."
    )

if model.output_shape[-1] != len(actions):
    raise ValueError(
        "La cantidad de salidas del modelo no coincide con las etiquetas."
    )


@tf.function(reduce_retracing=True)
def predict_fast(sequence_tensor):
    """Inferencia compilada sin sobrecarga de model.predict()."""
    return model(sequence_tensor, training=False)


# Warm-up del grafo de TensorFlow para evitar lag en el primer disparo
_ = predict_fast(tf.zeros((1, SEQUENCE_LENGTH, NUM_FEATURES), dtype=tf.float32))


# ============================================================
# 3. COLORES
# ============================================================

COLORS = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
    (245, 16, 117),
    (16, 245, 117),
    (117, 16, 245),
]


# ============================================================
# 4. DETECCION Y EXTRACCION IDENTICA A LA CAPTURA (454 FEATURES)
# ============================================================

def mediapipe_detection(image, holistic_model):
    """Procesa una imagen BGR con MediaPipe Holistic."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic_model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results


def landmarks_to_pixels(landmarks, width, height):
    """Convierte landmarks normalizados a espacio de pixeles proporcional."""
    return np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks.landmark],
        dtype=np.float32,
    )


def extract_body_pose(results, width, height, last_body_ref):
    """Extrae los 6 puntos del tren superior (hombros, codos, munecas)."""
    if results.pose_landmarks is None:
        return np.zeros(BODY_FEATURES, dtype=np.float32), last_body_ref, 0.0, None

    pose_landmarks = results.pose_landmarks.landmark
    pose_pts = landmarks_to_pixels(results.pose_landmarks, width, height)
    visibility = np.array([lm.visibility for lm in pose_landmarks], dtype=np.float32)

    pose_nose = pose_pts[0].copy() if visibility[0] >= 0.3 else None

    left_shoulder, right_shoulder = 11, 12
    body_reference = last_body_ref
    pose_present = 0.0

    if visibility[left_shoulder] >= 0.3 and visibility[right_shoulder] >= 0.3:
        shoulder_center = (pose_pts[left_shoulder] + pose_pts[right_shoulder]) / 2.0
        shoulder_width = float(
            np.linalg.norm(pose_pts[left_shoulder, :2] - pose_pts[right_shoulder, :2])
        )
        if shoulder_width >= 1e-6:
            body_reference = (shoulder_center, shoulder_width)
            pose_present = 1.0

    if body_reference is None:
        return np.zeros(BODY_FEATURES, dtype=np.float32), None, 0.0, pose_nose

    shoulder_center, shoulder_width = body_reference

    body_pts = pose_pts[list(BODY_INDICES)].copy()
    body_vis = visibility[list(BODY_INDICES)].copy()

    body_normalized = (body_pts - shoulder_center) / shoulder_width

    for i, vis in enumerate(body_vis):
        if vis < 0.3:
            body_normalized[i] = 0.0
            body_vis[i] = 0.0

    body_features = np.column_stack([body_normalized, body_vis]).flatten().astype(np.float32)
    return body_features, body_reference, pose_present, pose_nose


def extract_face_mesh(results, width, height, body_reference, pose_nose):
    """Extrae posicion de cabeza (3) + 18 puntos de expresion facial (54) = 57."""
    if results.face_landmarks is None:
        return np.zeros(FACE_FEATURES, dtype=np.float32), 0.0, pose_nose

    face_pts = landmarks_to_pixels(results.face_landmarks, width, height)
    nose_pt = face_pts[NOSE_INDEX].copy()

    head_rel_body = np.zeros(3, dtype=np.float32)
    if body_reference is not None:
        shoulder_center, shoulder_width = body_reference
        head_rel_body = (nose_pt - shoulder_center) / shoulder_width

    eye_dist = float(
        np.linalg.norm(face_pts[LEFT_EYE_OUTER, :2] - face_pts[RIGHT_EYE_OUTER, :2])
    )
    if eye_dist < 1e-6:
        return np.zeros(FACE_FEATURES, dtype=np.float32), 0.0, nose_pt

    expr_pts = face_pts[list(FACE_EXPR_INDICES)]
    expr_normalized = ((expr_pts - nose_pt) / eye_dist).flatten().astype(np.float32)

    face_features = np.concatenate([head_rel_body, expr_normalized]).astype(np.float32)
    return face_features, 1.0, nose_pt


def extract_hand(hand_landmarks, width, height, body_reference, nose_pt):
    """Extrae forma local (63) + posicion respecto a hombros (3) y nariz (3) = 69."""
    if hand_landmarks is None:
        return np.zeros(HAND_FEATURES, dtype=np.float32), 0.0, None

    hand_pts = landmarks_to_pixels(hand_landmarks, width, height)
    wrist = hand_pts[0].copy()
    local_pts = hand_pts - wrist

    palm_scale = float(np.linalg.norm(hand_pts[9] - hand_pts[0]))
    if palm_scale < 1e-6:
        return np.zeros(HAND_FEATURES, dtype=np.float32), 0.0, None

    local_pts = local_pts / palm_scale

    wrist_rel_body = np.zeros(3, dtype=np.float32)
    wrist_rel_face = np.zeros(3, dtype=np.float32)

    if body_reference is not None:
        shoulder_center, shoulder_width = body_reference
        wrist_rel_body = (wrist - shoulder_center) / shoulder_width
        if nose_pt is not None:
            wrist_rel_face = (wrist - nose_pt) / shoulder_width

    hand_features = np.concatenate(
        [local_pts.flatten(), wrist_rel_body, wrist_rel_face]
    ).astype(np.float32)

    return hand_features, 1.0, wrist


def compute_velocity(curr_static, prev_static):
    """Calcula la velocidad temporal anulando saltos de entrada/salida de cuadro."""
    if prev_static is None:
        return np.zeros(STATIC_FEATURES, dtype=np.float32)

    vel = curr_static - prev_static

    curr_pres = curr_static[-4:]
    prev_pres = prev_static[-4:]

    face_end = FACE_FEATURES
    body_end = face_end + BODY_FEATURES
    left_end = body_end + HAND_FEATURES
    right_end = left_end + HAND_FEATURES
    inter_end = right_end + INTER_HAND_FEATURES

    if not (curr_pres[1] > 0.5 and prev_pres[1] > 0.5):
        vel[0:face_end] = 0.0

    if not (curr_pres[0] > 0.5 and prev_pres[0] > 0.5):
        vel[face_end:body_end] = 0.0

    if not (curr_pres[2] > 0.5 and prev_pres[2] > 0.5):
        vel[body_end:left_end] = 0.0

    if not (curr_pres[3] > 0.5 and prev_pres[3] > 0.5):
        vel[left_end:right_end] = 0.0

    if not (
        curr_pres[2] > 0.5 and prev_pres[2] > 0.5
        and curr_pres[3] > 0.5 and prev_pres[3] > 0.5
    ):
        vel[right_end:inter_end] = 0.0

    vel[inter_end:] = 0.0
    return vel.astype(np.float32)


def extract_keypoints(results, width, height, last_body_ref, prev_static):
    """Construye el vector completo de 454 caracteristicas por frame."""
    body_features, body_reference, pose_present, pose_nose = extract_body_pose(
        results, width, height, last_body_ref
    )

    face_features, face_present, nose_pt = extract_face_mesh(
        results, width, height, body_reference, pose_nose
    )

    left_hand, left_present, left_wrist = extract_hand(
        results.left_hand_landmarks, width, height, body_reference, nose_pt
    )
    right_hand, right_present, right_wrist = extract_hand(
        results.right_hand_landmarks, width, height, body_reference, nose_pt
    )

    inter_hand = np.zeros(INTER_HAND_FEATURES, dtype=np.float32)
    if (
        left_wrist is not None
        and right_wrist is not None
        and body_reference is not None
    ):
        _, shoulder_width = body_reference
        diff = (right_wrist - left_wrist) / shoulder_width
        dist = np.float32(np.linalg.norm(diff))
        inter_hand = np.array([diff[0], diff[1], diff[2], dist], dtype=np.float32)

    presence = np.array(
        [pose_present, face_present, left_present, right_present],
        dtype=np.float32,
    )

    static_vec = np.concatenate(
        [face_features, body_features, left_hand, right_hand, inter_hand, presence]
    ).astype(np.float32)

    velocity_vec = compute_velocity(static_vec, prev_static)
    keypoints = np.concatenate([static_vec, velocity_vec]).astype(np.float32)

    if keypoints.shape != (NUM_FEATURES,):
        raise ValueError(
            f"Se esperaban {NUM_FEATURES} caracteristicas, se obtuvieron {keypoints.shape}."
        )

    has_hand = (left_present > 0.5) or (right_present > 0.5)
    return keypoints, body_reference, static_vec, has_hand


# ============================================================
# 5. CALCULO DE MOVIMIENTO NORMALIZADO
# ============================================================

def calculate_normalized_motion(sequence_buffer):
    """
    Mide la energia de movimiento de ambas manos usando las velocidades
    ya normalizadas dentro del tensor (invariante a la distancia a la camara).
    """
    seq_arr = np.asarray(sequence_buffer, dtype=np.float32)
    vel_block = seq_arr[:, STATIC_FEATURES:]

    # Velocidades de mano izquierda (81..150) y mano derecha (150..219)
    left_vel = vel_block[:, 81:150]
    right_vel = vel_block[:, 150:219]

    left_motion = float(np.mean(np.abs(left_vel)))
    right_motion = float(np.mean(np.abs(right_vel)))

    return max(left_motion, right_motion)


# ============================================================
# 6. VISUALIZACION
# ============================================================

def draw_styled_landmarks(image, results):
    """Dibuja malla facial, tren superior y ambas manos."""
    height, width = image.shape[:2]

    # 1. Cara
    if results.face_landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 190, 255), thickness=1, circle_radius=1
            ),
        )
        highlight_indices = (NOSE_INDEX, LEFT_EYE_OUTER, RIGHT_EYE_OUTER) + FACE_EXPR_INDICES
        for idx in highlight_indices:
            pt = results.face_landmarks.landmark[idx]
            x, y = int(pt.x * width), int(pt.y * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)

    # 2. Tren superior
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark
        for start, end in UPPER_BODY_CONNECTIONS:
            p1, p2 = landmarks[start], landmarks[end]
            if p1.visibility < 0.3 or p2.visibility < 0.3:
                continue
            x1, y1 = int(p1.x * width), int(p1.y * height)
            x2, y2 = int(p2.x * width), int(p2.y * height)
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                cv2.line(image, (x1, y1), (x2, y2), (180, 70, 130), 2, cv2.LINE_AA)

        for idx in BODY_INDICES:
            lm = landmarks[idx]
            if lm.visibility >= 0.3:
                x, y = int(lm.x * width), int(lm.y * height)
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(image, (x, y), 4, (255, 80, 40), -1, cv2.LINE_AA)

    # 3. Manos
    for hand_lms in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_lms is not None:
            mp_drawing.draw_landmarks(
                image,
                hand_lms,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 80, 40), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            )


def prob_viz(probs, input_frame):
    """Dibuja barras de probabilidad debajo del panel superior (sin solapamiento)."""
    output_frame = input_frame.copy()
    start_y = 95
    row_height = 32

    for num, prob in enumerate(probs):
        color = COLORS[num % len(COLORS)]
        y1 = start_y + num * row_height
        y2 = y1 + 24

        cv2.rectangle(
            output_frame,
            (10, y1),
            (10 + int(prob * 180), y2),
            color,
            -1,
        )
        cv2.putText(
            output_frame,
            f"{actions[num]}: {prob:.2f}",
            (200, y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


# ============================================================
# 7. BUCLE DE PREDICCION EN TIEMPO REAL
# ============================================================

sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=STABILITY_WINDOW)

sentence = []
res = np.zeros(len(actions), dtype=np.float32)
current_label = "NEUTRO"
cooldown_until = 0.0

last_body_ref = None
prev_static = None
no_hand_frames = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara web.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.time()

try:
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            height, width = frame.shape[:2]

            results = mediapipe_detection(frame, holistic)
            image = frame.copy()
            draw_styled_landmarks(image, results)

            in_cooldown = now < cooldown_until

            if in_cooldown:
                # Ignorar frames durante la pausa para evitar falsos disparos al bajar la mano
                sequence.clear()
                predictions.clear()
                prev_static = None
                no_hand_frames = 0
                res = np.zeros(len(actions), dtype=np.float32)
            else:
                if len(sequence) == 0 and current_label != "NEUTRO":
                    current_label = "NEUTRO"

                keypoints, last_body_ref, prev_static, has_hand = extract_keypoints(
                    results, width, height, last_body_ref, prev_static
                )

                if not has_hand:
                    no_hand_frames += 1
                else:
                    no_hand_frames = 0

                # Reiniciar ventana solo si las manos llevan varios frames fuera de cuadro
                if no_hand_frames >= MAX_NO_HAND_FRAMES:
                    sequence.clear()
                    predictions.clear()
                    prev_static = None
                    res = np.zeros(len(actions), dtype=np.float32)
                    current_label = "NEUTRO"
                else:
                    sequence.append(keypoints)

                    if len(sequence) == SEQUENCE_LENGTH:
                        motion = calculate_normalized_motion(sequence)

                        if motion >= MIN_NORM_MOTION:
                            input_tensor = tf.convert_to_tensor(
                                np.expand_dims(np.asarray(sequence, dtype=np.float32), axis=0)
                            )
                            res = predict_fast(input_tensor)[0].numpy()

                            pred_idx = int(np.argmax(res))
                            predictions.append((pred_idx, float(res[pred_idx])))

                            if len(predictions) == STABILITY_WINDOW:
                                recent_indices = [p[0] for p in predictions]
                                dominant_idx = max(set(recent_indices), key=recent_indices.count)
                                consistency = recent_indices.count(dominant_idx)

                                dominant_probs = [
                                    p[1] for p in predictions if p[0] == dominant_idx
                                ]
                                mean_conf = float(np.mean(dominant_probs))

                                if (
                                    consistency >= MIN_CONSISTENT_FRAMES
                                    and mean_conf >= THRESHOLD
                                ):
                                    current_label = actions[dominant_idx]
                                    if not sentence or current_label != sentence[-1]:
                                        sentence.append(current_label)

                                    # Limpiar ventana y activar pausa para evitar encadenamientos (ej. J -> SER)
                                    sequence.clear()
                                    predictions.clear()
                                    prev_static = None
                                    res = np.zeros(len(actions), dtype=np.float32)
                                    cooldown_until = now + COOLDOWN_SECONDS
                        else:
                            predictions.clear()
                            res = np.zeros(len(actions), dtype=np.float32)
                            current_label = "NEUTRO"

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Mostrar barras de probabilidad cuando el bufer esta lleno y activo
            if len(sequence) == SEQUENCE_LENGTH and np.max(res) > 0:
                image = prob_viz(res, image)

            # =================================================
            # INTERFAZ SUPERIOR
            # =================================================
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            status_color = (
                (80, 80, 80)
                if current_label == "NEUTRO"
                else (245, 117, 16)
            )
            status_text = (
                f"{current_label} (PAUSA)"
                if time.time() < cooldown_until
                else current_label
            )

            cv2.rectangle(image, (0, 0), (width, 80), status_color, -1)

            cv2.putText(
                image,
                f"ESTADO: {status_text}  |  Buffer: {len(sequence)}/{SEQUENCE_LENGTH}  |  FPS: {fps:.0f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "FRASE: " + (" ".join(sentence) if sentence else "..."),
                (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ALIS - Prediccion Dinamica (Q: salir | C: limpiar)", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("c"), ord("C")):
                sentence.clear()
                sequence.clear()
                predictions.clear()
                prev_static = None
                cooldown_until = 0.0
                res = np.zeros(len(actions), dtype=np.float32)
                current_label = "NEUTRO"

finally:
    cap.release()
    cv2.destroyAllWindows()