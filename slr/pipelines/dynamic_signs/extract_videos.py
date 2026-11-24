import json
import shutil
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from paths import LSTM_VIDEOS_DIR, LSTM_RAW_DIR


# ============================================================
# CONFIGURACION (IDENTICA A INFERENCIA EN TIEMPO REAL)
# ============================================================

SEQUENCE_LENGTH = 30
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
MIN_VALID_HAND_RATIO = 0.35

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
STATIC_FEATURES = (
    FACE_FEATURES + BODY_FEATURES + HAND_FEATURES * 2 + INTER_HAND_FEATURES + PRESENCE_FEATURES
)  # 227
TOTAL_FEATURES = STATIC_FEATURES * 2           # 454

mp_holistic = mp.solutions.holistic


# ============================================================
# FUNCIONES DE EXTRACCION (454 CARACTERISTICAS)
# ============================================================

def landmarks_to_pixels(landmarks, width, height):
    return np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks.landmark],
        dtype=np.float32,
    )


def extract_body_pose(results, width, height, last_body_ref):
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


def extract_keypoints(results, image_shape, last_body_ref, prev_static):
    height, width = image_shape[:2]

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
    if left_wrist is not None and right_wrist is not None and body_reference is not None:
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

    has_hand = (left_present > 0.5) or (right_present > 0.5)
    return keypoints, body_reference, static_vec, has_hand


# ============================================================
# PROCESAMIENTO POR LOTE DE TODOS LOS VIDEOS
# ============================================================

def process_all_videos():
    if not LSTM_VIDEOS_DIR.exists():
        raise FileNotFoundError(f"No existe el directorio de videos: {LSTM_VIDEOS_DIR}")

    action_dirs = sorted([d for d in LSTM_VIDEOS_DIR.iterdir() if d.is_dir()])
    if not action_dirs:
        raise ValueError(f"No se encontraron carpetas de senas en: {LSTM_VIDEOS_DIR}")

    total_ok = 0
    total_skipped = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
    ) as holistic:

        for action_dir in action_dirs:
            action = action_dir.name
            seq_dirs = sorted(
                [d for d in action_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda p: int(p.name),
            )

            print(f"\nExtrayendo landmarks para clase: {action}")

            for seq_dir in seq_dirs:
                video_path = seq_dir / "video.avi"
                src_meta_path = seq_dir / "metadata.json"

                if not video_path.is_file():
                    total_skipped += 1
                    continue

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    total_skipped += 1
                    continue

                # Reiniciar estado interno del tracker entre videos distintos
                holistic.reset()

                last_body_ref = None
                prev_static = None
                frames_keypoints = []
                frames_with_hands = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = holistic.process(image_rgb)
                    image_rgb.flags.writeable = True

                    kp, last_body_ref, prev_static, has_hand = extract_keypoints(
                        results, frame.shape, last_body_ref, prev_static
                    )
                    if has_hand:
                        frames_with_hands += 1
                    frames_keypoints.append(kp)

                cap.release()

                if len(frames_keypoints) != SEQUENCE_LENGTH:
                    print(f"  [DESCARTADO] {action}/{seq_dir.name}: {len(frames_keypoints)} frames.")
                    total_skipped += 1
                    continue

                hand_ratio = frames_with_hands / SEQUENCE_LENGTH
                if hand_ratio < MIN_VALID_HAND_RATIO:
                    print(f"  [DESCARTADO] {action}/{seq_dir.name}: baja deteccion de manos ({hand_ratio:.0%}).")
                    total_skipped += 1
                    continue

                # Guardar .npy en LSTM_RAW_DIR manteniendo el mismo ID y sesion
                out_seq_dir = LSTM_RAW_DIR / action / seq_dir.name
                shutil.rmtree(out_seq_dir, ignore_errors=True)
                out_seq_dir.mkdir(parents=True, exist_ok=True)

                for idx, kp in enumerate(frames_keypoints):
                    np.save(out_seq_dir / f"{idx}.npy", kp)

                session_id = "unknown"
                if src_meta_path.is_file():
                    with open(src_meta_path, "r", encoding="utf-8") as f:
                        src_meta = json.load(f)
                        session_id = src_meta.get("session", "unknown")

                out_metadata = {
                    "action": action,
                    "session": session_id,
                    "sequence_id": int(seq_dir.name),
                    "sequence_length": SEQUENCE_LENGTH,
                    "num_features": TOTAL_FEATURES,
                    "static_features": STATIC_FEATURES,
                    "hand_detection_ratio": round(hand_ratio, 3),
                    "extracted_at": datetime.now().isoformat(timespec="seconds"),
                    "complete": True,
                }
                with open(out_seq_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(out_metadata, f, indent=4, ensure_ascii=False)

                total_ok += 1

    print("\n" + "=" * 50)
    print(f"Extraccion finalizada: {total_ok} secuencias validas | {total_skipped} descartadas.")
    print(f"Listo para ejecutar splits.py desde: {LSTM_RAW_DIR}")


if __name__ == "__main__":
    process_all_videos()