"""Extracción y normalización de landmarks para MLP (88) y LSTM (454)."""
from __future__ import annotations

import numpy as np

from app.core.config import LSTM_NUM_FEATURES, MLP_NUM_FEATURES

# ============================================================
# CONSTANTES: DYNAMIC LSTM (454 CARACTERISTICAS)
# ============================================================

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


def calculate_dynamic_frame_motion(keypoints: np.ndarray) -> float:
    """Calcula movimiento de manos en un frame con el mismo criterio del LSTM."""
    features = np.asarray(keypoints, dtype=np.float32).reshape(-1)
    if features.size != LSTM_NUM_FEATURES:
        raise ValueError(
            f"Se esperaban ({LSTM_NUM_FEATURES},) características dinámicas."
        )
    velocity = features[STATIC_FEATURES:]
    left_motion = float(np.mean(np.abs(velocity[81:150])))
    right_motion = float(np.mean(np.abs(velocity[150:219])))
    return max(left_motion, right_motion)


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    return np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks.landmark],
        dtype=np.float32,
    )


# ============================================================
# EXTRACCION ESTATICA (88 CARACTERISTICAS)
# ============================================================

def extract_static_features(results, width: int, height: int) -> tuple[np.ndarray | None, str]:
    """Extrae las 63 coordenadas normalizadas que espera el modelo MLP."""
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None, "no_hand"

    hand_lms = results.multi_hand_landmarks[0]
    pts = landmarks_to_pixels(hand_lms, width, height)
    if pts.shape != (21, 3) or not np.all(np.isfinite(pts)):
        return None, "invalid_frame"

    local_pts = pts - pts[0]
    scale = float(np.max(np.linalg.norm(local_pts, axis=1)))
    if scale < 1e-4:
        return None, "invalid_frame"

    features = (local_pts / scale).flatten().astype(np.float32)

    if features.shape != (MLP_NUM_FEATURES,) or not np.all(np.isfinite(features)):
        return None, "invalid_frame"

    return features, "ready"


# ============================================================
# EXTRACCION DINAMICA (454 CARACTERISTICAS)
# ============================================================

def _extract_body_pose(results, width: int, height: int, last_body_ref):
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


def _extract_face_mesh(results, width: int, height: int, body_reference, pose_nose):
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


def _extract_holistic_hand(hand_landmarks, width: int, height: int, body_reference, nose_pt):
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


def _compute_velocity(curr_static: np.ndarray, prev_static: np.ndarray | None) -> np.ndarray:
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


def extract_dynamic_features(
    results,
    width: int,
    height: int,
    last_body_ref,
    prev_static: np.ndarray | None,
) -> tuple[np.ndarray, object, np.ndarray, bool]:
    """Extrae el vector de 454 características desde MediaPipe Holistic."""
    body_features, body_reference, pose_present, pose_nose = _extract_body_pose(
        results, width, height, last_body_ref
    )
    face_features, face_present, nose_pt = _extract_face_mesh(
        results, width, height, body_reference, pose_nose
    )
    left_hand, left_present, left_wrist = _extract_holistic_hand(
        results.left_hand_landmarks, width, height, body_reference, nose_pt
    )
    right_hand, right_present, right_wrist = _extract_holistic_hand(
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

    velocity_vec = _compute_velocity(static_vec, prev_static)
    keypoints = np.concatenate([static_vec, velocity_vec]).astype(np.float32)

    if keypoints.shape != (LSTM_NUM_FEATURES,) or not np.all(np.isfinite(keypoints)):
        raise ValueError(f"Se esperaban ({LSTM_NUM_FEATURES},) características válidas.")

    has_hand = (left_present > 0.5) or (right_present > 0.5)
    return keypoints, body_reference, static_vec, has_hand