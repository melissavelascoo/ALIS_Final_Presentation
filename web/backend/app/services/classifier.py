"""Clasificación con dataset completo y ejecución simultánea de MLP (estático) y LSTM (dinámico)."""
from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from app.core.config import (
    CONFIDENCE_THRESHOLD,
    LSTM_LABELS_PATH,
    LSTM_MODEL_PATH,
    LSTM_NUM_FEATURES,
    LSTM_SEQUENCE_LENGTH,
    MIN_DET_CONF,
    MIN_TRK_CONF,
    MLP_CONFIDENCE_THRESHOLD,
    MLP_LABELS_PATH,
    MLP_MODEL_PATH,
    MLP_NUM_FEATURES,
)
from app.services.hand_processing import (
    STATIC_FEATURES,
    calculate_dynamic_frame_motion,
    extract_dynamic_features,
    extract_static_features,
)

MIN_DYNAMIC_HAND_RATIO = 0.35
MIN_NORM_MOTION = 0.012
MAX_NO_HAND_FRAMES = 4
DYNAMIC_THRESHOLD = CONFIDENCE_THRESHOLD
STABILITY_WINDOW = 5
MIN_CONSISTENT_FRAMES = 3
COOLDOWN_SECONDS = 1.0
STATIC_SMOOTHING_WINDOW = 5


def read_labels(labels_path) -> list[str]:
    content = labels_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError(f"{labels_path.name} está vacío.")

    if content.startswith("["):
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError(f"Formato JSON inválido en {labels_path.name}.")
        labels = [str(item).strip().upper() for item in parsed if str(item).strip()]
    else:
        labels = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            match = re.match(r"^\d+\s+(.+)$", line)
            labels.append((match.group(1) if match else line).strip().upper())

    if not labels:
        raise ValueError(f"{labels_path.name} está vacío.")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{labels_path.name} contiene etiquetas repetidas.")
    return labels


def _calculate_normalized_motion(sequence_buffer: deque[np.ndarray]) -> float:
    seq_arr = np.asarray(sequence_buffer, dtype=np.float32)
    vel_block = seq_arr[:, STATIC_FEATURES:]
    left_motion = float(np.mean(np.abs(vel_block[:, 81:150])))
    right_motion = float(np.mean(np.abs(vel_block[:, 150:219])))
    return max(left_motion, right_motion)


def _dynamic_has_priority(result: dict[str, Any]) -> bool:
    return (
        result.get("accepted") is True
        or result.get("activity") == "moving"
        or result.get("state") in {"uncertain", "cooldown", "recognized"}
    )


def _prepare_static_vector(features: Any, expected_dim: int) -> np.ndarray | None:
    if features is None:
        return None
    arr = np.asarray(features, dtype=np.float32).reshape(-1)
    if arr.size < expected_dim:
        raise ValueError(
            f"Se obtuvieron {arr.size} características estáticas y se esperaban {expected_dim}."
        )
    if arr.size > expected_dim:
        arr = arr[:expected_dim]
    return arr.reshape(1, expected_dim)


class RecognitionService:
    def __init__(self):
        self._guard = threading.RLock()
        self._hands: Any | None = None
        self._holistic: Any | None = None

        self._static_model: Any | None = None
        self._static_labels: list[str] = []
        self._static_error: str | None = "Modelo estático pendiente de conectar."
        self._static_fingerprint: tuple | None = None
        self._static_predictions: deque[np.ndarray] = deque(maxlen=STATIC_SMOOTHING_WINDOW)

        self._dynamic_model: Any | None = None
        self._dynamic_labels: list[str] = []
        self._dynamic_error: str | None = "Modelo dinámico pendiente de conectar."
        self._dynamic_fingerprint: tuple | None = None

        self._seq_buffer: deque[np.ndarray] = deque(maxlen=LSTM_SEQUENCE_LENGTH)
        self._hand_buffer: deque[int] = deque(maxlen=LSTM_SEQUENCE_LENGTH)
        self._dyn_predictions: deque[tuple[int, float]] = deque(maxlen=STABILITY_WINDOW)
        self._last_body_ref: Any = None
        self._prev_static: np.ndarray | None = None
        self._no_hand_frames: int = 0
        self._cooldown_until: float = 0.0
        self._last_dynamic_confirmed: tuple[str, float, float] | None = None

    def _ensure_static_model(self):
        paths = (MLP_MODEL_PATH, MLP_LABELS_PATH)
        fingerprint = tuple(
            (p.is_file(), p.stat().st_mtime_ns if p.is_file() else None) for p in paths
        )
        if fingerprint == self._static_fingerprint:
            return
        self._static_fingerprint = fingerprint
        self._static_predictions.clear()
        self._static_model = None
        self._static_labels = []
        self._static_error = None

        if not MLP_MODEL_PATH.is_file() or not MLP_LABELS_PATH.is_file():
            self._static_error = "Faltan model.h5 o labels.txt en static_signs."
            return
        try:
            labels = read_labels(MLP_LABELS_PATH)
            model = tf.keras.models.load_model(str(MLP_MODEL_PATH), compile=False)
            if tuple(model.input_shape[1:]) != (MLP_NUM_FEATURES,):
                raise ValueError(
                    f"Entrada no compatible: {model.input_shape}. Se esperaba ({MLP_NUM_FEATURES},)."
                )
            if int(model.output_shape[-1]) != len(labels):
                raise ValueError(
                    f"El modelo tiene {model.output_shape[-1]} salidas pero {len(labels)} etiquetas."
                )
            if self._hands is None:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=MIN_DET_CONF,
                    min_tracking_confidence=MIN_TRK_CONF,
                )
            _ = model(tf.zeros((1, MLP_NUM_FEATURES), dtype=np.float32), training=False)
            self._static_model = model
            self._static_labels = labels
        except Exception as exc:
            self._static_error = f"{type(exc).__name__}: {exc}"

    def _ensure_dynamic_model(self):
        paths = (LSTM_MODEL_PATH, LSTM_LABELS_PATH)
        fingerprint = tuple(
            (p.is_file(), p.stat().st_mtime_ns if p.is_file() else None) for p in paths
        )
        if fingerprint == self._dynamic_fingerprint:
            return
        self._dynamic_fingerprint = fingerprint
        self._dynamic_model = None
        self._dynamic_labels = []
        self._dynamic_error = None

        if not LSTM_MODEL_PATH.is_file() or not LSTM_LABELS_PATH.is_file():
            self._dynamic_error = "Faltan model.h5 o labels.txt en dynamic_signs."
            return
        try:
            labels = read_labels(LSTM_LABELS_PATH)
            model = tf.keras.models.load_model(str(LSTM_MODEL_PATH), compile=False)
            if tuple(model.input_shape[1:]) != (LSTM_SEQUENCE_LENGTH, LSTM_NUM_FEATURES):
                raise ValueError(
                    f"Entrada no compatible: {model.input_shape}. "
                    f"Se esperaba ({LSTM_SEQUENCE_LENGTH}, {LSTM_NUM_FEATURES})."
                )
            if int(model.output_shape[-1]) != len(labels):
                raise ValueError(
                    f"El modelo tiene {model.output_shape[-1]} salidas pero {len(labels)} etiquetas."
                )
            if self._holistic is None:
                self._holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=MIN_DET_CONF,
                    min_tracking_confidence=MIN_TRK_CONF,
                )
            _ = model(
                tf.zeros((1, LSTM_SEQUENCE_LENGTH, LSTM_NUM_FEATURES), dtype=np.float32),
                training=False,
            )
            self._dynamic_model = model
            self._dynamic_labels = labels
        except Exception as exc:
            self._dynamic_error = f"{type(exc).__name__}: {exc}"

    def _clear_dynamic_window(self):
        self._seq_buffer.clear()
        self._hand_buffer.clear()
        self._dyn_predictions.clear()
        self._prev_static = None
        self._no_hand_frames = 0

    def reset(self) -> dict:
        with self._guard:
            self._clear_dynamic_window()
            self._last_body_ref = None
            self._cooldown_until = 0.0
            self._last_dynamic_confirmed = None
            if self._holistic is not None:
                self._holistic.reset()
            return {"status": "ok"}

    def status(self) -> dict:
        with self._guard:
            self._ensure_static_model()
            self._ensure_dynamic_model()
            static_ok = self._static_model is not None
            dynamic_ok = self._dynamic_model is not None
            loaded_models = [
                name for name, loaded in (("static", static_ok), ("dynamic", dynamic_ok))
                if loaded
            ]
            combined_classes = list(
                dict.fromkeys(list(self._static_labels) + list(self._dynamic_labels))
            )
            return {
                "loaded": static_ok or dynamic_ok,
                "models_loaded": loaded_models,
                "message": (
                    "Reconocimiento disponible."
                    if loaded_models
                    else (self._static_error or self._dynamic_error)
                ),
                "classes": combined_classes,
                "model_filename": MLP_MODEL_PATH.name if static_ok else (LSTM_MODEL_PATH.name if dynamic_ok else None),
                "model_filenames": {
                    "static": MLP_MODEL_PATH.name if static_ok else None,
                    "dynamic": LSTM_MODEL_PATH.name if dynamic_ok else None,
                },
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "static": {
                    "loaded": static_ok,
                    "message": "Reconocimiento estático disponible." if static_ok else self._static_error,
                    "classes": list(self._static_labels),
                    "model_filename": MLP_MODEL_PATH.name if static_ok else None,
                },
                "dynamic": {
                    "loaded": dynamic_ok,
                    "message": "Reconocimiento dinámico disponible." if dynamic_ok else self._dynamic_error,
                    "classes": list(self._dynamic_labels),
                    "model_filename": LSTM_MODEL_PATH.name if dynamic_ok else None,
                    "sequence_length": LSTM_SEQUENCE_LENGTH,
                },
            }

    def _eval_static(self, image_rgb, width: int, height: int) -> dict[str, Any]:
        if self._static_model is None or self._hands is None:
            self._static_predictions.clear()
            return {
                "source": "static",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "unavailable",
                "activity": "unavailable",
                "predictions": {},
            }

        results = self._hands.process(image_rgb)
        raw_features, state = extract_static_features(results, width, height)
        model_input = _prepare_static_vector(raw_features, MLP_NUM_FEATURES)
        if model_input is None:
            self._static_predictions.clear()
            return {
                "source": "static",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": state,
                "predictions": {},
            }

        raw_scores = self._static_model(model_input, training=False).numpy()[0]
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if scores.size != len(self._static_labels) or not np.isfinite(scores).all():
            self._static_predictions.clear()
            return {
                "source": "static",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "invalid",
                "predictions": {},
            }

        self._static_predictions.append(scores)
        scores = np.mean(self._static_predictions, axis=0)
        idx = int(np.argmax(scores))
        label = self._static_labels[idx]
        conf = float(scores[idx])
        accepted = conf >= MLP_CONFIDENCE_THRESHOLD

        return {
            "source": "static",
            "accepted": accepted,
            "label": label,
            "confidence": round(conf, 6),
            "state": "recognized" if accepted else "uncertain",
            "predictions": {name: round(float(val), 6) for name, val in zip(self._static_labels, scores)},
        }

    def _eval_dynamic(self, image_rgb, width: int, height: int, now: float) -> dict[str, Any]:
        if self._dynamic_model is None or self._holistic is None:
            return {
                "source": "dynamic",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "unavailable",
                "predictions": {},
            }

        if now < self._cooldown_until:
            self._clear_dynamic_window()
            hold_label = (
                self._last_dynamic_confirmed[0] if self._last_dynamic_confirmed else None
            )
            hold_conf = (
                self._last_dynamic_confirmed[1] if self._last_dynamic_confirmed else 0.0
            )
            return {
                "source": "dynamic",
                "accepted": hold_label is not None,
                "label": hold_label,
                "confidence": round(hold_conf, 6),
                "state": "cooldown",
                "activity": "cooldown",
                "predictions": {},
            }

        results = self._holistic.process(image_rgb)
        keypoints, self._last_body_ref, self._prev_static, has_hand = extract_dynamic_features(
            results, width, height, self._last_body_ref, self._prev_static
        )
        activity = (
            "moving"
            if calculate_dynamic_frame_motion(keypoints) >= MIN_NORM_MOTION
            else "stable"
        )

        if not has_hand:
            self._no_hand_frames += 1
        else:
            self._no_hand_frames = 0

        if self._no_hand_frames >= MAX_NO_HAND_FRAMES:
            self._clear_dynamic_window()
            return {
                "source": "dynamic",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "no_hand",
                "activity": "stable",
                "predictions": {},
            }

        self._seq_buffer.append(keypoints)
        self._hand_buffer.append(1 if has_hand else 0)

        if len(self._seq_buffer) < LSTM_SEQUENCE_LENGTH:
            return {
                "source": "dynamic",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "collecting",
                "activity": activity,
                "predictions": {},
            }

        if (sum(self._hand_buffer) / LSTM_SEQUENCE_LENGTH) < MIN_DYNAMIC_HAND_RATIO:
            self._dyn_predictions.clear()
            return {
                "source": "dynamic",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "no_hand",
                "activity": activity,
                "predictions": {},
            }

        if _calculate_normalized_motion(self._seq_buffer) < MIN_NORM_MOTION:
            self._dyn_predictions.clear()
            return {
                "source": "dynamic",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "idle",
                "activity": "stable",
                "predictions": {},
            }

        seq_input = np.expand_dims(np.asarray(self._seq_buffer, dtype=np.float32), axis=0)
        raw_scores = self._dynamic_model(seq_input, training=False).numpy()[0]
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if scores.size != len(self._dynamic_labels) or not np.isfinite(scores).all():
            return {
                "source": "dynamic",
                "accepted": False,
                "label": None,
                "confidence": 0.0,
                "state": "invalid",
                "activity": activity,
                "predictions": {},
            }

        idx = int(np.argmax(scores))
        label = self._dynamic_labels[idx]
        conf = float(scores[idx])
        self._dyn_predictions.append((idx, conf))

        if len(self._dyn_predictions) == STABILITY_WINDOW:
            recent_indices = [p[0] for p in self._dyn_predictions]
            dominant_idx = max(set(recent_indices), key=recent_indices.count)
            dominant_label = self._dynamic_labels[dominant_idx]
            consistency = recent_indices.count(dominant_idx)
            dominant_probs = [p[1] for p in self._dyn_predictions if p[0] == dominant_idx]
            mean_conf = float(np.mean(dominant_probs))

            if consistency >= MIN_CONSISTENT_FRAMES and mean_conf >= DYNAMIC_THRESHOLD:
                self._clear_dynamic_window()
                self._cooldown_until = now + COOLDOWN_SECONDS
                self._last_dynamic_confirmed = (dominant_label, mean_conf, now)
                return {
                    "source": "dynamic",
                    "accepted": True,
                    "label": dominant_label,
                    "confidence": round(mean_conf, 6),
                    "state": "recognized",
                    "activity": "moving",
                    "predictions": {name: round(float(val), 6) for name, val in zip(self._dynamic_labels, scores)},
                }

        return {
            "source": "dynamic",
            "accepted": False,
            "label": label,
            "confidence": round(conf, 6),
            "state": "uncertain",
            "activity": activity,
            "predictions": {name: round(float(val), 6) for name, val in zip(self._dynamic_labels, scores)},
        }

    def predict(self, frame_bytes: bytes, mode: str = "both") -> dict:
        with self._guard:
            buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("No se pudo decodificar la imagen enviada.")

            height, width = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            now = time.time()

            self._ensure_static_model()
            self._ensure_dynamic_model()

            if self._static_model is None and self._dynamic_model is None:
                raise RuntimeError(self._static_error or self._dynamic_error or "Modelos no disponibles.")

            evaluate_static = mode in {"auto", "both", "static"}
            evaluate_dynamic = mode in {"auto", "both", "dynamic"}

            dynamic_res = (
                self._eval_dynamic(image_rgb, width, height, now)
                if evaluate_dynamic
                else {"source": "dynamic", "accepted": False, "label": None, "confidence": 0.0, "state": "disabled", "predictions": {}}
            )
            suppress_static = (
                evaluate_static and evaluate_dynamic and _dynamic_has_priority(dynamic_res)
            )
            if not evaluate_static or suppress_static:
                self._static_predictions.clear()
            static_res = (
                self._eval_static(image_rgb, width, height)
                if evaluate_static and not suppress_static
                else {
                    "source": "static",
                    "accepted": False,
                    "label": None,
                    "confidence": 0.0,
                    "state": "suppressed" if suppress_static else "disabled",
                    "predictions": {},
                }
            )

            static_label = static_res["label"] if static_res.get("accepted") else None
            dynamic_label = dynamic_res["label"] if dynamic_res.get("accepted") else None

            if dynamic_label:
                primary_label = dynamic_label
                primary_conf = dynamic_res["confidence"]
                model_used = "dynamic"
                accepted = True
                final_state = "recognized"
            elif static_label:
                primary_label = static_label
                primary_conf = static_res["confidence"]
                model_used = "static"
                accepted = True
                final_state = "recognized"
            else:
                primary_label = None
                primary_conf = None
                model_used = None
                accepted = False
                if dynamic_res.get("state") == "cooldown":
                    final_state = "cooldown"
                elif static_res.get("state") == "no_hand" and dynamic_res.get("state") in {"no_hand", "unavailable"}:
                    final_state = "no_hand"
                elif dynamic_res.get("state") in {"collecting", "idle"}:
                    final_state = dynamic_res["state"]
                else:
                    final_state = static_res.get("state") or dynamic_res.get("state") or "uncertain"

            combined_predictions = {
                **{f"static:{k}": v for k, v in static_res.get("predictions", {}).items()},
                **{f"dynamic:{k}": v for k, v in dynamic_res.get("predictions", {}).items()},
            }

            return {
                "class_name": primary_label,
                "confidence": primary_conf,
                "accepted": accepted,
                "state": final_state,
                "model_used": model_used,
                "static": static_res,
                "dynamic": dynamic_res,
                "predictions": combined_predictions,
            }


recognition = RecognitionService()