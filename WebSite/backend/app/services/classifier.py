"""Clasificación de fotogramas mediante el modelo de HandSignDetector."""
from __future__ import annotations

import re
import threading
from typing import Any

import cv2
import numpy as np

from app.core.config import (
    ALLOWED_CLASSES, CONFIDENCE_THRESHOLD, LABELS_FILE, MODEL_FILE,
)
from app.services.hand_processing import hand_to_white_canvas


def read_labels():
    labels: list[str] = []
    for raw in LABELS_FILE.read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r'^\d+\s+(.+)$', line)
        labels.append((match.group(1) if match else line).strip().upper())
    if not labels:
        raise ValueError('labels.txt está vacío.')
    if len(labels) != len(set(labels)):
        raise ValueError('labels.txt contiene etiquetas repetidas.')
    return labels


class RecognitionService:
    def __init__(self):
        self._guard = threading.RLock()
        self._classifier: Any | None = None
        self._detector: Any | None = None
        self._labels: list[str] = []
        self._error: str | None = 'Modelo pendiente de conectar.'
        self._fingerprint: tuple | None = None

    def _ensure_model(self):
        """Recarga el modelo si cambian sus archivos."""
        paths = (MODEL_FILE, LABELS_FILE)
        fingerprint = tuple((path.is_file(), path.stat().st_mtime_ns if path.is_file() else None) for path in paths)
        if fingerprint == self._fingerprint:
            return
        with self._guard:
            if fingerprint == self._fingerprint:
                return
            self._fingerprint = fingerprint
            self._classifier = None
            self._detector = None
            self._error = None
            self._labels = []
            if not MODEL_FILE.is_file() or not LABELS_FILE.is_file():
                self._error = 'Faltan keras_model.h5 o labels.txt en HandSignDetector/Model.'
                return
            try:
                labels = read_labels()
                self._labels = labels

                from cvzone.ClassificationModule import Classifier
                from cvzone.HandTrackingModule import HandDetector

                classifier = Classifier(str(MODEL_FILE), str(LABELS_FILE))
                model = classifier.model
                input_shape = tuple(model.input_shape)
                output_shape = tuple(model.output_shape)
                if len(input_shape) != 4 or tuple(input_shape[1:]) != (224, 224, 3):
                    raise ValueError(f'Entrada no compatible: {input_shape}. Se esperaba 224x224x3.')
                if int(output_shape[-1]) != len(labels):
                    raise ValueError(f'El modelo tiene {output_shape[-1]} salidas pero {len(labels)} etiquetas.')
                detector = HandDetector(maxHands=1)
                self._classifier = classifier
                self._detector = detector
                self._labels = labels
            except Exception as exc:
                self._error = f'{type(exc).__name__}: {exc}'

    def status(self) -> dict:
        with self._guard:
            self._ensure_model()
            return {
                'loaded': self._classifier is not None and self._detector is not None,
                'message': 'Reconocimiento disponible.' if self._classifier is not None and self._detector is not None else self._error,
                'classes': list(self._labels),
                'model_filename': MODEL_FILE.name if MODEL_FILE.is_file() else None,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
            }

    def predict(self, frame_bytes: bytes) -> dict:
        with self._guard:
            self._ensure_model()
            if self._classifier is None or self._detector is None:
                raise RuntimeError(self._error or 'Reconocimiento no disponible.')

            buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('No se pudo decodificar la imagen enviada.')
            hand_image, state = hand_to_white_canvas(image, self._detector)
            if hand_image is None:
                return {
                    'class_name': None,
                    'confidence': None,
                    'accepted': False,
                    'state': state,
                    'predictions': {},
                }

            # cvzone gestiona el resize a 224 y la normalización Keras.
            raw_scores, index = self._classifier.getPrediction(hand_image, draw=False)
            scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
            if scores.size != len(self._labels) or not np.isfinite(scores).all():
                raise RuntimeError('La salida del modelo no es válida.')
            index = int(index)
            if not 0 <= index < len(self._labels):
                raise RuntimeError('El índice devuelto por el modelo no es válido.')
            label = self._labels[index]
            confidence = float(scores[index])
            accepted = label in ALLOWED_CLASSES and confidence >= CONFIDENCE_THRESHOLD
            return {
                'class_name': label,
                'confidence': round(confidence, 6),
                'accepted': accepted,
                'state': 'recognized' if accepted else 'uncertain',
                'predictions': {name: round(float(value), 6) for name, value in zip(self._labels, scores)},
            }


recognition = RecognitionService()
