"""Preprocesamiento de fotogramas con el recorte utilizado en HandSignDetector.

Se conserva el dibujo de MediaPipe para mantener la entrada similar al dataset.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.core.config import HAND_IMAGE_SIZE, HAND_OFFSET


def hand_to_white_canvas(image: np.ndarray, detector) -> tuple[np.ndarray | None, str]:
    """Prepara la mano en un lienzo blanco o devuelve el estado del recorte."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return None, 'invalid_frame'

    height, width = image.shape[:2]

    hands, marked_frame = detector.findHands(image.copy())
    if not hands:
        return None, 'no_hand'

    x, y, w, h = (int(value) for value in hands[0]['bbox'])
    if w <= 0 or h <= 0:
        return None, 'no_hand'

    # Evita recortes inválidos cuando la mano sale del encuadre.
    off = HAND_OFFSET
    if x - off < 0 or y - off < 0 or x + w + off > width or y + h + off > height:
        return None, 'hand_out_of_frame'

    crop = marked_frame[y - off : y + h + off, x - off : x + w + off]
    if crop.size == 0:
        return None, 'hand_out_of_frame'

    output = np.ones((HAND_IMAGE_SIZE, HAND_IMAGE_SIZE, 3), dtype=np.uint8) * 255
    ratio = h / w
    try:
        if ratio > 1:

            target_w = math.ceil((HAND_IMAGE_SIZE / h) * w)
            target_w = min(HAND_IMAGE_SIZE, max(1, target_w))
            resized = cv2.resize(crop, (target_w, HAND_IMAGE_SIZE))
            left = math.ceil((HAND_IMAGE_SIZE - target_w) / 2)
            output[:, left:left + target_w] = resized
        else:
            target_h = math.ceil((HAND_IMAGE_SIZE / w) * h)
            target_h = min(HAND_IMAGE_SIZE, max(1, target_h))
            resized = cv2.resize(crop, (HAND_IMAGE_SIZE, target_h))
            top = math.ceil((HAND_IMAGE_SIZE - target_h) / 2)
            output[top:top + target_h, :] = resized
    except cv2.error:
        return None, 'invalid_frame'
    return output, 'ready'
