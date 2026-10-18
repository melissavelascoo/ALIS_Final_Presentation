"""Pruebas de recorte que no necesitan cargar TensorFlow ni MediaPipe."""
import numpy as np

from app.services.hand_processing import hand_to_white_canvas


class FakeDetector:
    def __init__(self, bbox=None):
        self.bbox = bbox

    def findHands(self, image):
        return ([{'bbox': self.bbox}] if self.bbox else [], image)


def test_centered_hand_returns_300_square():
    frame = np.zeros((480, 640, 3), np.uint8)
    cropped, status = hand_to_white_canvas(frame, FakeDetector((200, 100, 100, 220)))
    assert status == 'ready'
    assert cropped.shape == (300, 300, 3)
    assert cropped.dtype == np.uint8
    assert cropped[0, 0].tolist() == [255, 255, 255]


def test_no_hand():
    frame = np.zeros((480, 640, 3), np.uint8)
    cropped, status = hand_to_white_canvas(frame, FakeDetector())
    assert cropped is None and status == 'no_hand'


def test_hand_near_border():
    frame = np.zeros((480, 640, 3), np.uint8)
    cropped, status = hand_to_white_canvas(frame, FakeDetector((1, 100, 100, 180)))
    assert cropped is None and status == 'hand_out_of_frame'
