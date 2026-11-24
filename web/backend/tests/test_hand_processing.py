from types import SimpleNamespace

import numpy as np

from app.services.classifier import RecognitionService
from app.services.hand_processing import (
    calculate_dynamic_frame_motion,
    extract_static_features,
)


def _results(points):
    landmarks = [
        SimpleNamespace(x=x, y=y, z=z)
        for x, y, z in points
    ]
    return SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
        multi_handedness=[object()],
    )


def test_static_features_match_mlp_input_contract():
    points = [(0.1 + index * 0.01, 0.2, 0.0) for index in range(21)]

    features, status = extract_static_features(_results(points), 640, 480)

    assert status == "ready"
    assert features.shape == (63,)
    assert features.dtype == np.float32
    assert np.isclose(features[0:3], 0).all()
    assert np.isclose(np.max(np.linalg.norm(features.reshape(21, 3), axis=1)), 1.0)


def test_static_features_reject_collapsed_hand():
    points = [(0.1, 0.2, 0.0)] * 21

    features, status = extract_static_features(_results(points), 640, 480)

    assert features is None
    assert status == "invalid_frame"


def test_static_features_report_missing_hand():
    results = SimpleNamespace(multi_hand_landmarks=[], multi_handedness=[])

    features, status = extract_static_features(results, 640, 480)

    assert features is None
    assert status == "no_hand"


def test_dynamic_frame_motion_reads_both_hand_velocity_blocks():
    keypoints = np.zeros(454, dtype=np.float32)
    keypoints[227 + 81:227 + 150] = 0.02
    keypoints[227 + 150:227 + 219] = 0.01

    assert np.isclose(calculate_dynamic_frame_motion(keypoints), 0.02)


def test_static_prediction_averages_five_frames_to_ignore_a_transient_label():
    points = [(0.1 + index * 0.01, 0.2, 0.0) for index in range(21)]
    results = _results(points)

    class Hands:
        def process(self, _image):
            return results

    class Output:
        def __init__(self, scores):
            self.scores = np.asarray([scores], dtype=np.float32)

        def numpy(self):
            return self.scores

    class Model:
        def __init__(self):
            self.outputs = iter([
                [0.90, 0.10],
                [0.90, 0.10],
                [0.90, 0.10],
                [0.90, 0.10],
                [0.10, 0.90],
            ])

        def __call__(self, _features, training=False):
            return Output(next(self.outputs))

    service = RecognitionService()
    service._hands = Hands()
    service._static_model = Model()
    service._static_labels = ["A", "B"]

    first = service._eval_static(None, 640, 480)
    for _ in range(3):
        service._eval_static(None, 640, 480)
    fifth = service._eval_static(None, 640, 480)

    assert first["accepted"] is True
    assert fifth["label"] == "A"
    assert np.isclose(fifth["confidence"], 0.74)
