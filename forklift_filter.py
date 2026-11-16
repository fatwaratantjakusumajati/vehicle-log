"""Forklift filters used by the legacy vehicle pipeline in aw.py."""

import os
import threading
from collections import defaultdict, deque

import cv2
import numpy as np


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_YOLO_PATH = os.environ.get("FORKLIFT_YOLO_PATH", "").strip() or os.path.join(
    _BASE_DIR, "forklift_openvino_model"
)
if not os.path.isabs(_YOLO_PATH):
    _YOLO_PATH = os.path.join(_BASE_DIR, _YOLO_PATH)

_YOLO_CONF = float(os.environ.get("FORKLIFT_YOLO_CONF", "0.35"))
_YOLO_IOU = float(os.environ.get("FORKLIFT_YOLO_IOU", "0.45"))
_YOLO_CONFIRM_HITS = max(1, int(os.environ.get("FORKLIFT_YOLO_CONFIRM_HITS", "2")))
_MODEL_THRESHOLD = float(os.environ.get("FORKLIFT_MODEL_THRESHOLD", "0.85"))
_COLOR_FILTER_ENABLED = os.environ.get(
    "FORKLIFT_COLOR_FILTER_ENABLED", "0"
).strip().lower() in ("1", "true", "yes", "on")
_ORANGE_THRESHOLD = float(os.environ.get("FORKLIFT_COLOR_MIN_PCT", "20"))
_COLOR_CONFIRM_HITS = max(1, int(os.environ.get("FORKLIFT_COLOR_CONFIRM_HITS", "3")))

_load_lock = threading.Lock()
_detector = None
_detector_kind = None
_load_attempted = False
_yolo_hits = defaultdict(int)
_model_votes = {}
_color_votes = {}
_state_lock = threading.Lock()


def _trim_state():
    if len(_yolo_hits) > 2000:
        for key in list(_yolo_hits)[:1000]:
            _yolo_hits.pop(key, None)
            _model_votes.pop(key, None)
            _color_votes.pop(key, None)


def yolo_available():
    if os.path.isdir(_YOLO_PATH):
        return any(name.lower().endswith(".xml") for name in os.listdir(_YOLO_PATH))
    return os.path.isfile(_YOLO_PATH)


def _load_yolo():
    """Load the configured forklift detector once; return None if unavailable."""
    global _detector, _detector_kind, _load_attempted
    if _load_attempted:
        return _detector

    with _load_lock:
        if _load_attempted:
            return _detector
        _load_attempted = True
        if not yolo_available():
            return None

        try:
            if os.path.isdir(_YOLO_PATH):
                from fk_openvino import DirectForklift

                _detector = DirectForklift(
                    _YOLO_PATH, conf=_YOLO_CONF, iou=_YOLO_IOU,
                    name_contains="forklift",
                )
                _detector_kind = "openvino"
            else:
                from ultralytics import YOLO

                _detector = YOLO(_YOLO_PATH, task="detect")
                _detector_kind = "ultralytics"
            print(f"[forklift_filter] Detektor dimuat dari '{_YOLO_PATH}'")
        except Exception as exc:
            print(f"[forklift_filter] Gagal memuat detektor '{_YOLO_PATH}': {exc}")
            _detector = None
            _detector_kind = None
        return _detector


def _detector_confidence(detector, crop):
    if _detector_kind == "openvino":
        detections = detector(crop, conf=_YOLO_CONF)
        return max((float(row[4]) for row in detections), default=0.0)

    results = detector.predict(crop, conf=_YOLO_CONF, iou=_YOLO_IOU, verbose=False)
    if not results:
        return 0.0
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or boxes.conf is None:
        return 0.0

    names = getattr(result, "names", {}) or {}
    confidences = []
    for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        label = names.get(int(class_id), "") if isinstance(names, dict) else ""
        if "forklift" in str(label).lower() or (not label and int(class_id) == 0):
            confidences.append(float(confidence))
    return max(confidences, default=0.0)


def is_forklift_by_detector(track_id, crop):
    detector = _load_yolo()
    if detector is None or crop is None or getattr(crop, "size", 0) == 0:
        return False, 0.0

    try:
        confidence = _detector_confidence(detector, crop)
    except Exception as exc:
        print(f"[forklift_filter] Inferensi detektor gagal untuk track {track_id}: {exc}")
        return False, 0.0

    key = str(track_id)
    with _state_lock:
        _trim_state()
        if confidence >= _YOLO_CONF:
            _yolo_hits[key] += 1
        else:
            _yolo_hits.pop(key, None)
    return confidence >= _YOLO_CONF, confidence


def yolo_is_locked(track_id):
    with _state_lock:
        return _yolo_hits.get(str(track_id), 0) >= _YOLO_CONFIRM_HITS


def is_forklift_by_model(track_id, crop, probability_fn):
    if crop is None or getattr(crop, "size", 0) == 0:
        return False, 0.0

    try:
        probability = probability_fn(crop)
    except Exception as exc:
        print(f"[forklift_filter] Klasifikasi model gagal untuk track {track_id}: {exc}")
        return False, 0.0
    if probability is None:
        return False, 0.0

    probability = float(probability)
    key = str(track_id)
    with _state_lock:
        _trim_state()
        votes = _model_votes.setdefault(key, deque(maxlen=3))
        votes.append(probability >= _MODEL_THRESHOLD)
        confirmed = sum(votes) >= 2
    return confirmed, probability


def _orange_percentage(crop):
    small = cv2.resize(crop, (96, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (3, 110, 100), (25, 255, 255))
    return float(np.count_nonzero(mask)) * 100.0 / mask.size


def is_available():
    return _COLOR_FILTER_ENABLED


def is_forklift(crop):
    if not _COLOR_FILTER_ENABLED or crop is None or getattr(crop, "size", 0) == 0:
        return False, 0.0
    percentage = _orange_percentage(crop)
    return percentage >= _ORANGE_THRESHOLD, percentage


def is_forklift_tracked(track_id, crop):
    detected, score = is_forklift(crop)
    if not _COLOR_FILTER_ENABLED:
        return False, score

    key = str(track_id)
    with _state_lock:
        _trim_state()
        votes = _color_votes.setdefault(key, deque(maxlen=3))
        votes.append(detected)
        confirmed = sum(votes) >= _COLOR_CONFIRM_HITS
    return confirmed, score
