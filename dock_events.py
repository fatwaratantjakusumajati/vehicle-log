"""Repeated cargo-score transition events for the legacy aw.py pipeline."""

import threading

import activity_estimator


_ANNOTATION_TTL_SECONDS = 8.0
_annotations = {}
_lock = threading.Lock()


def _process_vehicle(vehicle, timestamp, callback, flush=False):
    events = activity_estimator.poll_transitions(
        vehicle,
        timestamp,
        evidence_fn=lambda _start, _end: (0.0, 0.0),
        flush=flush,
        in_dock=True,
    )
    timeline = vehicle.setdefault("timeline_events", [])
    for event in events:
        event = dict(event)
        event["no"] = int(vehicle.get("_dock_event_no", 0)) + 1
        vehicle["_dock_event_no"] = event["no"]
        event.setdefault("area", "dock")
        event.setdefault("pelaku", "tidak terlihat")
        timeline.append(event)

        bbox = vehicle.get("last_bbox")
        if bbox:
            annotation = {
                "bbox": tuple(bbox),
                "label": "AKTIVITAS",
                "text": f"{event['activity']} #{event['no']}",
                "timestamp": float(event["t_end"]),
            }
            with _lock:
                _annotations.setdefault(vehicle.get("camera_id", ""), []).append(annotation)
        callback(vehicle, event)


def step(camera_id, roi_frame, tracks, now, callback):
    del roi_frame
    for vehicle in list(tracks.vehicles.values()):
        if vehicle.get("camera_id") != camera_id or vehicle.get("session_closed"):
            continue
        _process_vehicle(vehicle, float(now), callback)

    with _lock:
        recent = [
            item for item in _annotations.get(camera_id, [])
            if float(now) - item["timestamp"] <= _ANNOTATION_TTL_SECONDS
        ]
        _annotations[camera_id] = recent


def annotations(camera_id, now):
    with _lock:
        return [
            {key: value for key, value in item.items() if key != "timestamp"}
            for item in _annotations.get(camera_id, [])
            if float(now) - item["timestamp"] <= _ANNOTATION_TTL_SECONDS
        ]


def finalize_vehicle(camera_id, vehicle, timestamp, callback):
    if vehicle.get("camera_id") != camera_id:
        return
    _process_vehicle(vehicle, float(timestamp), callback, flush=True)


def summarize(vehicle):
    events = [
        event for event in vehicle.get("timeline_events", [])
        if isinstance(event, dict) and event.get("activity")
    ]
    if not events:
        return None, None

    loading = sum(event["activity"] == activity_estimator.LOADING for event in events)
    unloading = sum(event["activity"] == activity_estimator.UNLOADING for event in events)
    if loading and unloading:
        summary = f"Loading x{loading}, Unloading x{unloading}"
    elif loading:
        summary = f"Loading x{loading}"
    elif unloading:
        summary = f"Unloading x{unloading}"
    else:
        summary = events[-1]["activity"]

    confidences = [float(event.get("confidence", 0.0)) for event in events]
    info = {
        "activity": summary,
        "source": "dock_transitions",
        "confidence": round(sum(confidences) / len(confidences), 3),
        "reason": (
            f"Ringkasan {len(events)} perubahan skor muatan di dock "
            f"(Loading={loading}, Unloading={unloading})"
        ),
    }
    return summary, info
