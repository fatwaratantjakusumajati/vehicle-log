# aw.py - deteksi kendaraan dual-camera (cam1 Loading Kiri, cam2 Halaman Depan)
# [CAM1] cam1 dijalankan oleh cam1.py (camera_loop langsung memanggil cam1.run_forever()). cam2 memakai kode di file ini.
# CAM1 memakai pipeline cam1.py; CAM2 memakai pipeline aw.py.
# Pipeline dan konfigurasi CAM2 di bawah tetap memakai implementasi aw.py yang ada.

import base64
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

# [DEPLOY-CWD] Semua path relatif (model, ocsort_custom.yaml, pkl, folder logs/ dan debug) dihitung dari folder
# aw.py, bukan dari folder tempat proses kebetulan dijalankan (Task Scheduler / service / shortcut).
# Hanya saat dijalankan langsung; video.py yang `import aw` tidak terpengaruh.
if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except Exception as _e:
        print(f"[WARN] gagal pindah ke folder aw.py: {_e}")

_ENV_PATH_EARLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH_EARLY)

try:
    import cargo_classifier
except Exception as _e:
    cargo_classifier = None
    print(f"[INFO] cargo_classifier ML tidak tersedia ({_e}), pakai deteksi rule-based.")

try:
    import forklift_filter
except Exception as _e:
    forklift_filter = None
    print(f"[INFO] forklift_filter tidak tersedia ({_e}), filter forklift dimatikan.")

try:
    import dock_events
except Exception as _e:
    dock_events = None
    print(f"[INFO] dock_events tidak tersedia ({_e}), kejadian beruntun per sesi dimatikan.")

try:
    import activity_estimator
except Exception as _e:
    activity_estimator = None
    print(f"[INFO] activity_estimator tidak tersedia ({_e}), pakai logika lama.")

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)

_MISSING_ENV = []
_N8N_THREADS = []
_N8N_THREADS_LOCK = threading.Lock()


def env(key, cast=str, required=True):
    val = os.getenv(key)
    if val is None or val == "":
        if required:
            _MISSING_ENV.append(key)
        return None
    if cast is bool:
        return val.strip().lower() in ("1", "true", "yes", "on")
    try:
        return cast(val)
    except (ValueError, TypeError):
        _MISSING_ENV.append(f"{key} (format tidak valid, harus {cast.__name__})")
        return None


def _mask_url(text):
    """Samarkan password di URL (rtsp://user:PASS@host -> rtsp://user:***@host) sebelum dicetak ke log."""
    try:
        return re.sub(r"(://[^:/@\s]+:)[^@\s]+@", r"\1***@", str(text))
    except Exception:
        return "<url disamarkan>"


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5|stimeout;5000000"
)

N8N_WEBHOOK = env("N8N_WEBHOOK_URL")
N8N_TIMEOUT_SECONDS = env("N8N_TIMEOUT_SECONDS", float, required=False) or 10.0
N8N_RETRY_ATTEMPTS = env("N8N_RETRY_ATTEMPTS", int, required=False) or 3
N8N_RETRY_DELAY_SECONDS = env("N8N_RETRY_DELAY_SECONDS", float, required=False) or 3.0
N8N_OUTBOX_PATH = env("N8N_OUTBOX_PATH", required=False) or "n8n_outbox.jsonl"
N8N_OUTBOX_MAX_MB = env("N8N_OUTBOX_MAX_MB", float, required=False) or 500.0
N8N_OUTBOX_INTERVAL_SECONDS = env("N8N_OUTBOX_INTERVAL_SECONDS", float, required=False) or 30.0
N8N_OUTBOX_MAX_TRIES = env("N8N_OUTBOX_MAX_TRIES", int, required=False) or 20
MODEL_PATH = env("MODEL_PATH")

CAM1_MODEL_PATH = env("CAM1_MODEL_PATH", required=False) or MODEL_PATH
CAM2_MODEL_PATH = env("CAM2_MODEL_PATH", required=False) or MODEL_PATH

CAM_USER = env("CAMERA_USER")
CAM_PASS = env("CAMERA_PASS")
CAM_HOST = env("CAMERA_HOST")
CAM_PORT = env("CAMERA_PORT")
CAM1_CHANNEL = env("CAM1_CHANNEL")
CAM2_CHANNEL = env("CAM2_CHANNEL")

_CAM1_RTSP_OVERRIDE = env("CAM1_RTSP_URL", required=False)
_CAM2_RTSP_OVERRIDE = env("CAM2_RTSP_URL", required=False)
_CAM1_VIDEO_SOURCE = env("CAM1_VIDEO_SOURCE", required=False)
_CAM2_VIDEO_SOURCE = env("CAM2_VIDEO_SOURCE", required=False)


def _build_rtsp_url(override, channel):
    if override:
        return override
    return f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_HOST}:{CAM_PORT}/Streaming/Channels/{channel}"


CAM1_RTSP_URL = _build_rtsp_url(_CAM1_RTSP_OVERRIDE, CAM1_CHANNEL)
CAM2_RTSP_URL = _build_rtsp_url(_CAM2_RTSP_OVERRIDE, CAM2_CHANNEL)
RTSP_SNAPSHOT_WARMUP_FRAMES = env("RTSP_SNAPSHOT_WARMUP_FRAMES", int, required=False) or 8

CAM2_PLATE_SNAPSHOT_CHANNEL = env("CAM2_PLATE_SNAPSHOT_CHANNEL", required=False)

CAM1_PLATE_SNAPSHOT_CHANNEL = env("CAM1_PLATE_SNAPSHOT_CHANNEL", required=False)

PLATE_SNAPSHOT_MIN_AREA_RATIO = env(
    "PLATE_SNAPSHOT_MIN_AREA_RATIO", float, required=False
) or 0.008
PLATE_SNAPSHOT_INTERVAL_SECONDS = env(
    "PLATE_SNAPSHOT_INTERVAL_SECONDS", float, required=False
) or 2.0
PLATE_SNAPSHOT_MAX_PER_VEHICLE = env(
    "PLATE_SNAPSHOT_MAX_PER_VEHICLE", int, required=False
) or 5
PLATE_SNAPSHOT_EXTENDED_INTERVAL_SECONDS = env(
    "PLATE_SNAPSHOT_EXTENDED_INTERVAL_SECONDS", float, required=False
) or 10.0
PLATE_SNAPSHOT_MAX_TOTAL_ATTEMPTS = env(
    "PLATE_SNAPSHOT_MAX_TOTAL_ATTEMPTS", int, required=False
) or 20

CONFIDENCE_MIN = env("CONFIDENCE_MIN", float)
CAM1_CONFIDENCE_MIN = env("CAM1_CONFIDENCE_MIN", float, required=False) or CONFIDENCE_MIN
CAM2_CONFIDENCE_MIN = env("CAM2_CONFIDENCE_MIN", float, required=False) or CONFIDENCE_MIN

CAM1_TRUCK_MIN_CONFIDENCE = env(
    "CAM1_TRUCK_MIN_CONFIDENCE", float, required=False
) or 0.0
IOU_THRESHOLD = env("IOU_THRESHOLD", float)

MIN_AREA_NOISE = 0

# [DEPLOY] Default MATI (dulu hardcoded True -> cetak baris SKIP per frame). Nyalakan: DEBUG_MODE=true di .env
DEBUG_MODE = bool(env("DEBUG_MODE", bool, required=False))
DEBUG_LOG_EVERY_N_FRAMES = 30
SAVE_DEBUG_ROI_FRAMES = 3
SAVE_DEBUG_IMAGES = False
SNAPSHOT_INTERVAL_SECONDS = 300

FRAME_SKIP = 3

CAM1_FRAME_SKIP = env("CAM1_FRAME_SKIP", int, required=False) or FRAME_SKIP
CAM2_FRAME_SKIP = env("CAM2_FRAME_SKIP", int, required=False) or FRAME_SKIP

_LAST_SNAPSHOT_TIME = {}

CONFIRM_SECONDS = env("CONFIRM_SECONDS", float)

ENTRY_MAX_EXTRA_WAIT_SECONDS = env("ENTRY_MAX_EXTRA_WAIT_SECONDS", float, required=False) or 150.0
LOST_SECONDS = env("LOST_SECONDS", float)

STATIONARY_LOST_SECONDS = env("STATIONARY_LOST_SECONDS", float, required=False) or 150.0

REVISIT_RETENTION_SECONDS = env("REVISIT_RETENTION_SECONDS", float)

MIN_GENUINE_REVISIT_GAP_SECONDS = env("MIN_GENUINE_REVISIT_GAP_SECONDS", float, required=False) or 120.0
REVISIT_MIN_APPEARANCE_SCORE = env("REVISIT_MIN_APPEARANCE_SCORE", float)
TRACK_RELINK_GRACE_SECONDS = env("TRACK_RELINK_GRACE_SECONDS", float)

SINGLE_LANE_MERGE_MAX_GAP_SECONDS = env(
    "SINGLE_LANE_MERGE_MAX_GAP_SECONDS", float, required=False
) or LOST_SECONDS

SESSION_HARD_MAX_SECONDS = env(
    "SESSION_HARD_MAX_SECONDS", float, required=False
) or 3600.0

CAM1_SESSION_HARD_MAX_SECONDS = env(
    "CAM1_SESSION_HARD_MAX_SECONDS", float, required=False
) or SESSION_HARD_MAX_SECONDS
CAM2_SESSION_HARD_MAX_SECONDS = env(
    "CAM2_SESSION_HARD_MAX_SECONDS", float, required=False
) or SESSION_HARD_MAX_SECONDS

STALE_RELINK_GRACE_SECONDS = env(
    "STALE_RELINK_GRACE_SECONDS", float, required=False
) or 15.0

REVISIT_MAX_DIST_CAP_PIXELS = env(
    "REVISIT_MAX_DIST_CAP_PIXELS", float, required=False
) or 500.0

TRACK_MATCH_MAX_CENTER_RATIO = env("TRACK_MATCH_MAX_CENTER_RATIO", float)
TRACK_MATCH_MIN_SIZE_RATIO = env("TRACK_MATCH_MIN_SIZE_RATIO", float)
TRACK_MATCH_MAX_SIZE_RATIO = env("TRACK_MATCH_MAX_SIZE_RATIO", float)
TRACK_MATCH_MIN_IOU = env("TRACK_MATCH_MIN_IOU", float)
MAX_TRACK_ALIASES = env("MAX_TRACK_ALIASES", int)

TRACK_MATCH_MAX_DIST_PIXELS = env(
    "TRACK_MATCH_MAX_DIST_PIXELS", float, required=False
) or 400.0
SINGLE_LANE_TRACK_MATCH_MAX_DIST_PIXELS = env(
    "SINGLE_LANE_TRACK_MATCH_MAX_DIST_PIXELS", float, required=False
) or 1600.0

SHADOW_FILTER_ENABLE = env("SHADOW_FILTER_ENABLE", bool, required=False)
if SHADOW_FILTER_ENABLE is None:
    SHADOW_FILTER_ENABLE = True

SHADOW_FILTER_CONF_GATE = env("SHADOW_FILTER_CONF_GATE", float, required=False)
if SHADOW_FILTER_CONF_GATE is None:
    SHADOW_FILTER_CONF_GATE = 0.55

SHADOW_MAX_VALUE = env("SHADOW_MAX_VALUE", float, required=False)
if SHADOW_MAX_VALUE is None:
    SHADOW_MAX_VALUE = 70.0

SHADOW_MAX_SATURATION = env("SHADOW_MAX_SATURATION", float, required=False)
if SHADOW_MAX_SATURATION is None:
    SHADOW_MAX_SATURATION = 35.0

SHADOW_MAX_EDGE_DENSITY = env("SHADOW_MAX_EDGE_DENSITY", float, required=False)
if SHADOW_MAX_EDGE_DENSITY is None:
    SHADOW_MAX_EDGE_DENSITY = 0.03

SHADOW_MAX_TEXTURE_VAR = env("SHADOW_MAX_TEXTURE_VAR", float, required=False)
if SHADOW_MAX_TEXTURE_VAR is None:
    SHADOW_MAX_TEXTURE_VAR = 80.0

if _MISSING_ENV:
    print("!! FATAL: variabel berikut belum diisi / salah format di .env:")
    for k in _MISSING_ENV:
        print(f"   - {k}")
    print(f"!! Cek file .env di: {_ENV_PATH}")
    print("!! Lihat .env.example untuk daftar lengkap & formatnya.")
    sys.exit(1)

INFER_TARGET_WIDTH = 960
TRACKER_CONFIG = "ocsort_custom.yaml"

# [NO-FILTER] Filter peningkat tampilan gambar (CLAHE dan upscale crop) DIHAPUS: gambar dipakai apa adanya.
SAVE_JPEG_QUALITY = 95
_FRAME_COUNTERS = {}

DIRECTION_SCORE_THRESHOLD = 0.15
DIRECTION_WINDOW_SECONDS = 4.0

# [FIX-PARKIR-ARAH] Mobil parkir yang kotaknya bergoyang terbaca "KELUAR" hanya dari perubahan ukuran kotak -> sesi ditutup cepat (LOST_SECONDS)
# -> KELUAR palsu lalu MASUK lagi. Arah (selain cam1) kini butuh perpindahan pusat >= DIRECTION_MIN_DISPLACEMENT x ukuran kotak. 0 = perilaku lama.
DIRECTION_MIN_DISPLACEMENT = float(os.environ.get("DIRECTION_MIN_DISPLACEMENT", "0.35"))
# Kendaraan yang DIAM saat hilang: KELUAR ditahan PARKED_HOLD_SECONDS dan dibatalkan bila muncul lagi di tempat yang sama (IoU >= SAME_SPOT_MIN_IOU).
PARKED_HOLD_SECONDS = float(os.environ.get("PARKED_HOLD_SECONDS", "1800"))
SAME_SPOT_MIN_IOU = float(os.environ.get("SAME_SPOT_MIN_IOU", "0.5"))
# [CAM2-FRAGMEN] Kotak kecil yang SELURUHNYA berada di dalam kotak mobil parkir (sesi terbuka, atau baru tertutup tapi diam)
# adalah pecahan mobil itu (atap/bagian terdeteksi sendiri), bukan mobil lain -> JANGAN buka sesi baru (cegah MASUK ganda).
# Kasus: 19 Nov, sesi ...000005 kotak (219,327,691,453) berada di dalam kotak mobil ...000001 (215,329,912,1077).
# CAM2_FRAGMENT_GUARD=0 untuk mematikan. Cukup ketat agar mobil tetangga sungguhan tidak ikut terbuang.
CAM2_FRAGMENT_GUARD = int(os.environ.get("CAM2_FRAGMENT_GUARD", "1"))
CAM2_FRAGMENT_MIN_CONTAIN = float(os.environ.get("CAM2_FRAGMENT_MIN_CONTAIN", "0.85"))   # bagian kotak baru yang berada di dalam kotak lama
CAM2_FRAGMENT_MAX_AREA_RATIO = float(os.environ.get("CAM2_FRAGMENT_MAX_AREA_RATIO", "0.5"))  # luas kotak baru <= ini x luas kotak lama
REVISIT_RETENTION_EFF = max(REVISIT_RETENTION_SECONDS, PARKED_HOLD_SECONDS)
# [FIX-FOTO-PLAT] foto untuk kamera di PHOTO_PLATE_PRIORITY_CAMS: utamakan crop yang plat-nya terlihat.
PHOTO_FRONTAL_WEIGHT = float(os.environ.get("PHOTO_FRONTAL_WEIGHT", "1.0"))
PHOTO_PLATE_LIK_WEIGHT = float(os.environ.get("PHOTO_PLATE_LIK_WEIGHT", "1.5"))
_PHOTO_PLATE_CAMS = {s.strip().lower() for s in os.environ.get("PHOTO_PLATE_PRIORITY_CAMS", "cam2").split(",") if s.strip()}

STATIC_FILTER_ENABLED = env("STATIC_FILTER_ENABLED", bool, required=False)
if STATIC_FILTER_ENABLED is None:
    STATIC_FILTER_ENABLED = True
STATIC_FILTER_MIN_SAMPLES = 5
STATIC_FILTER_MIN_TIMESPAN = env("STATIC_FILTER_MIN_TIMESPAN", float, required=False) or 1.5
STATIC_FILTER_POS_RATIO = env("STATIC_FILTER_POS_RATIO", float, required=False) or 0.03
STATIC_FILTER_SIZE_RATIO = env("STATIC_FILTER_SIZE_RATIO", float, required=False) or 1.05


def _looks_like_static_fixture(vehicle):
    if not STATIC_FILTER_ENABLED:
        return False

    history = vehicle.get("direction_history") or []
    if len(history) < STATIC_FILTER_MIN_SAMPLES:
        return False

    t_first, t_last = history[0][0], history[-1][0]
    if t_last - t_first < STATIC_FILTER_MIN_TIMESPAN:
        return False

    cxs, cys, areas, sizes = [], [], [], []
    for _, (x1, y1, x2, y2) in history:
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cxs.append((x1 + x2) / 2.0)
        cys.append((y1 + y2) / 2.0)
        areas.append(w * h)
        sizes.append(max(w, h))

    avg_size = sum(sizes) / len(sizes)
    pos_spread = ((max(cxs) - min(cxs)) ** 2 + (max(cys) - min(cys)) ** 2) ** 0.5
    pos_spread_ratio = (pos_spread / avg_size) if avg_size else 0.0
    size_ratio = (max(areas) / min(areas)) if min(areas) > 0 else 1.0

    return (
        pos_spread_ratio < STATIC_FILTER_POS_RATIO
        and size_ratio < STATIC_FILTER_SIZE_RATIO
    )
DIRECTION_MIN_TIME_SPAN = 0.5

CAMERAS = {
    "cam1": {
        "kamera_id": "cam1",
        "kamera_nama": "Loading Kiri",
        "video_source": _CAM1_VIDEO_SOURCE or CAM1_RTSP_URL,
        "model_path": CAM1_MODEL_PATH,
        "confidence_min": CAM1_CONFIDENCE_MIN,
        "frame_skip": CAM1_FRAME_SKIP,
        "roi": {"x_min": 0.15, "x_max": 0.890, "y_min": 0.155, "y_max": 0.955},
        # [FIX-EXCLUDE-V2] Koordinat dinormalisasi terhadap ROI; yang dicek adalah TITIK TENGAH kotak kendaraan.
        #  Z1  ram kuning kiri bawah (tetap dikecualikan).
        #  Z2  (dihapus) ram/pintu dock kanan = tempat proses loading, harus terlacak agar status Loading terbaca.
        #  Z3a barikade kuning + tumpukan terpal hitam (x 0.50-0.62, y 0-0.19)
        #  Z3b tumpukan terpal biru + parkiran mobil sampai tepi kanan ROI (x 0.62-1.0, y 0-0.36)
        #  Z4  (dihapus)
        "exclude_zones": [
            {"x_min": 0.0, "x_max": 0.50, "y_min": 0.51, "y_max": 1.0},
            {"x_min": 0.50, "x_max": 0.62, "y_min": 0.0, "y_max": 0.19},
            {"x_min": 0.62, "x_max": 1.0, "y_min": 0.0, "y_max": 0.36},
        ],
        "classes_allowed": [2, 7],
        "truck_class_id": 7,
        # [FIX-PICKUP] Pickup/van bak terbuka dideteksi YOLO COCO sebagai 'car' (2), bukan 'truck' (7).
        # Sebelumnya sampling muatan HANYA jalan untuk cls 7, jadi pickup tidak pernah disampel
        # (muatan '-', forklift dekat 0s, aktivitas 'Tidak Dapat Dipastikan').
        "cargo_classes": [2, 7],
        # Label jenis generik HANYA untuk cam1 (area loading). cam2 tidak diubah (tetap Truk/Mobil seperti semula).
        # Kosongkan VEHICLE_LABEL di .env untuk mengembalikan cam1 ke Truk/Mobil.
        "vehicle_label": os.environ.get("VEHICLE_LABEL", "Kendaraan").strip(),
        "truck_min_confidence": CAM1_TRUCK_MIN_CONFIDENCE,
        "forklift_gate": True,
        "dock_events": True,
        "single_lane_mode": True,
        "line_position_ratio": 0.55,
        "cargo_detection": True,
        "plate_detection": False,
        "read_own_plate": False,
        "plate_snapshot_channel": CAM1_PLATE_SNAPSHOT_CHANNEL,
        "direction_axis": "y",
        "direction_sign_masuk": 1,
        "direction_size_trend_masuk": "growing",
        "min_confirm_crop_area": 45000,
        "session_hard_max_seconds": CAM1_SESSION_HARD_MAX_SECONDS,
    },
    "cam2": {
        "kamera_id": "cam2",
        "kamera_nama": "Halaman Depan",
        "video_source": _CAM2_VIDEO_SOURCE or CAM2_RTSP_URL,
        "model_path": CAM2_MODEL_PATH,
        "confidence_min": CAM2_CONFIDENCE_MIN,
        "frame_skip": CAM2_FRAME_SKIP,
        "roi": {"x_min": 0.5078, "x_max": 0.975, "y_min": 0.12, "y_max": 0.9972},
        "exclude_zones": [
            {"x_min": 0.84, "x_max": 1.0, "y_min": 0.0, "y_max": 0.06},
        ],
        # CAM2 hanya mencatat mobil (kelas COCO 2); kelas truk (7) tidak diizinkan.
        "classes_allowed": [2],
        "truck_class_id": 7,
        "single_lane_mode": False,
        "line_position_ratio": 0.55,
        "cargo_detection": False,
        "plate_detection": False,
        "plate_snapshot_channel": CAM2_PLATE_SNAPSHOT_CHANNEL,
        "min_confirm_crop_area": 3000,
        "direction_axis": "x",
        "direction_sign_masuk": -1,
        "direction_size_trend_masuk": "growing",
        "session_hard_max_seconds": CAM2_SESSION_HARD_MAX_SECONDS,
        # [FIX-CAM2-PARKIR] cam2 hanya mencatat kendaraan yang benar-benar PARKIR: harus diam berturut-turut >= park_confirm_seconds
        # sebelum MASUK dikirim. Mobil yang cuma lewat (mis. 3 dtk terlihat -> MASUK, 7 dtk kemudian KELUAR) tidak dicatat.
        # CAM2_PARK_CONFIRM_SECONDS=0 mematikan aturan ini (perilaku lama).
        "park_confirm_seconds": float(os.environ.get("CAM2_PARK_CONFIRM_SECONDS", "10") or 0),
    },
}

CARGO_LOADED = "Bermuatan"
CARGO_EMPTY = "Kosong"
CARGO_UNKNOWN = "Tidak dapat dipastikan"
CARGO_HISTORY_SIZE = 5
CARGO_AGREEMENT_RATIO = env("CARGO_AGREEMENT_RATIO", float, required=False) or 0.7
CARGO_MIN_HISTORY = env("CARGO_MIN_HISTORY", int, required=False) or 4
CARGO_PERIODIC_CHECK_INTERVAL_SECONDS = env(
    "CARGO_PERIODIC_CHECK_INTERVAL_SECONDS", float, required=False
) or 2.0
CARGO_ASPECT_RATIO_MAX = env("CARGO_ASPECT_RATIO_MAX", float, required=False) or 2.0
CARGO_ASPECT_RATIO_RELAXED_MAX = env(
    "CARGO_ASPECT_RATIO_RELAXED_MAX", float, required=False
) or 2.6
CARGO_ASPECT_RELAX_SECONDS = env(
    "CARGO_ASPECT_RELAX_SECONDS", float, required=False
) or 15.0
CARGO_MIN_AREA_RATIO = (
    env("CARGO_MIN_AREA_RATIO", float, required=False)
    or env("CARGO_MIN_AREA", float, required=False)
    or 0.02
)
CARGO_MIN_CROP_SIDE = env("CARGO_MIN_CROP_SIDE", int, required=False) or 120
CARGO_MIN_CROP_PIXELS = env("CARGO_MIN_CROP_PIXELS", int, required=False) or 20000
CARGO_RULE_BASED_FALLBACK = env("CARGO_RULE_BASED_FALLBACK", bool, required=False)
if CARGO_RULE_BASED_FALLBACK is None:
    CARGO_RULE_BASED_FALLBACK = False

CARGO_DISABLE_BED_CROP = env("CARGO_DISABLE_BED_CROP", bool, required=False)
if CARGO_DISABLE_BED_CROP is None:
    CARGO_DISABLE_BED_CROP = False

CARGO_ENTRY_CROPS_MAX = env("CARGO_ENTRY_CROPS_MAX", int, required=False) or 5
CARGO_MIN_AVG_CONFIDENCE = env("CARGO_MIN_AVG_CONFIDENCE", float, required=False) or 0.70
CARGO_MOMENT_MIN_AGREEMENT = env(
    "CARGO_MOMENT_MIN_AGREEMENT", float, required=False
) or 0.8

CARGO_MOMENT_MIN_VOTES = env("CARGO_MOMENT_MIN_VOTES", int, required=False) or 3

CARGO_BOX_GATE_ENABLED = env("CARGO_BOX_GATE_ENABLED", bool, required=False)
if CARGO_BOX_GATE_ENABLED is None:
    CARGO_BOX_GATE_ENABLED = True

CARGO_BOX_CONFIRM_HITS = env("CARGO_BOX_CONFIRM_HITS", int, required=False) or 3

ACTIVITY_MODE = env("ACTIVITY_MODE", bool, required=False)
if ACTIVITY_MODE is None:
    ACTIVITY_MODE = True
ACTIVITY_MODE = bool(ACTIVITY_MODE and activity_estimator is not None)
ACTIVITY_PENDING_LABEL = "-"

ENTRY_STATIONARY_CONFIRM_SECONDS = env(
    "ENTRY_STATIONARY_CONFIRM_SECONDS", float, required=False
) or 5.0

MIN_STATIONARY_CONFIRM_AREA_RATIO = env(
    "MIN_STATIONARY_CONFIRM_AREA_RATIO", float, required=False
) or 0.02

STATIONARY_MOTION_RATIO_MAX = env(
    "STATIONARY_MOTION_RATIO_MAX", float, required=False
) or 0.04
STATIONARY_PHOTO_MIN_SECONDS = env(
    "STATIONARY_PHOTO_MIN_SECONDS", float, required=False
) or 1.0
STATIONARY_PHOTO_BONUS = env(
    "STATIONARY_PHOTO_BONUS", float, required=False
) or 1.6

CLS_VOTE_MIN_DOMINANCE = env("CLS_VOTE_MIN_DOMINANCE", float, required=False) or 0.6


def _cls_vote_is_decisive(vehicle, config=None):
    if config and config.get("vehicle_label"):
        return True  # jenis tidak dipakai di kamera ini, jangan tunda MASUK karena vote car/truck belum dominan
    weights = vehicle.get("cls_weights") or {}
    total = sum(weights.values())
    if total <= 0:
        return False
    return (max(weights.values()) / total) >= CLS_VOTE_MIN_DOMINANCE


def _weighted_cls(vehicle, default_cls=None):
    weights = vehicle.get("cls_weights") or {}
    if weights:
        return max(weights, key=weights.get)
    votes = vehicle.get("cls_votes") or {}
    if votes:
        return max(votes, key=votes.get)
    return default_cls


_MATCH_TOLERANT_CLASS_PAIRS = {frozenset({2, 7})}


def is_local_video_file(source: str) -> bool:
    return not str(source).lower().startswith(("rtsp://", "http://", "https://"))


_NO_PHOTO_B64 = None


def _no_photo_b64():
    """Gambar placeholder 'TIDAK ADA FOTO' supaya event tetap terkirim bila tidak ada crop (jangan crash)."""
    global _NO_PHOTO_B64
    if _NO_PHOTO_B64 is None:
        img = np.full((96, 160, 3), 60, dtype=np.uint8)
        cv2.putText(img, "TIDAK ADA FOTO", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img)
        _NO_PHOTO_B64 = base64.b64encode(buf).decode("utf-8") if ok else ""
    return _NO_PHOTO_B64


def encode_image_base64(img, quality=SAVE_JPEG_QUALITY):
    # [FIX-NONE-CROP] img bisa None (tidak ada crop usable) -> dulu cv2.imencode melempar error dan menjatuhkan loop kamera.
    if img is None or getattr(img, "size", 0) == 0:
        return _no_photo_b64()
    try:
        # [FIX-PLAT-FOTO] upscale ringan untuk crop kecil agar OCR plat (Gemini) lebih terbantu. Nonaktifkan: PHOTO_UPSCALE_MIN_SIDE=0
        _min_side = int(float(os.environ.get("PHOTO_UPSCALE_MIN_SIDE", "700")))
        if _min_side > 0 and max(img.shape[:2]) < _min_side:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except Exception:
        return _no_photo_b64()
    return base64.b64encode(buf).decode("utf-8") if ok else _no_photo_b64()


def prepare_roi_for_inference(roi_frame, target_width=INFER_TARGET_WIDTH):
    h, w = roi_frame.shape[:2]
    if w == 0 or h == 0:
        return roi_frame, 1.0
    scale = target_width / w
    if abs(scale - 1.0) < 0.02:
        return roi_frame, 1.0
    resized = cv2.resize(roi_frame, (target_width, int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    return resized, w / target_width


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_size(bbox):
    return (max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1]))


def bbox_iou(a, b):
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_compatible(old_cls, new_cls):
    if old_cls == new_cls:
        return True
    return frozenset({old_cls, new_cls}) in _MATCH_TOLERANT_CLASS_PAIRS


def _bbox_iou(a, b):
    try:
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    except Exception:
        return 0.0


def _view_factor_bbox(bbox):
    """Kotak hampir persegi = mobil menghadap kamera (plat terlihat); kotak sangat lebar = tampak samping."""
    if PHOTO_FRONTAL_WEIGHT <= 0 or not bbox:
        return 1.0
    try:
        asp = max(1.0, bbox[2] - bbox[0]) / max(1.0, bbox[3] - bbox[1])
    except Exception:
        return 1.0
    f = 1.0 if asp <= 1.5 else max(0.45, 1.0 - 0.55 * (asp - 1.5))
    return f ** PHOTO_FRONTAL_WEIGHT


def _plate_likelihood(crop):
    """0..1: ada tidaknya persegi panjang mirip plat pada crop. Heuristik klasik, hanya untuk MEMILIH foto. Gagal -> 0."""
    try:
        if crop is None or crop.size == 0:
            return 0.0
        h, w = crop.shape[:2]
        if h < 40 or w < 40:
            return 0.0
        m = max(h, w)
        if m > 480:
            s = 480.0 / m
            crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sob = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
        _, bw = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kw = max(7, int(w * 0.04))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3)))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = 0.0
        img_area = float(w * h)
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch < 6 or cw < 18:
                continue
            ar = cw / float(ch)
            frac = (cw * ch) / img_area
            if not (2.0 <= ar <= 6.0) or not (0.003 <= frac <= 0.08):
                continue
            if cv2.contourArea(c) / float(cw * ch) < 0.5:
                continue
            dens = float(np.mean(bw[y:y + ch, x:x + cw] > 0))
            if dens < 0.25:
                continue
            best = max(best, 0.5 + 0.5 * min(1.0, (dens - 0.25) / 0.35))
        return float(best)
    except Exception:
        return 0.0


def _plate_priority(vehicle):
    vid = str(vehicle.get("vehicle_id", "")).lower()
    return any(vid.startswith(c) for c in _PHOTO_PLATE_CAMS)


def _stationary_bonus_factor(item):
    return STATIONARY_PHOTO_BONUS if item.get("is_stationary") else 1.0


# [FIX-DISPLAY-FORKLIFT] Foto MASUK/KELUAR di dashboard dulu bisa berisi FORKLIFT: crop forklift yang ikut tergabung ke
# sesi truk (RE-LINK) berukuran besar sehingga menang skor luas. Pada kasus nyata crop forklift itu hanya ~2% piksel oranye
# (di bawah ambang 5% yang lama), sedangkan crop truk 0%. Pemilihan foto sekarang: utamakan crop dengan oranye < DISPLAY_ORANGE_MAX_PCT;
# kalau semuanya beroranye, pakai foto saat MASUK (diambil sebelum forklift datang); kalau itu tak ada, pakai crop yang paling
# sedikit oranyenya. TIDAK PERNAH mengembalikan None bila ada crop (detektor YOLO forklift sengaja tidak dipakai di sini karena
# model itu juga menganggap truk sebagai forklift -- lihat komentar di forklift_filter.py).
DISPLAY_ORANGE_MAX_PCT = float(os.environ.get("DISPLAY_ORANGE_MAX_PCT", "1.0"))


# [FIX-CAM2-EDGE] Foto kendaraan cam2 sering diambil saat mobil baru separuh masuk ROI (bbox menempel tepi ROI, mis. y=1),
# padahal luasnya besar sehingga menang skor. Crop yang menyentuh tepi ROI dipotong skornya: tiap sisi yang menempel dikali
# EDGE_CROP_PENALTY. Hanya untuk kamera selain cam1 (cam1 tidak diubah). EDGE_CROP_PENALTY=1 mematikan aturan ini.
EDGE_CROP_PENALTY = float(os.environ.get("EDGE_CROP_PENALTY", "0.3"))
EDGE_CROP_MARGIN_PX = float(os.environ.get("EDGE_CROP_MARGIN_PX", "6"))


def _edge_cut_factor(bbox, roi_w, roi_h):
    """1.0 bila kendaraan utuh di dalam ROI; EDGE_CROP_PENALTY**n bila n sisi bbox menempel tepi ROI."""
    if EDGE_CROP_PENALTY >= 1.0 or not roi_w or not roi_h:
        return 1.0
    m = EDGE_CROP_MARGIN_PX
    n = int(bbox[0] <= m) + int(bbox[1] <= m) + int(bbox[2] >= roi_w - m) + int(bbox[3] >= roi_h - m)
    return EDGE_CROP_PENALTY ** n if n else 1.0


_FULL_FRAME_CTX = {}  # cam_name -> (frame penuh, offset_x ROI, offset_y ROI); diisi loop kamera sebelum handle_detections


def _display_crop_from_full(cam_name, bbox, roi_wh, pad_ratio=0.08):
    """Crop FOTO dari frame penuh (bukan ROI) agar kendaraan yang terpotong batas ROI tetap utuh.
    Mengembalikan (crop, faktor_tepi): faktor_tepi < 1 bila kendaraan memang menempel tepi frame penuh."""
    ctx = _FULL_FRAME_CTX.get(cam_name)
    if not ctx:
        return None, 1.0
    frame, ox, oy = ctx
    H, W = frame.shape[:2]
    bx1, by1, bx2, by2 = bbox[0] + ox, bbox[1] + oy, bbox[2] + ox, bbox[3] + oy
    f = _edge_cut_factor((bx1, by1, bx2, by2), W, H)
    bw, bh = bx2 - bx1, by2 - by1
    pw, ph = bw * pad_ratio, bh * pad_ratio
    # sisi bbox yang menempel batas ROI berarti kendaraan masih berlanjut di luar ROI -> perluas ke arah itu
    # Nilai ini adalah kelipatan ukuran bbox yang ditambahkan ke arah tepi ROI.
    # Bbox yang menempel ROI sering hanya menangkap bagian mobil; 0.6x terlalu kecil.
    # Default 1.8x memberi ruang lebih besar, tetap dibatasi oleh batas frame penuh.
    try:
        ext = max(0.0, min(4.0, float(os.environ.get("CAM2_PHOTO_EDGE_EXTEND", "1.8"))))
    except (TypeError, ValueError):
        ext = 1.8
    m = EDGE_CROP_MARGIN_PX
    l_ext = bw * ext if bbox[0] <= m else 0.0
    t_ext = bh * ext if bbox[1] <= m else 0.0
    r_ext = bw * ext if bbox[2] >= roi_wh[0] - m else 0.0
    b_ext = bh * ext if bbox[3] >= roi_wh[1] - m else 0.0
    x1, y1 = int(max(0, bx1 - pw - l_ext)), int(max(0, by1 - ph - t_ext))
    x2, y2 = int(min(W, bx2 + pw + r_ext)), int(min(H, by2 + ph + b_ext))
    if x2 <= x1 or y2 <= y1:
        return None, f
    return frame[y1:y2, x1:x2].copy(), f


# [FIX-FOTO-TAJAM] Foto yang dikirim ke OCR plat dipilih dari crop berskor tertinggi. Skor lama hanya memakai ukuran x confidence,
# padahal mobil parkir memberi puluhan frame berukuran hampir sama; yang terpilih jadi acak (kadang buram -> plat tak terbaca).
# Bobot ketajaman: 0 = nonaktif (perilaku lama), 1 = normal.
PHOTO_SHARP_WEIGHT = float(os.environ.get("PHOTO_SHARP_WEIGHT", "1.0"))


def _sharp_factor(item):
    sh = item.get("sharpness")
    if sh is None or PHOTO_SHARP_WEIGHT <= 0:
        return 1.0
    return (0.5 + min(float(sh), 500.0) / 500.0) ** PHOTO_SHARP_WEIGHT


def _display_score(item):
    return (
        item["area"]
        * item.get("edge_factor", 1.0)
        * (0.5 + item["conf"])
        * _sharp_factor(item)
        * _view_factor_bbox(item.get("own_bbox"))
        # [FIX-PLAT-FOTO] penalti foreshorten DIHAPUS: frame saat mobil dekat (plat paling besar) tidak lagi dihukum.
        * _stationary_bonus_factor(item)
    )


# [FIX-FOTO-WAKTU] best_recent_crop() memilih foto berdasar ukuran/ketajaman dari 10 crop yang bertahan (yang berskor
# rendah dibuang), sehingga foto Loading/Keluar di web selalu foto awal parkir (bak kosong). Riwayat foto berstempel
# waktu ini dipakai agar foto AKTIVITAS = bak SESUDAH kejadian dan foto Keluar = bak sesaat sebelum pergi.
PHOTO_HIST_MAX = int(float(os.environ.get("PHOTO_HIST_MAX", "90")))
PHOTO_HIST_GAP = float(os.environ.get("PHOTO_HIST_GAP", "3.0"))
PHOTO_AFTER_EVENT_SECONDS = float(os.environ.get("PHOTO_AFTER_EVENT_SECONDS", "6.0"))
PHOTO_EXIT_WINDOW_SECONDS = float(os.environ.get("PHOTO_EXIT_WINDOW_SECONDS", "15.0"))


def _record_photo(vehicle, crop, timestamp, area, sharpness, usable):
    try:
        hist = vehicle.setdefault("photo_hist", deque(maxlen=PHOTO_HIST_MAX))
        if hist and timestamp - hist[-1]["ts"] < PHOTO_HIST_GAP:
            return
        h, w = crop.shape[:2]
        m = max(h, w)
        small = crop if m <= 480 else cv2.resize(crop, (int(w * 480 / m), int(h * 480 / m)), interpolation=cv2.INTER_AREA)
        hist.append({"ts": float(timestamp), "crop": small.copy(), "area": float(area),
                     "sharp": float(sharpness or 0.0), "usable": bool(usable)})
    except Exception:
        pass


def photo_at_time(vehicle, t_ref, window=8.0, tag=""):
    """Foto kendaraan yang diambil paling dekat dengan t_ref (detik pipeline). None bila riwayat kosong."""
    hist = list(vehicle.get("photo_hist") or [])
    if not hist:
        return None
    pool = [h for h in hist if h["usable"] and abs(h["ts"] - t_ref) <= window] \
        or [h for h in hist if abs(h["ts"] - t_ref) <= window] \
        or [h for h in hist if h["usable"]] or hist
    if DISPLAY_ORANGE_MAX_PCT > 0:
        try:
            clean = [h for h in pool if orange_pct(h["crop"]) < DISPLAY_ORANGE_MAX_PCT]
            pool = clean or pool
        except Exception:
            pass
    best = max(pool, key=lambda h: h["area"] * (0.5 + min(h["sharp"], 500.0) / 500.0)
               * max(0.2, 1.0 - abs(h["ts"] - t_ref) / (2.0 * window)))
    if tag:
        print(f"[FOTO] {tag}: memakai crop jam {datetime.fromtimestamp(best['ts']).strftime('%H:%M:%S')} "
              f"(acuan {datetime.fromtimestamp(t_ref).strftime('%H:%M:%S')}, {len(hist)} foto di riwayat)")
    return best["crop"]


def photo_before_exit(vehicle, t_last_seen):
    """Foto terbaik dalam PHOTO_EXIT_WINDOW_SECONDS terakhir sebelum kendaraan hilang."""
    hist = list(vehicle.get("photo_hist") or [])
    if not hist:
        return None
    recent = [h for h in hist if h["ts"] >= t_last_seen - PHOTO_EXIT_WINDOW_SECONDS and h["usable"]] \
        or [h for h in hist if h["ts"] >= t_last_seen - PHOTO_EXIT_WINDOW_SECONDS] or hist[-3:]
    if DISPLAY_ORANGE_MAX_PCT > 0:
        try:
            clean = [h for h in recent if orange_pct(h["crop"]) < DISPLAY_ORANGE_MAX_PCT]
            recent = clean or recent
        except Exception:
            pass
    best = max(recent, key=lambda h: h["area"] * (0.5 + min(h["sharp"], 500.0) / 500.0))
    print(f"[FOTO] KELUAR: memakai crop jam {datetime.fromtimestamp(best['ts']).strftime('%H:%M:%S')} "
          f"(terakhir terlihat {datetime.fromtimestamp(t_last_seen).strftime('%H:%M:%S')})")
    return best["crop"]


def best_recent_crop(vehicle):
    confirmed_crop = vehicle.get("confirmed_plate_crop")
    if confirmed_crop is not None:
        return confirmed_crop

    recent = [it for it in (vehicle.get("recent_crops") or []) if it.get("crop") is not None]
    if not recent:
        return vehicle.get("last_crop") if vehicle.get("last_crop") is not None else vehicle.get("entry_display_crop")

    _pic = lambda it: it.get("display_crop") if it.get("display_crop") is not None else it["crop"]
    _key = _display_score
    if _plate_priority(vehicle) and PHOTO_PLATE_LIK_WEIGHT > 0:
        def _key(it):   # [FIX-FOTO-PLAT] utamakan crop yang terlihat ada plat-nya
            lik = it.get("_plik")
            if lik is None:
                lik = it["_plik"] = _plate_likelihood(_pic(it))
            return _display_score(it) * (1.0 + PHOTO_PLATE_LIK_WEIGHT * lik)
    if DISPLAY_ORANGE_MAX_PCT <= 0:
        return _pic(max(recent, key=_key))

    try:
        scored = [(orange_pct(it["crop"]), it) for it in recent]
    except Exception:
        return _pic(max(recent, key=_key))

    clean = [it for pct, it in scored if pct < DISPLAY_ORANGE_MAX_PCT]
    if clean:
        return _pic(max(clean, key=_key))
    if vehicle.get("entry_display_crop") is not None:
        return vehicle["entry_display_crop"]
    return min(scored, key=lambda pi: pi[0])[1]["crop"]


def best_cargo_detection(vehicle, top_n=5, debug_tag=None):
    if vehicle.get("is_box_body"):
        if debug_tag:
            print(
                f"[DEBUG-CARGO-VOTE] {vehicle.get('vehicle_id','?')}: SKIP cargo "
                f"[{debug_tag}], sesi ini sudah dikunci sebagai BODI BOKS TERTUTUP "
                f"-- isi muatan tidak mungkin dilihat dari kamera"
            )
        return CARGO_UNKNOWN, 0.0

    if CARGO_MIN_AREA_RATIO > 0:
        best_ratio = vehicle.get("best_area_ratio", 1.0)
        if best_ratio < CARGO_MIN_AREA_RATIO:
            if debug_tag:
                print(
                    f"[DEBUG-CARGO-VOTE] {vehicle.get('vehicle_id','?')}: SKIP cargo "
                    f"[{debug_tag}], best_area_ratio={best_ratio:.4f} < "
                    f"{CARGO_MIN_AREA_RATIO:.4f} (kendaraan belum pernah cukup besar)"
                )
            return CARGO_UNKNOWN, 0.0

    _entry_pool = vehicle.get("cargo_entry_crops") or []
    if not vehicle.get("entered") and _entry_pool:
        recent = [c for c in _entry_pool if c.get("usable_cargo", True)]
    else:
        recent = [
            c
            for c in (vehicle.get("recent_crops") or [])
            if c.get("usable_cargo", True)
        ]


    if not recent:
        if debug_tag:
            print(
                f"[DEBUG-CARGO-VOTE] {vehicle.get('vehicle_id','?')}: SKIP cargo "
                f"[{debug_tag}], belum ada crop yang lolos filter usable_cargo"
            )
        return CARGO_UNKNOWN, 0.0


    ranked = sorted(
        recent,
        key=lambda item: item.get(
            "score",
            item["area"] * (0.5 + item["conf"]),
        ),
        reverse=True
    )

    candidates = ranked[:top_n]


    votes = {}
    confidence_totals = {}
    box_hits = 0

    for idx, item in enumerate(candidates):

        if CARGO_BOX_GATE_ENABLED:
            _is_box_c, _ = _looks_like_closed_box(item["crop"])
            if _is_box_c:
                box_hits += 1

        prediction = detect_cargo_status(
            item["crop"]
        )

        if debug_tag and SAVE_DEBUG_IMAGES:
            try:
                debug_dir = "debug_cargo_vote"
                os.makedirs(
                    debug_dir,
                    exist_ok=True
                )

                if prediction is not None:
                    status, conf = prediction

                    safe_status = (
                        status
                        .replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                    )
                else:
                    safe_status = "UNKNOWN"

                vid = vehicle.get(
                    "vehicle_id",
                    "UNKNOWN"
                )

                fname = (
                    f"{vid}_{debug_tag}_"
                    f"cand{idx}_{safe_status}_"
                    f"t{item['timestamp']:.0f}.jpg"
                )

                cv2.imwrite(
                    os.path.join(
                        debug_dir,
                        fname
                    ),
                    item["crop"]
                )

            except Exception as e:
                print(
                    f"[DEBUG-CARGO-VOTE] "
                    f"Gagal simpan crop: {e}"
                )

        if prediction is None:
            continue

        status, conf = prediction

        if status == CARGO_UNKNOWN:
            continue

        votes[status] = (
            votes.get(status, 0) + 1
        )

        confidence_totals[status] = (
            confidence_totals.get(status, 0.0)
            + conf
        )

    if CARGO_BOX_GATE_ENABLED and candidates:
        if box_hits >= (len(candidates) + 1) // 2:
            vehicle["box_body_hits"] = vehicle.get("box_body_hits", 0) + 1
            if vehicle["box_body_hits"] >= CARGO_BOX_CONFIRM_HITS:
                if not vehicle.get("is_box_body"):
                    print(
                        f"[CARGO-BOX] {vehicle.get('vehicle_id','?')}: dikunci sebagai "
                        f"BODI BOKS TERTUTUP ({vehicle['box_body_hits']}x momen berturut, "
                        f"{box_hits}/{len(candidates)} crop) -- status muatan akan "
                        f"'{CARGO_UNKNOWN}' untuk seluruh sesi ini"
                    )
                vehicle["is_box_body"] = True
                return CARGO_UNKNOWN, 0.0
        else:
            vehicle["box_body_hits"] = 0

    if not votes:
        return CARGO_UNKNOWN, 0.0


    winner = max(
        votes,
        key=lambda status: (
            votes[status],
            confidence_totals[status]
        )
    )

    winner_votes = votes[winner]

    current_confidence = (
        confidence_totals[winner]
        / winner_votes
    )

    current_prediction = (
        winner,
        current_confidence
    )

    _total_votes = sum(votes.values())
    vehicle["last_moment_cargo"] = {
        "status": winner,
        "confidence": current_confidence,
        "agreement": (winner_votes / _total_votes) if _total_votes else 0.0,
        "votes": winner_votes,
    }

    if debug_tag:
        print(
            f"[DEBUG-CARGO-VOTE] "
            f"{vehicle.get('vehicle_id')} "
            f"[{debug_tag}] "
            f"crop_votes={votes} "
            f"winner={current_prediction}"
        )


    final_prediction = update_cargo_history(
        vehicle,
        current_prediction
    )

    if final_prediction is not None:

        if debug_tag:
            print(
                f"[DEBUG-CARGO-VOTE] "
                f"{vehicle.get('vehicle_id')} "
                f"[{debug_tag}] "
                f"TEMPORAL FINAL = "
                f"{final_prediction}"
            )

        return final_prediction

    return CARGO_UNKNOWN, 0.0


def _soft_loaded_score(truck_crop):
    if truck_crop is None or truck_crop.size == 0:
        return None
    if cargo_classifier is not None:
        try:
            _crop_ml = truck_crop if CARGO_DISABLE_BED_CROP else _crop_to_cargo_bed(truck_crop)
            _p = cargo_classifier.loaded_probability(_crop_ml)
            if _p is not None:
                return _p
            if cargo_classifier._load() is not None:
                return None
        except Exception:
            pass
    if CARGO_RULE_BASED_FALLBACK:
        _st, _ = _detect_cargo_status_rule_based(truck_crop)
        if _st == CARGO_LOADED:
            return 0.8
        if _st == CARGO_EMPTY:
            return 0.2
    return None


def _activity_box_fn(crop):
    if not CARGO_BOX_GATE_ENABLED:
        return False
    return bool(_looks_like_closed_box(crop)[0])


def determine_activity_type(cargo_in, cargo_out):
    is_loaded_in = cargo_in in [CARGO_LOADED]
    is_loaded_out = cargo_out in [CARGO_LOADED]
    is_empty_in = (cargo_in == CARGO_EMPTY)
    is_empty_out = (cargo_out == CARGO_EMPTY)

    if is_loaded_in and is_empty_out:
        return "Bongkar Muatan (Unloading)"
    elif is_empty_in and is_loaded_out:
        return "Mengambil Muatan (Loading)"
    elif is_loaded_in and is_loaded_out:
        return "Transit / membawa Muatan"
    elif is_empty_in and is_empty_out:
        return "Lewat Kosong"
    else:
        return "Tidak Dapat Dipastikan"


def _is_truck_class(cls, config):
    return cls == config.get("truck_class_id", 7)


def _is_cargo_class(cls, config):
    """Kelas yang boleh disampel muatannya (truk + pickup yang terbaca 'car')."""
    if config.get("vehicle_label"):
        return True  # mode generik: semua kendaraan yang lolos filter disampel
    classes = config.get("cargo_classes")
    if classes:
        return cls in classes
    return _is_truck_class(cls, config)


PICKUP_LABEL = os.environ.get("PICKUP_LABEL", "Truk")
PICKUP_MIN_VALID_SAMPLES = int(float(os.environ.get("PICKUP_MIN_VALID_SAMPLES", "3")))


def _cargo_model_has_reject():
    """True bila cargo_model.pkl sudah dilatih dengan kelas penolak (bukan_truk/forklift/tertutup)."""
    try:
        b = cargo_classifier._load() if cargo_classifier is not None else None
        if b is None:
            return False
        classes = {str(c) for c in b["model"].classes_}
        return "bukan_truk" in classes
    except Exception:
        return False


def _jenis_label(cls, config, vehicle=None):
    # [GENERIK] Per kamera: bila config punya "vehicle_label" (cam1), jenis tidak dibedakan.
    if config.get("vehicle_label"):
        return config["vehicle_label"]
    if cls == config.get("truck_class_id", 7):
        return "Truk"
    if vehicle is not None and vehicle.get("is_pickup"):
        return PICKUP_LABEL
    return "Mobil"


def _maybe_mark_pickup(vehicle, voted_cls, config):
    """Mobil (cls 2) yang bak-nya lolos classifier >= N kali = pickup.
    HANYA aktif bila model punya kelas 'bukan_truk' (kalau tidak, SUV pun lolos)."""
    if config.get("vehicle_label") or vehicle.get("is_pickup") or voted_cls != 2 or not _cargo_model_has_reject():
        return
    st = vehicle.get("_act") or {}
    valid = (st.get("diag") or {}).get("sampel_masuk", 0)
    if valid >= PICKUP_MIN_VALID_SAMPLES:
        vehicle["is_pickup"] = True
        print(f"[PICKUP] {vehicle.get('vehicle_id')}: cls=car tapi bak terbaca {valid}x "
              f"-> jenis kendaraan = {PICKUP_LABEL}")


def _crop_to_cargo_bed(truck_crop):
    if truck_crop is None or truck_crop.size == 0:
        return truck_crop

    h, w = truck_crop.shape[:2]
    if h < 20 or w < 20:
        return truck_crop

    bed = truck_crop[int(h * 0.3):int(h * 0.85), int(w * 0.2):int(w * 0.95)]

    if bed.size < 1000:
        return truck_crop

    return bed


def detect_cargo_status(truck_crop):
    if CARGO_BOX_GATE_ENABLED:
        _is_box, _box_metrics = _looks_like_closed_box(truck_crop)
        if DEBUG_MODE and _box_metrics:
            print(f"[DEBUG-CARGO-BOX-v1] is_box={_is_box} metrics={_box_metrics}")
        if _is_box:
            return CARGO_UNKNOWN, 0.0

    ml_available = False
    if cargo_classifier is not None:
        try:
            ml_available = bool(cargo_classifier.is_available())
        except Exception:
            ml_available = False

        _crop_for_ml = truck_crop if CARGO_DISABLE_BED_CROP else _crop_to_cargo_bed(truck_crop)
        ml_result = cargo_classifier.predict(_crop_for_ml)
        if ml_result is not None:
            status, confidence = ml_result
            return status, confidence

    if ml_available and not CARGO_RULE_BASED_FALLBACK:
        return CARGO_UNKNOWN, 0.0

    return _detect_cargo_status_rule_based(truck_crop)


def update_cargo_history(vehicle, prediction):

    if "cargo_history" not in vehicle:
        vehicle["cargo_history"] = deque(
            maxlen=CARGO_HISTORY_SIZE
        )

    if prediction is not None:
        vehicle["cargo_history"].append(prediction)

    history = list(vehicle["cargo_history"])

    if len(history) < CARGO_MIN_HISTORY:
        return None

    votes = {}
    confidence_totals = {}

    for status, confidence in history:
        votes[status] = votes.get(status, 0) + 1
        confidence_totals[status] = (
            confidence_totals.get(status, 0.0)
            + confidence
        )

    winning_status = max(
        votes,
        key=votes.get
    )

    winning_votes = votes[winning_status]

    agreement = (
        winning_votes / len(history)
    )

    if agreement < CARGO_AGREEMENT_RATIO:
        return None

    avg_confidence = (
        confidence_totals[winning_status]
        / winning_votes
    )

    if avg_confidence < CARGO_MIN_AVG_CONFIDENCE:
        print(
            f"[CARGO-VOTE] {vehicle.get('vehicle_id', '?')} DITOLAK: "
            f"winner={winning_status} agreement={agreement:.2f} TAPI "
            f"confidence rata-rata cuma {avg_confidence:.2f} < "
            f"{CARGO_MIN_AVG_CONFIDENCE:.2f} -- dianggap belum pasti"
        )
        return None

    print(
        f"[CARGO-VOTE] {vehicle.get('vehicle_id', '?')} "
        f"history={len(history)} "
        f"votes={votes} "
        f"winner={winning_status} "
        f"agreement={agreement:.2f} "
        f"confidence={avg_confidence:.2f}"
    )

    return winning_status, avg_confidence

def _looks_like_closed_box(truck_crop):
    if truck_crop is None or truck_crop.size < 5000:
        return False, {}

    h, w = truck_crop.shape[:2]
    if h < 20 or w < 20:
        return False, {}

    panel = truck_crop[int(h * 0.15):int(h * 0.90), int(w * 0.05):int(w * 0.98)]
    if panel.size < 1000:
        return False, {}

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    grid_n = 6
    gh, gw = gray.shape[0] // grid_n, gray.shape[1] // grid_n
    cell_vars = []
    if gh > 0 and gw > 0:
        for i in range(grid_n):
            for j in range(grid_n):
                cell = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                if cell.size > 0:
                    cell_vars.append(float(cell.var()))

    if not cell_vars:
        return False, {}

    cell_vars = np.array(cell_vars)
    mean_var = float(np.mean(cell_vars))
    std_var = float(np.std(cell_vars))

    metrics = {"mean_var": round(mean_var, 1), "std_var": round(std_var, 1)}

    is_box = mean_var < 180 and std_var < 90

    if is_box:
        panel_hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        ph = panel_hsv.shape[0]
        top_half = panel_hsv[: ph // 2]
        bottom_half = panel_hsv[ph // 2:]

        top_hue = float(np.mean(top_half[:, :, 0]))
        bottom_hue = float(np.mean(bottom_half[:, :, 0]))
        top_sat = float(np.mean(top_half[:, :, 1]))
        bottom_sat = float(np.mean(bottom_half[:, :, 1]))

        hue_diff = abs(top_hue - bottom_hue)
        hue_diff = min(hue_diff, 180 - hue_diff)
        sat_diff = abs(top_sat - bottom_sat)

        metrics["hue_diff"] = round(hue_diff, 1)
        metrics["sat_diff"] = round(sat_diff, 1)

        if hue_diff > 15 or sat_diff > 40:
            is_box = False

    return bool(is_box), metrics


def _is_usable_cargo_crop(crop):
    if crop is None or crop.size == 0:
        return False

    crop_h, crop_w = crop.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return False

    if min(crop_w, crop_h) < CARGO_MIN_CROP_SIDE:
        return False
    if (crop_w * crop_h) < CARGO_MIN_CROP_PIXELS:
        return False

    aspect_ratio = crop_w / max(crop_h, 1)
    if aspect_ratio > CARGO_ASPECT_RATIO_RELAXED_MAX:
        return False

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
    except Exception:
        return False

    if texture_var < 30 and edge_density < 0.02:
        return False

    return True


def _detect_cargo_status_rule_based(truck_crop):
    if truck_crop is None or truck_crop.size < 5000:
        return CARGO_UNKNOWN, 0.0

    h, w = truck_crop.shape[:2]
    if h < 20 or w < 20:
        return CARGO_UNKNOWN, 0.0

    if not CARGO_BOX_GATE_ENABLED:
        is_box, box_metrics = _looks_like_closed_box(truck_crop)
        if DEBUG_MODE:
            print(f"[DEBUG-CARGO-BOX-v1] is_box={is_box} metrics={box_metrics}")
        if is_box:
            return CARGO_UNKNOWN, 0.0

    bed_crop = truck_crop[int(h * 0.3):int(h * 0.85), int(w * 0.2):int(w * 0.95)]

    if bed_crop.size < 1000:
        return CARGO_UNKNOWN, 0.0

    hsv_bed = cv2.cvtColor(bed_crop, cv2.COLOR_BGR2HSV)
    sat = hsv_bed[:, :, 1]
    val = hsv_bed[:, :, 2]
    vivid_mask = (sat > 70) & (val > 40)
    dark_mask = (val < 60)
    vivid_ratio = np.sum(vivid_mask) / vivid_mask.size
    dark_ratio = np.sum(dark_mask) / dark_mask.size

    bed_gray_for_edges = cv2.cvtColor(bed_crop, cv2.COLOR_BGR2GRAY)
    bed_edges = cv2.Canny(bed_gray_for_edges, 50, 150)
    bed_edge_density = float(np.sum(bed_edges > 0)) / bed_edges.size

    if (vivid_ratio > 0.35 or dark_ratio > 0.5) and bed_edge_density > 0.03:
        return CARGO_LOADED, 0.7


    gray = cv2.cvtColor(bed_crop, cv2.COLOR_BGR2GRAY)
    texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    hue_std = np.std(hsv_bed[:, :, 0])
    value_std = np.std(hsv_bed[:, :, 2])

    area_ratio = bed_crop.shape[0] * bed_crop.shape[1] / 20000.0
    threshold_texture = 600 * area_ratio
    threshold_color = 35 * area_ratio

    score = 0
    if texture_var > threshold_texture:
        score += 2
    if hue_std > threshold_color:
        score += 1.5
    if value_std > 25:
        score += 1
    if bed_edge_density > 0.03:
        score += 1

    if score >= 4.5:
        return CARGO_LOADED, 0.75
    elif score <= 2:
        return CARGO_EMPTY, 0.75
    else:
        return CARGO_UNKNOWN, 0.0


def looks_like_shadow(crop):
    if crop is None or crop.size < 3000:
        return False
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        texture_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        is_dark = mean_val < SHADOW_MAX_VALUE
        is_low_sat = mean_sat < SHADOW_MAX_SATURATION
        is_flat = edge_density < SHADOW_MAX_EDGE_DENSITY and texture_var < SHADOW_MAX_TEXTURE_VAR

        return is_dark and is_low_sat and is_flat
    except Exception:
        return False


# [FIX-PARKED-HOP] Mobil yang sedang PARKIR tidak mungkin pindah ratusan piksel dalam sekejap. Dari log
# (cam2, 7 Okt 2026): "RE-APPEAR ... hilang 0.2s (jarak=219px, sim=0.78)" -- track mobil putih di sebelahnya
# menempel ke sesi mobil pertama karena warna (histogram HSV) kedua SUV putih nyaris sama. Aturan ini: sesi yang
# kendaraannya sedang diam hanya boleh disambung ke kotak yang posisinya hampir sama (pergeseran pusat <= nilai
# ini x ukuran kotak). 0 = matikan aturan ini.
STATIONARY_RELINK_MAX_SHIFT_RATIO = float(os.environ.get("STATIONARY_RELINK_MAX_SHIFT_RATIO", "0.35"))


def parked_vehicle_cannot_jump(vehicle, new_bbox):
    if STATIONARY_RELINK_MAX_SHIFT_RATIO <= 0 or vehicle.get("stationary_since") is None:
        return False
    last_box = vehicle.get("last_bbox")
    if not last_box:
        return False
    lw, lh = bbox_size(last_box)
    nw, nh = bbox_size(new_bbox)
    lcx, lcy = bbox_center(last_box)
    ncx, ncy = bbox_center(new_bbox)
    norm = max(lw, lh, nw, nh, 1.0)
    return math.hypot(ncx - lcx, ncy - lcy) / norm > STATIONARY_RELINK_MAX_SHIFT_RATIO


def appearance_signature(crop):
    if crop is None or crop.size < 5000:
        return None
    try:
        img = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)
    except Exception:
        return None


def cam2_appearance_signature(crop):
    if crop is None or crop.size < 5000:
        return None
    try:
        img = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)
    except Exception:
        return None


def appearance_similarity(sig_a, sig_b):
    if sig_a is None or sig_b is None:
        return None
    try:
        distance = cv2.compareHist(sig_a.astype(np.float32), sig_b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except Exception:
        return None


def update_appearance_sig(vehicle, crop):
    sig = appearance_signature(crop)
    if sig is None:
        return
    old = vehicle.get("appearance_sig")
    if old is None or old.shape != sig.shape:
        vehicle["appearance_sig"] = sig
    else:
        vehicle["appearance_sig"] = (0.7 * old + 0.3 * sig).astype(np.float32)


def update_cam2_appearance_sig(vehicle, crop):
    sig = cam2_appearance_signature(crop)
    if sig is None:
        return
    old = vehicle.get("cam2_appearance_sig")
    if old is None or old.shape != sig.shape:
        vehicle["cam2_appearance_sig"] = sig
    else:
        vehicle["cam2_appearance_sig"] = (0.7 * old + 0.3 * sig).astype(np.float32)


COLOR_VOTE_MIN_PIXELS = 10


def _classify_color_hsv(h, s, v):
    if v < 45:
        return "Hitam"
    if s < 30 and v > 195:
        return "Putih"
    if s < 55:
        return "Abu-abu/Silver"
    if h < 8 or h >= 172:
        return "Merah"
    if 8 <= h < 20:
        return "Coklat" if v < 150 else "Oranye"
    if 20 <= h < 35:
        return "Kuning"
    if 35 <= h < 85:
        return "Hijau"
    if 85 <= h < 130:
        return "Biru"
    if 130 <= h < 172:
        return "Ungu/Merah Muda"
    return "Lainnya"


def detect_vehicle_color(crop_bgr, sample_max_dim=80):
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return None, None

    y0, y1 = int(h * 0.20), int(h * 0.80)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    region = crop_bgr[y0:y1, x0:x1]
    if region.size == 0:
        region = crop_bgr

    rh, rw = region.shape[:2]
    scale = sample_max_dim / max(rh, rw)
    if scale < 1.0:
        region = cv2.resize(
            region, (max(1, int(rw * scale)), max(1, int(rh * scale))),
            interpolation=cv2.INTER_AREA,
        )

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels_bgr = region.reshape(-1, 3).astype(np.float32)
    pixels_hsv = hsv_region.reshape(-1, 3)

    v_channel = pixels_hsv[:, 2]
    keep_mask = (v_channel > 25) & (v_channel < 250)
    if keep_mask.sum() >= COLOR_VOTE_MIN_PIXELS:
        pixels_bgr = pixels_bgr[keep_mask]

    if len(pixels_bgr) < COLOR_VOTE_MIN_PIXELS:
        return None, None

    n_clusters = 3 if len(pixels_bgr) >= 30 else 1
    if n_clusters > 1:
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels_bgr, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            counts = np.bincount(labels.flatten(), minlength=n_clusters)
            dominant_bgr = centers[int(np.argmax(counts))]
        except Exception:
            dominant_bgr = pixels_bgr.mean(axis=0)
    else:
        dominant_bgr = pixels_bgr.mean(axis=0)

    b, g, r = [max(0, min(255, int(round(c)))) for c in dominant_bgr]
    dom_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    nama = _classify_color_hsv(int(dom_hsv[0]), int(dom_hsv[1]), int(dom_hsv[2]))
    return nama, (r, g, b)


def _update_color_vote(vehicle, crop):
    nama, rgb = detect_vehicle_color(crop)
    if nama is None:
        return

    if "color_votes" not in vehicle:
        vehicle["color_votes"] = {}
        vehicle["color_rgb_sum"] = {}

    vehicle["color_votes"][nama] = vehicle["color_votes"].get(nama, 0) + 1
    prev_sum = vehicle["color_rgb_sum"].get(nama, [0, 0, 0])
    vehicle["color_rgb_sum"][nama] = [prev_sum[i] + rgb[i] for i in range(3)]


def _voted_color(vehicle):
    locked = vehicle.get("locked_color")
    if locked is not None:
        return locked

    votes = vehicle.get("color_votes") or {}
    if not votes:
        return "Tidak Terdeteksi", (128, 128, 128)

    nama = max(votes, key=votes.get)
    count = votes[nama]
    rgb_sum = vehicle.get("color_rgb_sum", {}).get(nama, [128 * count] * 3)
    rgb = tuple(int(round(s / count)) for s in rgb_sum)
    return nama, rgb


def classify_vehicle_direction(vehicle, config):
    history = vehicle.get("direction_history") or []
    if len(history) < 2:
        return "TIDAK_JELAS", 0.0

    t_old, old_bbox = history[0]
    t_new, new_bbox = history[-1]
    if t_new - t_old < DIRECTION_MIN_TIME_SPAN:
        return "TIDAK_JELAS", 0.0

    SMOOTH_N = 3
    old_sample = history[:SMOOTH_N]
    new_sample = history[-SMOOTH_N:]
    old_bbox = tuple(sum(b[i] for _, b in old_sample) / len(old_sample) for i in range(4))
    new_bbox = tuple(sum(b[i] for _, b in new_sample) / len(new_sample) for i in range(4))

    axis = config.get("direction_axis", "x")
    sign_masuk = config.get("direction_sign_masuk", 1)
    size_trend_masuk = config.get("direction_size_trend_masuk", "growing")

    fx1, fy1, fx2, fy2 = old_bbox
    lx1, ly1, lx2, ly2 = new_bbox

    f_cx, f_cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    l_cx, l_cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0

    f_w, f_h = max(1.0, fx2 - fx1), max(1.0, fy2 - fy1)
    l_w, l_h = max(1.0, lx2 - lx1), max(1.0, ly2 - ly1)

    if axis == "y":
        delta = l_cy - f_cy
        norm = max(f_h, l_h, 1.0)
    else:
        delta = l_cx - f_cx
        norm = max(f_w, l_w, 1.0)

    if DIRECTION_MIN_DISPLACEMENT > 0 and config.get("kamera_id") != "cam1" and abs(delta) / norm < DIRECTION_MIN_DISPLACEMENT:
        return "TIDAK_JELAS", 0.0   # hampir tidak berpindah = parkir; perubahan ukuran kotak saja bukan bukti masuk/keluar

    pos_component = max(-1.0, min(1.0, (delta / norm))) * sign_masuk

    f_area = f_w * f_h
    l_area = l_w * l_h
    size_ratio = (l_area / f_area) if f_area > 0 else 1.0
    size_delta = max(-1.0, min(1.0, size_ratio - 1.0))
    if size_trend_masuk == "shrinking":
        size_delta = -size_delta

    score = 0.7 * pos_component + 0.3 * size_delta

    if score > DIRECTION_SCORE_THRESHOLD:
        return "MASUK", score
    if score < -DIRECTION_SCORE_THRESHOLD:
        return "KELUAR", score
    return "TIDAK_JELAS", score


def _evict_worst_crop(recent_crops):
    non_snapshot_indices = [
        i for i, c in enumerate(recent_crops) if not c.get("from_snapshot")
    ]
    pool = non_snapshot_indices if non_snapshot_indices else range(len(recent_crops))
    worst_idx = min(
        pool,
        key=lambda i: recent_crops[i].get("plate_score", recent_crops[i]["score"]),
    )
    recent_crops.pop(worst_idx)


# [BUKTI-BAK] Bukti VISUAL bahwa ada barang masuk/keluar bak: bandingkan crop kendaraan yang DIAM dengan baseline (median 5 crop
# pertama saat kendaraan sudah diam). Area bak yang berubah >= BED_CHANGE_MIN_PCT % dianggap "bak berubah". Dipakai sebagai syarat
# pembenar event "lintasan forklift" (yang sendirian tidak bisa dipercaya karena YOLO forklift salah mengenali truk).
# Uji pada 3 foto nyata: bak kosong vs kosong = 0.1% (orang bergerak), bak kosong vs ada kotak = ~10%.
BED_CHANGE_MIN_PCT = float(os.environ.get("BED_CHANGE_MIN_PCT", "4.0"))
BED_CHANGE_PIXEL_DIFF = int(float(os.environ.get("BED_CHANGE_PIXEL_DIFF", "40")))
BED_CHANGE_INTERVAL = float(os.environ.get("BED_CHANGE_INTERVAL", "1.0"))
BED_CHANGE_MIN_HITS = int(float(os.environ.get("BED_CHANGE_MIN_HITS", "2")))
# "bak" = event lintasan-forklift hanya dipakai bila bak terbukti berubah; "off" = selalu diabaikan
ACTIVITY_FORKLIFT_ONLY_CONFIRM = os.environ.get("ACTIVITY_FORKLIFT_ONLY_CONFIRM", "bak").strip().lower()


def _bed_gray(crop):
    g = cv2.cvtColor(cv2.resize(crop, (128, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (7, 7), 0).astype(np.int16)


def _record_bed_change(vehicle, crop, timestamp):
    try:
        if not vehicle.get("is_stationary"):
            return  # baseline & bukti hanya saat kendaraan diam (bbox stabil)
        st = vehicle.setdefault("_bed", {"frames": [], "base": None, "last_t": -1e18, "hits": []})
        if timestamp - st["last_t"] < BED_CHANGE_INTERVAL:
            return
        st["last_t"] = timestamp
        g = _bed_gray(crop)
        if st["base"] is None:
            st["frames"].append(g)
            if len(st["frames"]) >= 5:
                st["base"] = np.median(np.stack(st["frames"]), axis=0).astype(np.int16)
                st["frames"] = []
            return
        pct = float((np.abs(g - st["base"]) > BED_CHANGE_PIXEL_DIFF)[40:92, 25:110].mean()) * 100.0
        if pct >= BED_CHANGE_MIN_PCT:
            st["hits"].append((timestamp, pct))
            if len(st["hits"]) > 1000:
                del st["hits"][:200]
    except Exception:
        pass


def _bed_change_supports(vehicle, ev, pad_before=1.0, pad_after=2.0):
    """True bila selama jendela event (t_start..t_end, +pad) area bak terbukti berubah >= BED_CHANGE_MIN_HITS kali."""
    try:
        t0 = float(ev["t_start"]) - pad_before
        t1 = float(ev["t_end"]) + pad_after
        hits = (vehicle.get("_bed") or {}).get("hits", [])
        return sum(1 for t, _p in hits if t0 <= t <= t1) >= BED_CHANGE_MIN_HITS
    except Exception:
        return False


class TruckTrackState:
    def __init__(self, camera_id, single_lane_mode=False, cargo_detection=True, plate_detection=False, read_own_plate=True):
        self.camera_id = camera_id
        self.single_lane_mode = single_lane_mode
        self.cargo_detection = cargo_detection
        self.plate_detection = plate_detection
        self.read_own_plate = read_own_plate
        self._crop_lock = threading.Lock()
        self.tracks = {}
        self.vehicles = {}
        self.track_to_vehicle = {}
        self.recently_exited = []
        self.vehicle_seq = 0
        self.session_start = int(time.time())
        self.sent_event_ids = set()
        self.class_history = {}
        self.pending_exits = {}
        self._stale_relink_logged = set()
        self.lost_grace_until = 0.0  # [GAP-GUARD]

    def _new_vehicle_id(self):
        self.vehicle_seq += 1
        return f"{self.camera_id.upper()}_SES_{self.session_start}_{self.vehicle_seq:06d}"

    def can_send_event(self, event_id):
        if event_id in self.sent_event_ids:
            return False
        self.sent_event_ids.add(event_id)
        if len(self.sent_event_ids) > 10000:
            self.sent_event_ids = set(list(self.sent_event_ids)[-1000:])
        return True

    def make_event_id(self, vehicle, suffix):
        return f"{self.camera_id}:{vehicle['vehicle_id']}:{suffix}"

    def _calculate_distance(self, box1, box2):
        x1_center = (box1[0] + box1[2]) / 2.0
        y1_center = (box1[1] + box1[3]) / 2.0
        x2_center = (box2[0] + box2[2]) / 2.0
        y2_center = (box2[1] + box2[3]) / 2.0
        return math.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)

    def get_or_create_vehicle(self, track_id, cls, bbox, timestamp, crop=None,
                               this_frame_track_ids=None):
        is_new_track = track_id not in self.tracks
        was_relinked = False
        was_return = False

        if track_id in self.track_to_vehicle:
            v_id = self.track_to_vehicle[track_id]
            return self.vehicles.get(v_id), is_new_track, was_relinked, was_return

        def has_other_visible_track(vehicle):
            if self.camera_id != "cam2":
                return False
            active_tracks = set(vehicle.get("active_tracks") or [])
            visible_tracks = this_frame_track_ids or set()
            return bool((active_tracks - {track_id}) & visible_tracks)

        def vehicle_appearance_similarity(vehicle):
            if crop is None:
                return None
            if self.camera_id == "cam2":
                return appearance_similarity(
                    vehicle.get("cam2_appearance_sig"), cam2_appearance_signature(crop)
                )
            return appearance_similarity(
                vehicle.get("appearance_sig"), appearance_signature(crop)
            )

        box_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_diag = math.sqrt(box_w**2 + bbox_h**2)
        THRESHOLD_DISTANCE_PIXELS = bbox_diag * 2.0
        THRESHOLD_DISTANCE_PIXELS = max(THRESHOLD_DISTANCE_PIXELS, 100.0)
        _dist_cap = (
            SINGLE_LANE_TRACK_MATCH_MAX_DIST_PIXELS
            if self.single_lane_mode
            else TRACK_MATCH_MAX_DIST_PIXELS
        )
        THRESHOLD_DISTANCE_PIXELS = min(THRESHOLD_DISTANCE_PIXELS, _dist_cap)

        best_match_v_id = None
        best_match_score = -1.0
        for v_id, vehicle in self.vehicles.items():
            if vehicle.get("session_closed"):
                continue
            if has_other_visible_track(vehicle):
                continue
            last_box = vehicle.get("last_bbox")
            if not last_box:
                continue

            if not class_compatible(vehicle.get("final_cls", cls), cls):
                continue

            dist = self._calculate_distance(last_box, bbox)
            if dist >= THRESHOLD_DISTANCE_PIXELS:
                continue

            last_w, last_h = bbox_size(last_box)
            new_w, new_h = bbox_size(bbox)
            size_ratio = max(new_w * new_h, 1.0) / max(last_w * last_h, 1.0)
            if size_ratio < TRACK_MATCH_MIN_SIZE_RATIO or size_ratio > TRACK_MATCH_MAX_SIZE_RATIO:
                continue

            f_cx, f_cy = bbox_center(last_box)
            n_cx, n_cy = bbox_center(bbox)
            center_norm = max(last_w, last_h, new_w, new_h, 1.0)
            center_ratio = math.sqrt((n_cx - f_cx) ** 2 + (n_cy - f_cy) ** 2) / center_norm
            if center_ratio > TRACK_MATCH_MAX_CENTER_RATIO:
                continue

            if parked_vehicle_cannot_jump(vehicle, bbox):
                _pk_key = ("PK", track_id, v_id)
                if _pk_key not in self._stale_relink_logged:
                    self._stale_relink_logged.add(_pk_key)
                    print(f"[{self.camera_id}] RE-LINK DITOLAK: {v_id} sedang DIAM (parkir) dan track {track_id} "
                          f"terlalu jauh dari posisinya -- kemungkinan kendaraan lain di sebelahnya.")
                continue

            iou = bbox_iou(last_box, bbox)

            if iou < TRACK_MATCH_MIN_IOU and center_ratio > TRACK_MATCH_MAX_CENTER_RATIO * 0.5:
                continue

            geometric_score = (1.0 - min(1.0, dist / THRESHOLD_DISTANCE_PIXELS)) + iou
            score = geometric_score

            STRONG_GEOMETRIC_MATCH = 1.5

            stale_gap = timestamp - vehicle.get("last_seen", timestamp)
            is_stale = stale_gap > STALE_RELINK_GRACE_SECONDS

            sim = vehicle_appearance_similarity(vehicle)
            if self.camera_id == "cam2" and (
                sim is None or sim < REVISIT_MIN_APPEARANCE_SCORE
            ):
                continue
            if crop is not None:
                if sim is not None:
                    if sim < REVISIT_MIN_APPEARANCE_SCORE * 0.7 and geometric_score < STRONG_GEOMETRIC_MATCH:
                        continue
                    score += sim

            if is_stale:
                _log_key = (track_id, v_id)
                _should_log = _log_key not in self._stale_relink_logged
                if _should_log:
                    self._stale_relink_logged.add(_log_key)
                    if len(self._stale_relink_logged) > 5000:
                        self._stale_relink_logged.clear()
                if sim is None:
                    if _should_log:
                        print(f"[{self.camera_id}] RE-LINK DITOLAK: {v_id} sudah "
                              f"{stale_gap:.0f}s tidak terlihat (batas {STALE_RELINK_GRACE_SECONDS:.0f}s) "
                              f"dan tidak ada bukti visual untuk dicocokkan -- "
                              f"track {track_id} tidak digabung ke sesi ini.")
                    continue
                if sim < REVISIT_MIN_APPEARANCE_SCORE:
                    if _should_log:
                        print(f"[{self.camera_id}] RE-LINK DITOLAK: {v_id} sudah "
                              f"{stale_gap:.0f}s tidak terlihat dan kemiripan visual cuma "
                              f"{sim:.2f} < {REVISIT_MIN_APPEARANCE_SCORE:.2f} -- kemungkinan "
                              f"KENDARAAN BERBEDA di posisi yang sama, bukan sesi yang sama.")
                    continue

            if score > best_match_score:
                best_match_score = score
                best_match_v_id = v_id

        if best_match_v_id is not None:
            print(f"[{self.camera_id}] RE-LINK: Track baru {track_id} "
                  f"digabung ke sesi aktif {best_match_v_id} (score={best_match_score:.2f})")
            self._attach_track(best_match_v_id, track_id, timestamp)
            vehicle = self.vehicles[best_match_v_id]
            vehicle["last_bbox"] = bbox
            was_relinked = True
            return vehicle, is_new_track, was_relinked, was_return

        MISSING_GRACE_SECONDS = min(TRACK_RELINK_GRACE_SECONDS, LOST_SECONDS)

        for v_id, vehicle in self.vehicles.items():
            if vehicle.get("session_closed"):
                continue
            if not vehicle.get("entered"):
                continue

            if has_other_visible_track(vehicle):
                continue

            time_since_last_seeen = timestamp - vehicle.get("last_seen", 0)

            if time_since_last_seeen > MISSING_GRACE_SECONDS:
                continue
            if time_since_last_seeen < 0.1:
                continue

            last_box = vehicle.get("last_bbox")
            if last_box:
                bbox_h = bbox[3] - bbox[1]
                bbox_diag = math.sqrt(box_w**2 + bbox_h**2)
                REAPPEAR_THRESHOLD = bbox_diag * 3.0
                REAPPEAR_THRESHOLD = min(REAPPEAR_THRESHOLD, 600.0)
                if not self.single_lane_mode:
                    REAPPEAR_THRESHOLD = min(REAPPEAR_THRESHOLD, bbox_diag * 1.5, 250.0)

                dist = self._calculate_distance(last_box, bbox)
                if dist < REAPPEAR_THRESHOLD and class_compatible(vehicle.get("final_cls", cls), cls) \
                        and parked_vehicle_cannot_jump(vehicle, bbox):
                    _pk_key = ("PK", track_id, v_id)
                    if _pk_key not in self._stale_relink_logged:
                        self._stale_relink_logged.add(_pk_key)
                        print(f"[{self.camera_id}] RE-APPEAR DITOLAK: {v_id} sedang DIAM (parkir) dan track "
                              f"{track_id} terlalu jauh dari posisinya (jarak={dist:.0f}px) -- kemungkinan "
                              f"kendaraan lain di sebelahnya.")
                elif dist < REAPPEAR_THRESHOLD and class_compatible(vehicle.get("final_cls", cls), cls):
                    sim = vehicle_appearance_similarity(vehicle)

                    if not self.single_lane_mode and sim is None or sim is not None and sim < REVISIT_MIN_APPEARANCE_SCORE * 0.7:
                        pass
                    else:
                        print(f"[{self.camera_id}] RE-APPEAR: {v_id} muncul kembali "
                            f"setelah hilang {time_since_last_seeen:.1f}s (jarak={dist:.0f}px"
                            f"{f', sim={sim:.2f}' if sim is not None else ''})")
                        self._attach_track(v_id, track_id, timestamp)
                        vehicle["last_bbox"] = bbox
                        vehicle["last_seen"] = timestamp
                        was_relinked = True
                        return vehicle, is_new_track, was_relinked, was_return

        REVISIT_MAX_DIST_PIXELS = min(THRESHOLD_DISTANCE_PIXELS * 2, REVISIT_MAX_DIST_CAP_PIXELS)

        # [FIX-RETURN-FRAGMENT] Track baru yang SELURUHNYA berada di dalam kotak sesi yang masih TERBUKA adalah
        # fragmen kendaraan itu (bak/pintu/orang di dekatnya), bukan kendaraan lain yang kembali. Dari log uji:
        # TRK_0394 (di dalam kotak pickup) dicocokkan ke sesi van yang sudah tertutup (kemiripan 0.71 >= 0.60)
        # sehingga muncul MASUK ke-3 palsu. Hanya dipakai utk RETURN VISIT (gap panjang), bukan RESUME.
        _inside_open = False
        _ba0 = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        for _ov_id, _ov in self.vehicles.items():
            if _ov.get("session_closed"):
                continue
            _lb0 = _ov.get("last_bbox")
            if not _lb0:
                continue
            if timestamp - _ov.get("last_seen", timestamp) > STALE_RELINK_GRACE_SECONDS:
                continue
            _la0 = max(1.0, (_lb0[2] - _lb0[0]) * (_lb0[3] - _lb0[1]))
            if _ba0 > _la0 * 1.1:
                continue
            _iw0 = max(0.0, min(_lb0[2], bbox[2]) - max(_lb0[0], bbox[0]))
            _ih0 = max(0.0, min(_lb0[3], bbox[3]) - max(_lb0[1], bbox[1]))
            if (_iw0 * _ih0) / _ba0 >= RETURN_FRAGMENT_MIN_CONTAIN:
                _inside_open = True
                break

        for v_id, vehicle in self.vehicles.items():
            if not vehicle.get("session_closed"):
                continue
            if vehicle.get("time_out") is None:
                continue
            if has_other_visible_track(vehicle):
                continue
            last_box = vehicle.get("last_bbox")
            # [FIX-PARKIR-RESUME] kendaraan yang DIAM saat hilang lalu muncul lagi di kotak yang sama = kendaraan yang sama
            _same_spot_parked = bool(last_box and vehicle.get("is_stationary")
                                     and _bbox_iou(last_box, bbox) >= SAME_SPOT_MIN_IOU)
            if timestamp - vehicle["time_out"] > (REVISIT_RETENTION_EFF if _same_spot_parked else REVISIT_RETENTION_SECONDS):
                continue
            if not (last_box and self._calculate_distance(last_box, bbox) < REVISIT_MAX_DIST_PIXELS):
                continue
            if not class_compatible(vehicle.get("final_cls", cls), cls):
                continue

            sim = vehicle_appearance_similarity(vehicle)

            if (sim is None and not _same_spot_parked) or \
                    (sim is not None and sim < REVISIT_MIN_APPEARANCE_SCORE * (0.5 if _same_spot_parked else 1.0)):
                _rv_key = ("RV", track_id, v_id)
                if _rv_key not in self._stale_relink_logged:
                    self._stale_relink_logged.add(_rv_key)
                    print(f"[{self.camera_id}] RETURN/RESUME DITOLAK: track {track_id} vs sesi tertutup {v_id} -- "
                          f"kemiripan={'N/A' if sim is None else f'{sim:.2f}'} < ambang {REVISIT_MIN_APPEARANCE_SCORE:.2f} "
                          f"(jika ini kendaraan yang sama, turunkan REVISIT_MIN_APPEARANCE_SCORE di _env)")
                if DEBUG_MODE:
                    print(
                        f"[{self.camera_id}] RETURN VISIT DITOLAK: {v_id} -- "
                        f"tidak ada bukti visual yang cukup kuat untuk "
                        f"menyamakan track baru {track_id} dengan sesi lama "
                        f"ini (sim={f'{sim:.2f}' if sim is not None else 'N/A (tidak ada crop/appearance_sig)'}, "
                        f"ambang={REVISIT_MIN_APPEARANCE_SCORE:.2f}) -- "
                        f"kemungkinan KENDARAAN BERBEDA, dicatat sebagai sesi baru."
                    )
                continue

            gap = timestamp - vehicle["time_out"]

            if gap < MIN_GENUINE_REVISIT_GAP_SECONDS or (_same_spot_parked and gap < PARKED_HOLD_SECONDS):
                print(f"[{self.camera_id}] RESUME (bukan return visit): {v_id} muncul lagi "
                      f"cuma {gap:.1f}s setelah ditutup -- disambung ke sesi yang sama, "
                      f"MASUK TIDAK dihitung ulang")
                vehicle["session_closed"] = False
                vehicle["time_out"] = None
                self._attach_track(v_id, track_id, timestamp)
                vehicle["last_bbox"] = bbox
                vehicle["last_seen"] = timestamp
                was_relinked = True
                return vehicle, is_new_track, was_relinked, was_return

            if _inside_open:
                _rf_key = ("RF", track_id, v_id)
                if _rf_key not in self._stale_relink_logged:
                    self._stale_relink_logged.add(_rf_key)
                    print(f"[{self.camera_id}] RETURN VISIT DITOLAK: track {track_id} berada di dalam kotak "
                          f"sesi yang masih terbuka -> fragmen kendaraan itu, bukan {v_id} yang kembali")
                continue

            print(f"[{self.camera_id}] RETURN VISIT: {v_id} kembali setelah "
                  f"{gap:.0f}s")
            vehicle["session_closed"] = False
            vehicle["time_out"] = None
            vehicle["entered"] = False
            vehicle["time_in"] = None
            vehicle["entry_event_sent"] = False
            vehicle["locked_cls"] = None
            vehicle["cls_weights"] = {}
            vehicle["locked_color"] = None
            vehicle["color_votes"] = {}
            vehicle["color_rgb_sum"] = {}
            vehicle["cargo_in_latched"] = None
            vehicle["cargo_history"] = deque(maxlen=CARGO_HISTORY_SIZE)
            vehicle["_parked_confirm_logged"] = False
            vehicle["cls_votes"] = {}
            vehicle["cls_confs"] = {}
            vehicle["_cls_indecisive_logged"] = False
            self.class_history.pop(v_id, None)
            vehicle["first_seen"] = timestamp
            vehicle["visit_no"] = vehicle.get("visit_no", 1) + 1
            self._attach_track(v_id, track_id, timestamp)
            vehicle["last_bbox"] = bbox
            was_return = True
            return vehicle, is_new_track, was_relinked, was_return

        COOLDOWN_SECONDS = 60.0
        self.recently_exited = [item for item in self.recently_exited
                                if timestamp - item["time"] < COOLDOWN_SECONDS]
        for item in self.recently_exited:
            dist = self._calculate_distance(item["bbox"], bbox)
            if dist < THRESHOLD_DISTANCE_PIXELS:
                print(f"[{self.camera_id}] COOLDOWN: Abaikan track {track_id}, "
                      f"kendaraan baru saja keluar")
                return None, is_new_track, was_relinked, was_return

        # CAM2 memantau mobil parkir; kotak mobil tetangga bisa tampak seperti fragmen pintu.
        if DOOR_MERGE_MIN_CONTAIN > 0 and self.camera_id != "cam2":
            _ba = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            _door_cands = []
            for _dv_id, _dv in self.vehicles.items():
                if _dv.get("session_closed"):
                    continue
                if has_other_visible_track(_dv):
                    continue
                if not class_compatible(_dv.get("final_cls", cls), cls):
                    continue
                _lb = _dv.get("last_bbox")
                if not _lb:
                    continue
                if timestamp - _dv.get("last_seen", timestamp) > STALE_RELINK_GRACE_SECONDS:
                    continue
                _iw = max(0.0, min(_lb[2], bbox[2]) - max(_lb[0], bbox[0]))
                _ih = max(0.0, min(_lb[3], bbox[3]) - max(_lb[1], bbox[1]))
                _la = max(1.0, (_lb[2] - _lb[0]) * (_lb[3] - _lb[1]))
                _contain = (_iw * _ih) / min(_ba, _la)
                if _contain >= DOOR_MERGE_MIN_CONTAIN:
                    _door_cands.append((_contain, _dv_id, _dv))
            if _door_cands:
                _door_cands.sort(key=lambda c: c[0], reverse=True)
                if len(_door_cands) == 1 or (_door_cands[0][0] - _door_cands[1][0]) >= 0.15:
                    _c, _dv_id, _dv = _door_cands[0]
                    print(f"[{self.camera_id}] DOOR-MERGE: Track baru {track_id} berada di dalam/menutupi kotak "
                          f"{_dv_id} (keterkungkungan={_c:.2f}) -> bagian dari kendaraan yang sama "
                          f"(pintu/bak terbuka), tidak dibuat sesi baru")
                    self._attach_track(_dv_id, track_id, timestamp)
                    _dv["last_bbox"] = bbox
                    was_relinked = True
                    return _dv, is_new_track, was_relinked, was_return

        OVERLAP_MERGE_MIN_IOU = 0.1 if self.single_lane_mode else 0.45
        overlap_candidates = []
        for v_id, vehicle in self.vehicles.items():
            if vehicle.get("session_closed"):
                continue
            if has_other_visible_track(vehicle):
                continue
            if not class_compatible(vehicle.get("final_cls", cls), cls):
                continue
            last_box = vehicle.get("last_bbox")
            if not last_box:
                continue
            if timestamp - vehicle.get("last_seen", timestamp) > STALE_RELINK_GRACE_SECONDS:
                continue

            iou = bbox_iou(last_box, bbox)
            if self.camera_id == "cam2":
                similarity = vehicle_appearance_similarity(vehicle)
                if similarity is None or similarity < REVISIT_MIN_APPEARANCE_SCORE:
                    continue
            if iou > OVERLAP_MERGE_MIN_IOU:
                overlap_candidates.append((iou, v_id, vehicle))

        if overlap_candidates:
            overlap_candidates.sort(key=lambda t: t[0], reverse=True)
            top_iou, top_v_id, top_vehicle = overlap_candidates[0]
            ambiguous = (
                len(overlap_candidates) > 1
                and (top_iou - overlap_candidates[1][0]) < 0.1
            )
            if not ambiguous:
                print(f"[{self.camera_id}] OVERLAP: Track baru {track_id} "
                      f"overlap {top_iou:.2f} dengan {top_v_id}, gabungkan")
                self._attach_track(top_v_id, track_id, timestamp)
                top_vehicle["last_bbox"] = bbox
                was_relinked = True
                return top_vehicle, is_new_track, was_relinked, was_return
            else:
                print(f"[{self.camera_id}] OVERLAP AMBIGU: track {track_id} "
                      f"punya {len(overlap_candidates)} kandidat overlap mirip kuat -- "
                      f"dianggap kendaraan BARU demi keamanan (hindari salah gabung)")

        if CAM2_FRAGMENT_GUARD and self.camera_id == "cam2":
            _frag_of = self._cam2_fragment_of(bbox, cls, timestamp)
            if _frag_of is not None:
                _frg_key = ("FRG", track_id, _frag_of)
                if _frg_key not in self._stale_relink_logged:
                    self._stale_relink_logged.add(_frg_key)
                    print(f"[{self.camera_id}] FRAGMEN-DIABAIKAN: track {track_id} berada di dalam kotak {_frag_of} "
                          f"(kotak={tuple(int(x) for x in bbox)}) -> pecahan mobil yang sama, tidak dibuat sesi baru")
                return None, is_new_track, was_relinked, was_return

        if self.single_lane_mode:
            open_vehicles = [
                (v_id, vehicle) for v_id, vehicle in self.vehicles.items()
                if not vehicle.get("session_closed")
                and class_compatible(vehicle.get("final_cls", cls), cls)
            ]
            if len(open_vehicles) == 1:
                v_id, vehicle = open_vehicles[0]
                gap = timestamp - vehicle.get("last_seen", timestamp)

                _open_last_box = vehicle.get("last_bbox")
                _dist = (
                    self._calculate_distance(_open_last_box, bbox)
                    if _open_last_box else None
                )
                _max_dist = THRESHOLD_DISTANCE_PIXELS * 2.0

                _own_tracks = set(vehicle.get("active_tracks") or [])
                _frame_ids = this_frame_track_ids or set()
                _truly_simultaneous = bool((_own_tracks - {track_id}) & _frame_ids)

                _gap_ok = gap <= SINGLE_LANE_MERGE_MAX_GAP_SECONDS
                _dist_ok = (_dist is None) or (_dist < _max_dist)

                if _truly_simultaneous:
                    print(f"[{self.camera_id}] SINGLE-LANE MERGE DITOLAK: Track baru {track_id} "
                          f"TERBUKTI muncul barengan sesi terbuka {v_id} (track lama sesi ini "
                          f"masih terdeteksi di frame yang SAMA) -- ini beneran objek lain, "
                          f"BUKAN truk yang sama kepecah oklusi. Diperlakukan sebagai kandidat "
                          f"terpisah, tidak digabung ke recent_crops truk.")
                elif _gap_ok and _dist_ok:
                    _dist_txt = f"{_dist:.0f}px" if _dist is not None else "n/a"
                    print(f"[{self.camera_id}] SINGLE-LANE MERGE: Track baru {track_id} "
                          f"digabung paksa ke sesi terbuka {v_id} (gap deteksi={gap:.1f}s, "
                          f"jarak={_dist_txt}, single_lane_mode aktif -> asumsikan truk yang sama)")
                    self._attach_track(v_id, track_id, timestamp)
                    vehicle["last_bbox"] = bbox
                    vehicle["last_seen"] = timestamp
                    was_relinked = True
                    return vehicle, is_new_track, was_relinked, was_return
                elif not _gap_ok:
                    print(f"[{self.camera_id}] SINGLE-LANE MERGE DITOLAK: Track baru {track_id} "
                          f"muncul {gap:.1f}s setelah sesi terbuka {v_id} terakhir terlihat "
                          f"(batas {SINGLE_LANE_MERGE_MAX_GAP_SECONDS:.1f}s) -- terlalu lama "
                          f"untuk sekadar oklusi, kemungkinan besar ini KENDARAAN BERIKUTNYA. "
                          f"Dibuat sebagai kendaraan baru; sesi {v_id} dibiarkan ditutup "
                          f"normal oleh check_lost_tracks().")
                else:
                    print(f"[{self.camera_id}] SINGLE-LANE MERGE DITOLAK: Track baru {track_id} "
                          f"jaraknya {_dist:.0f}px dari posisi terakhir sesi terbuka {v_id} "
                          f"(batas {_max_dist:.0f}px) -- terlalu jauh untuk truk yang sama, "
                          f"dibuat sebagai kendaraan baru.")
            elif len(open_vehicles) > 1:
                _detail = ", ".join(
                    f"{v}(umur={timestamp - vv.get('first_seen', timestamp):.0f}s, "
                    f"diam={timestamp - vv.get('last_seen', timestamp):.0f}s)"
                    for v, vv in open_vehicles
                )
                print(f"[{self.camera_id}] WARNING: single_lane_mode aktif tapi ada "
                      f"{len(open_vehicles)} sesi terbuka sekaligus -> {_detail} "
                      f"-- cek apakah ada sesi yang gagal KELUAR.")

                _frame_ids = this_frame_track_ids or set()
                _best_v_id = None
                _best_vehicle = None
                _best_norm_dist = None
                for _v_id, _vehicle in open_vehicles:
                    _gap = timestamp - _vehicle.get("last_seen", timestamp)
                    _own_tracks = set(_vehicle.get("active_tracks") or [])
                    if (_own_tracks - {track_id}) & _frame_ids:
                        continue
                    if _gap > SINGLE_LANE_MERGE_MAX_GAP_SECONDS:
                        continue
                    _last_box = _vehicle.get("last_bbox")
                    if not _last_box:
                        continue
                    _d = self._calculate_distance(_last_box, bbox)
                    _max_d = THRESHOLD_DISTANCE_PIXELS * 2.0
                    if _d >= _max_d:
                        continue
                    _norm_d = _d / _max_d
                    if _best_norm_dist is None or _norm_d < _best_norm_dist:
                        _best_norm_dist = _norm_d
                        _best_v_id = _v_id
                        _best_vehicle = _vehicle

                if _best_v_id is not None:
                    print(f"[{self.camera_id}] SINGLE-LANE MERGE (multi-sesi terbuka): "
                          f"Track baru {track_id} digabung ke kandidat terbaik "
                          f"{_best_v_id} dari {len(open_vehicles)} sesi terbuka "
                          f"(jarak ternormalisasi={_best_norm_dist:.2f})")
                    self._attach_track(_best_v_id, track_id, timestamp)
                    _best_vehicle["last_bbox"] = bbox
                    _best_vehicle["last_seen"] = timestamp
                    was_relinked = True
                    return _best_vehicle, is_new_track, was_relinked, was_return
                else:
                    print(f"[{self.camera_id}] SINGLE-LANE MERGE (multi-sesi terbuka) "
                          f"DITOLAK untuk semua {len(open_vehicles)} sesi -- tidak ada "
                          f"yang cocok gap+jarak, track {track_id} dibuat sebagai "
                          f"kendaraan baru.")

        vehicle = self._new_vehicle(track_id, timestamp, cls, bbox)
        return vehicle, is_new_track, was_relinked, was_return

    def _cam2_fragment_of(self, bbox, cls, timestamp):
        """Kembalikan id sesi mobil parkir yang kotaknya MELIPUTI bbox (bbox = pecahan), atau None."""
        ba = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        for v_id, v in self.vehicles.items():
            if not class_compatible(v.get("final_cls", cls), cls):
                continue
            lb = v.get("last_bbox")
            if not lb:
                continue
            if v.get("session_closed"):
                t_out = v.get("time_out")
                if t_out is None or not v.get("is_stationary") or timestamp - t_out > PARKED_HOLD_SECONDS:
                    continue
            elif timestamp - v.get("last_seen", timestamp) > max(STALE_RELINK_GRACE_SECONDS, STATIONARY_LOST_SECONDS):
                continue
            la = max(1.0, (lb[2] - lb[0]) * (lb[3] - lb[1]))
            if ba > CAM2_FRAGMENT_MAX_AREA_RATIO * la:
                continue
            iw = max(0.0, min(lb[2], bbox[2]) - max(lb[0], bbox[0]))
            ih = max(0.0, min(lb[3], bbox[3]) - max(lb[1], bbox[1]))
            if (iw * ih) / ba >= CAM2_FRAGMENT_MIN_CONTAIN:
                return v_id
        return None

    def _new_vehicle(self, track_id, timestamp, cls, bbox):
        vehicle_id = self._new_vehicle_id()
        self.vehicles[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "camera_id": self.camera_id,
            "entered": False,
            "entry_event_sent": False,
            "time_in": None,
            "time_out": None,
            "duration_seconds": 0,
            "cargo_in": None,
            "cargo_out": None,
            "activity_type": None,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "last_bbox": bbox,
            "last_crop": None,
            "final_cls": cls,
            "final_conf": 0.0,
            "cls_votes": {},
            "cls_confs": {},
            "cls_weights": {},
            "locked_cls": None,
            "cargo_in_latched": None,
            "last_moment_cargo": None,
            "is_box_body": False,
            "box_body_hits": 0,
            "active_tracks": set(),
            "track_aliases": [],
            "last_track_id": track_id,
            "session_closed": False,
            "direction_history": [],
            "appearance_sig": None,
            "cam2_appearance_sig": None,
            "visit_no": 1,
            "recent_crops": [],
            "cargo_entry_crops": [],
            "cargo_history": deque(maxlen=CARGO_HISTORY_SIZE),
            "motion_last_center": None,
            "motion_last_bbox_size": None,
            "stationary_since": None,
            "is_stationary": False,
        }
        self._attach_track(vehicle_id, track_id, timestamp)
        return self.vehicles[vehicle_id]

    def _attach_track(self, vehicle_id, track_id, timestamp):
        self.track_to_vehicle[track_id] = vehicle_id
        vehicle = self.vehicles[vehicle_id]
        vehicle["active_tracks"].add(track_id)
        vehicle["last_track_id"] = track_id

        if track_id not in vehicle["track_aliases"]:
            vehicle["track_aliases"].append(track_id)
            if len(vehicle["track_aliases"]) > MAX_TRACK_ALIASES:
                for old_tid in vehicle["track_aliases"]:
                    if old_tid not in vehicle["active_tracks"]:
                        vehicle["track_aliases"].remove(old_tid)
                        break

        self.tracks[track_id] = {
            "vehicle_id": vehicle_id,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "last_bbox": vehicle["last_bbox"],
        }

    def _detach_track(self, track_id):
        vehicle_id = self.track_to_vehicle.pop(track_id, None)
        self.tracks.pop(track_id, None)
        if vehicle_id and vehicle_id in self.vehicles:
            self.vehicles[vehicle_id]["active_tracks"].discard(track_id)
        _FRAME_COUNTERS.pop(f"{self.camera_id}_{track_id}", None)

    def _update_motion_state(self, vehicle, bbox, timestamp):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        bbox_scale = max(w, h)

        prev_center = vehicle.get("motion_last_center")
        is_moving_now = True

        if prev_center is not None:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            disp_ratio = math.hypot(dx, dy) / bbox_scale
            is_moving_now = disp_ratio > STATIONARY_MOTION_RATIO_MAX

        vehicle["motion_last_center"] = (cx, cy)
        vehicle["motion_last_bbox_size"] = (w, h)

        if is_moving_now:
            vehicle["stationary_since"] = None
            vehicle["is_stationary"] = False
        else:
            if vehicle.get("stationary_since") is None:
                vehicle["stationary_since"] = timestamp
            stationary_duration = timestamp - vehicle["stationary_since"]
            vehicle["is_stationary"] = stationary_duration >= STATIONARY_PHOTO_MIN_SECONDS

    def update_vehicle(self, vehicle, track_id, cls, conf, bbox, timestamp, crop=None, frame_area=None, has_plate_snapshot=False, roi_h=None, crop_origin=None, neighbor_boxes=None):
        if not vehicle:
            return

        vehicle["last_seen"] = timestamp
        vehicle["last_bbox"] = bbox
        vehicle["last_track_id"] = track_id

        self._update_motion_state(vehicle, bbox, timestamp)


        if crop is not None:
            vehicle["last_crop"] = crop
            update_appearance_sig(vehicle, crop)
            if self.camera_id == "cam2":
                update_cam2_appearance_sig(vehicle, crop)
            _update_color_vote(vehicle, crop)
            if ACTIVITY_MODE and ACTIVITY_FORKLIFT_ONLY_CONFIRM == "bak":
                _record_bed_change(vehicle, crop, timestamp)

            crop_h, crop_w = crop.shape[:2]
            aspect_ratio = crop_w / max(crop_h, 1)

            is_usable_cargo = False
            if self.cargo_detection:
                is_usable_cargo = aspect_ratio <= CARGO_ASPECT_RATIO_MAX

                _cargo_too_small = (
                    min(crop_w, crop_h) < CARGO_MIN_CROP_SIDE
                    or (crop_w * crop_h) < CARGO_MIN_CROP_PIXELS
                )
                if _cargo_too_small:
                    if is_usable_cargo and DEBUG_MODE:
                        print(
                            f"[DEBUG-CARGO-SIZE-v1] {vehicle.get('vehicle_id')}: "
                            f"crop DITOLAK usable_cargo, terlalu kecil "
                            f"(w={crop_w} h={crop_h}, min sisi={CARGO_MIN_CROP_SIDE}, "
                            f"min piksel={CARGO_MIN_CROP_PIXELS})"
                        )
                    is_usable_cargo = False

                if not is_usable_cargo and not _cargo_too_small and DEBUG_MODE:
                    print(
                        f"[DEBUG-CARGO-ASPECT-v1] {vehicle.get('vehicle_id')}: "
                        f"crop DITOLAK usable_cargo, aspect_ratio={aspect_ratio:.2f} "
                        f"(w={crop_w} h={crop_h}) > {CARGO_ASPECT_RATIO_MAX}"
                    )

                if not is_usable_cargo:
                    last_usable_ts = vehicle.get(
                        "last_usable_cargo_crop_ts", vehicle.get("first_seen", timestamp)
                    )
                    time_since_usable = timestamp - last_usable_ts
                    if (
                        not _cargo_too_small
                        and time_since_usable >= CARGO_ASPECT_RELAX_SECONDS
                        and aspect_ratio <= CARGO_ASPECT_RATIO_RELAXED_MAX
                    ):
                        is_usable_cargo = True
                        if DEBUG_MODE:
                            print(
                                f"[FIX-CAM1-CARGO-ASPECT-ADAPTIVE-v1] {vehicle.get('vehicle_id')}: "
                                f"ambang dilunakkan jadi {CARGO_ASPECT_RATIO_RELAXED_MAX} "
                                f"(sudah {time_since_usable:.1f}s sejak crop cargo usable terakhir, "
                                f"aspect_ratio={aspect_ratio:.2f})"
                            )

                if is_usable_cargo:
                    vehicle["ever_usable_cargo_crop"] = True
                    vehicle["last_usable_cargo_crop_ts"] = timestamp

                if is_usable_cargo:
                    try:
                        _gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        _texture_var = cv2.Laplacian(_gray, cv2.CV_64F).var()
                        _edges = cv2.Canny(_gray, 50, 150)
                        _edge_density = float(np.sum(_edges > 0)) / _edges.size
                        if _texture_var < 30 and _edge_density < 0.02:
                            is_usable_cargo = False
                            if DEBUG_MODE:
                                print(
                                    f"[DEBUG-CARGO-ASPECT-v1] {vehicle.get('vehicle_id')}: "
                                    f"crop DITOLAK usable_cargo, tekstur terlalu flat "
                                    f"(texture_var={_texture_var:.1f}, edge_density={_edge_density:.3f})"
                                )
                    except Exception:
                        pass

            is_usable_plate = False
            if self.plate_detection and self.read_own_plate:
                if not has_plate_snapshot:
                    is_usable_plate = crop_w >= 20 and crop_h >= 20
                else:
                    snap_success = vehicle.get("plate_snapshot_success_count", 0)
                    snap_attempts = vehicle.get("plate_snapshot_count", 0)
                    snapshot_exhausted_without_success = (
                        snap_success == 0
                        and snap_attempts >= PLATE_SNAPSHOT_MAX_TOTAL_ATTEMPTS
                    )
                    if not vehicle.get("plate_confirmed") and snapshot_exhausted_without_success:
                        is_usable_plate = crop_w >= 20 and crop_h >= 20
                        if DEBUG_MODE and is_usable_plate:
                            print(
                                f"[FIX-PLATE-SNAPSHOT-EXHAUSTED-FALLBACK-v1] "
                                f"{vehicle.get('vehicle_id')}: snapshot habis "
                                f"({snap_attempts}x percobaan, 0 berhasil) -- "
                                f"izinkan crop video jadi usable_plate sbg upaya terakhir."
                            )

            is_usable = is_usable_cargo or is_usable_plate or (not self.cargo_detection and crop_w >= 80 and crop_h >= 60)

            sharpness = 0.0
            if is_usable:
                try:
                    _gray_s = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(_gray_s, cv2.CV_64F).var())
                except Exception:
                    sharpness = 0.0

            if is_usable:
                area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

                area_for_score = (area / frame_area) if frame_area else area

                # [FIX-CAM2-EDGE] penalti crop yang terpotong tepi ROI (hanya non-cam1)
                edge_factor = 1.0
                display_crop = None
                if roi_h and frame_area and getattr(self, "camera_id", "") != "cam1":
                    _roi_w_ = frame_area / roi_h
                    if _edge_cut_factor(bbox, _roi_w_, roi_h) < 1.0:
                        # bbox menempel batas ROI: foto diambil dari frame penuh; penalti hanya bila menempel tepi frame penuh
                        display_crop, edge_factor = _display_crop_from_full(self.camera_id, bbox, (_roi_w_, roi_h))
                        if display_crop is None:
                            edge_factor = _edge_cut_factor(bbox, _roi_w_, roi_h)
                    area_for_score *= edge_factor

                score = area_for_score * (0.5 + conf)

                y_ratio = None
                if roi_h:
                    y_ratio = max(0.0, min(1.0, bbox[3] / roi_h))
                foreshorten_penalty = 1.0
                if y_ratio is not None and y_ratio > 0.55:
                    foreshorten_penalty = 1.0 - 0.65 * ((y_ratio - 0.55) / 0.45)

                plate_score = area_for_score * (1.0 + min(sharpness, 500.0) / 500.0) * foreshorten_penalty * _view_factor_bbox(bbox)
                with self._crop_lock:
                    _new_crop_entry = {
                        "crop": crop, "conf": conf, "area": area, "timestamp": timestamp,
                        "score": score, "sharpness": sharpness, "plate_score": plate_score,
                        "y_ratio": y_ratio, "edge_factor": edge_factor, "display_crop": display_crop,
                        "usable_cargo": is_usable_cargo, "usable_plate": is_usable_plate,
                        "is_stationary": vehicle.get("is_stationary", False),
                        "own_bbox": bbox, "crop_origin": crop_origin,
                        "neighbor_boxes": neighbor_boxes,
                    }
                    vehicle["recent_crops"].append(_new_crop_entry)
                    _record_photo(vehicle, display_crop if display_crop is not None else crop, timestamp, area * edge_factor, sharpness, is_usable_cargo)
                    # [FIX-ACT-CROPS] Buffer terpisah untuk activity_estimator: berisi crop cargo TERBARU, tanpa eviction
                    # berdasarkan skor. recent_crops membuang crop berskor terendah, sehingga crop baru yang lebih kecil/
                    # tertutup langsung terbuang dan sampel "saat pergi" tidak pernah terkumpul.
                    if is_usable_cargo:
                        _ac = vehicle.setdefault("act_crops", [])
                        _ac.append(_new_crop_entry)
                        if len(_ac) > 3:
                            del _ac[0]
                    if (
                        is_usable_cargo
                        and not vehicle.get("entered")
                        and len(vehicle.get("cargo_entry_crops") or []) < CARGO_ENTRY_CROPS_MAX
                    ):
                        vehicle.setdefault("cargo_entry_crops", []).append(_new_crop_entry)
                    RECENT_CROPS_MAX = 10
                    if len(vehicle["recent_crops"]) > RECENT_CROPS_MAX:
                        _evict_worst_crop(vehicle["recent_crops"])

        if track_id in self.tracks:
            self.tracks[track_id]["last_seen"] = timestamp
            self.tracks[track_id]["last_bbox"] = bbox

        if "direction_history" not in vehicle:
            vehicle["direction_history"] = []
        vehicle["direction_history"].append((timestamp, bbox))
        cutoff = timestamp - DIRECTION_WINDOW_SECONDS
        vehicle["direction_history"] = [
            (t, b) for t, b in vehicle["direction_history"] if t >= cutoff
        ]

        if "cls_votes" not in vehicle:
            vehicle["cls_votes"] = {}
        if "cls_confs" not in vehicle:
            vehicle["cls_confs"] = {}
        if "final_conf" not in vehicle:
            vehicle["final_conf"] = 0.0
        if "final_cls" not in vehicle:
            vehicle["final_cls"] = cls

        MIN_CLS_VOTE_AREA_RATIO = 0.02
        bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        cls_vote_area_ratio = (bbox_area / frame_area) if frame_area else None

        if cls_vote_area_ratio is not None:
            vehicle["best_area_ratio"] = max(
                vehicle.get("best_area_ratio", 0.0), cls_vote_area_ratio
            )

        cls_vote_allowed = (
            cls_vote_area_ratio is None or cls_vote_area_ratio >= MIN_CLS_VOTE_AREA_RATIO
        )

        if not cls_vote_allowed:
            if DEBUG_MODE:
                print(
                    f"[FIX-CAM1-CLS-VOTE-AREA-v1] {vehicle.get('vehicle_id')}: "
                    f"vote cls={cls} DITOLAK, area_ratio={cls_vote_area_ratio:.4f} "
                    f"< {MIN_CLS_VOTE_AREA_RATIO} (kendaraan masih terlalu kecil/jauh)"
                )
        else:
            vehicle["cls_votes"][cls] = vehicle["cls_votes"].get(cls, 0) + 1
            if cls not in vehicle["cls_confs"]:
                vehicle["cls_confs"][cls] = []
            vehicle["cls_confs"][cls].append(conf)

            if "cls_weights" not in vehicle:
                vehicle["cls_weights"] = {}
            _w = (cls_vote_area_ratio if cls_vote_area_ratio else 1.0) * max(conf, 0.01)
            vehicle["cls_weights"][cls] = vehicle["cls_weights"].get(cls, 0.0) + _w

            if vehicle["vehicle_id"] not in self.class_history:
                self.class_history[vehicle["vehicle_id"]] = []
            self.class_history[vehicle["vehicle_id"]].append(cls)

            if vehicle["cls_votes"]:
                _locked = vehicle.get("locked_cls")
                vehicle["final_cls"] = (
                    _locked if _locked is not None else _weighted_cls(vehicle, cls)
                )
                confs = vehicle["cls_confs"].get(vehicle["final_cls"], [conf])
                vehicle["final_conf"] = sum(confs) / len(confs) if confs else conf

    def add_plate_snapshot_crop(self, vehicle, crop, conf, timestamp, snap_frame_area=None, y_ratio=None, usable_cargo=False, own_bbox=None, neighbor_boxes=None):
        if crop is None or crop.size == 0:
            return
        crop_h, crop_w = crop.shape[:2]
        if crop_w < 20 or crop_h < 20:
            return

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            sharpness = 0.0

        area = float(crop_w * crop_h)
        area_for_score = (area / snap_frame_area) if snap_frame_area else area

        score = area_for_score * (0.5 + conf)

        foreshorten_penalty = 1.0
        if y_ratio is not None and y_ratio > 0.55:
            foreshorten_penalty = 1.0 - 0.65 * ((y_ratio - 0.55) / 0.45)

        area_score_component = math.sqrt(max(area_for_score, 0.0))
        plate_score = (
            area_score_component
            * (1.0 + min(sharpness, 500.0) / 500.0)
            * foreshorten_penalty
        )
        plate_score *= 1.15

        with self._crop_lock:
            vehicle.setdefault("recent_crops", [])
            _new_crop_entry2 = {
                "crop": crop, "conf": conf, "area": area, "timestamp": timestamp,
                "score": score, "sharpness": sharpness, "plate_score": plate_score,
                "usable_cargo": usable_cargo, "usable_plate": True,
                "from_snapshot": True,
                "own_bbox": own_bbox,
                "crop_origin": own_bbox[:2] if own_bbox else None,
                "neighbor_boxes": neighbor_boxes,
            }
            vehicle["recent_crops"].append(_new_crop_entry2)
            if (
                usable_cargo
                and not vehicle.get("entered")
                and len(vehicle.get("cargo_entry_crops") or []) < CARGO_ENTRY_CROPS_MAX
            ):
                vehicle.setdefault("cargo_entry_crops", []).append(_new_crop_entry2)
            vehicle["plate_snapshot_success_count"] = (
                vehicle.get("plate_snapshot_success_count", 0) + 1
            )
            RECENT_CROPS_MAX = 10
            if len(vehicle["recent_crops"]) > RECENT_CROPS_MAX:
                _evict_worst_crop(vehicle["recent_crops"])

    def mark_entered(self, vehicle, status_muatan, timestamp):
        if vehicle.get("entered"):
            return
        vehicle["entered"] = True
        vehicle["time_in"] = timestamp
        vehicle["cargo_in"] = status_muatan
        print(f"[{self.camera_id}] SESI KUNCI MASUK: {vehicle['vehicle_id']} | "
              f"Muatan Awal: {status_muatan}")

    def finalize_exit(self, vehicle, status_muatan_keluar, timestamp):
        if vehicle.get("session_closed"):
            return False

        vehicle["time_out"] = timestamp
        vehicle["cargo_out"] = status_muatan_keluar
        vehicle["session_closed"] = True

        if vehicle.get("time_in"):
            vehicle["duration_seconds"] = round(timestamp - vehicle["time_in"], 2)
        else:
            vehicle["duration_seconds"] = 0.0

        if ACTIVITY_MODE and dock_events is not None and _CAM_CONFIG.get(self.camera_id, {}).get("dock_events"):
            _cfg_fe = _CAM_CONFIG[self.camera_id]
            try:
                dock_events.finalize_vehicle(self.camera_id, vehicle, timestamp,
                                             lambda v, ev: _emit_activity_event(_cfg_fe, self, v, ev))
            except Exception as _e:
                print(f"[DOCK] finalize gagal (diabaikan): {_e}")

        if ACTIVITY_MODE:
            _act = activity_estimator.decide(vehicle)
            # [FIX-FORKLIFT-HANTU] buang event "lintasan forklift" dari ringkasan sesi (lihat ACTIVITY_ALLOW_FORKLIFT_ONLY)
            if not ACTIVITY_ALLOW_FORKLIFT_ONLY and vehicle.get("timeline_events"):
                _tl_all = vehicle["timeline_events"]
                _tl_ok = [e for e in _tl_all if (not _is_forklift_only_event(e))
                          or (ACTIVITY_FORKLIFT_ONLY_CONFIRM == "bak" and _bed_change_supports(vehicle, e))]
                if len(_tl_ok) != len(_tl_all):
                    print(f"[AKTIVITAS] {vehicle.get('vehicle_id')}: {len(_tl_all) - len(_tl_ok)} event lintasan-forklift tanpa bukti bak dibuang dari ringkasan")
                    vehicle["timeline_events"] = _tl_ok
            try:
                _sum_txt, _sum_info = (dock_events.summarize(vehicle) if dock_events is not None else (None, None))
            except Exception as _e:
                print(f"[DOCK] summarize gagal (diabaikan): {_e}")
                _sum_txt, _sum_info = None, None
            if _sum_info is not None:
                # [MULTI-EVENT] ringkasan sesi = gabungan SEMUA kejadian, bukan satu kesimpulan awal-vs-akhir
                _act = dict(_act)
                _act["activity"] = _sum_info["activity"]
                _act["source"] = _sum_info["source"]
                _act["confidence"] = _sum_info["confidence"]
                _act["reason"] = _sum_info["reason"]
            vehicle["activity_type"] = _act["activity"]
            vehicle["activity_info"] = _act
            _tl = vehicle.get("timeline_events") or []
            if _tl:
                # status muatan keluar mengikuti LABEL RINGKASAN siklus (Loading -> Bermuatan, Unloading -> Kosong)
                _last = _act.get("activity") or _tl[-1]["activity"]  # label ringkasan (dominan), bukan event terakhir yang bisa artefak
                _act["_cargo_out_override"] = CARGO_LOADED if "Loading" in _last else (CARGO_EMPTY if "Unloading" in _last else None)
            # [FIX-CARGO-LABEL] Label muatan MASUK harus sejalan dengan penilaian keadaan-tiba milik activity_estimator
            # (dulu pakai median >= 0.5 sehingga skor 0.502 berlabel "Bermuatan" padahal estimator menilai "kosong",
            # menghasilkan label bertentangan "Bermuatan -> Kosong" dengan aktivitas Loading).
            _ast = _act.get("arrival_state")
            if _act["score_in"] is None:
                vehicle["cargo_in"] = CARGO_UNKNOWN
            elif _ast == "kosong":
                vehicle["cargo_in"] = CARGO_EMPTY
            elif _ast == "penuh":
                vehicle["cargo_in"] = CARGO_LOADED
            elif _ast == "ragu":
                vehicle["cargo_in"] = CARGO_UNKNOWN
            else:
                vehicle["cargo_in"] = CARGO_LOADED if _act["score_in"] >= 0.5 else CARGO_EMPTY
            vehicle["cargo_out"] = (CARGO_UNKNOWN if _act["score_out"] is None else
                                    (CARGO_LOADED if _act["score_out"] >= 0.5 else CARGO_EMPTY))
            if _act.get("_cargo_out_override"):
                vehicle["cargo_out"] = _act["_cargo_out_override"]
            print(f"[ACTIVITY] {vehicle.get('vehicle_id', '?')}: {_act['activity']} "
                  f"({_act['reason']}) skor_tiba={_act['score_in']} "
                  f"skor_pergi={_act['score_out']} delta={_act['delta']} sumber={_act.get('source')} "
                  f"conf={_act['confidence']} sampel={_act['n_in']}/{_act['n_out']}")
        else:
            vehicle["activity_type"] = determine_activity_type(
                vehicle.get("cargo_in"), vehicle.get("cargo_out")
            )

        self.recently_exited.append({
            "time": timestamp,
            "bbox": vehicle.get("last_bbox"),
        })

        return True

    def apply_gap(self, gap, now, grace_seconds=10.0):
        # [GAP-GUARD] Setelah stream putus `gap` detik: (1) jangan tutup sesi dulu supaya kendaraan sempat terdeteksi lagi,
        # (2) kandidat yang belum dikonfirmasi tidak boleh lolos "3 detik terlihat" gara-gara jeda itu.
        self.lost_grace_until = now + grace_seconds
        for v in self.vehicles.values():
            if v.get("session_closed") or v.get("entered"):
                continue
            for k in ("first_seen", "stationary_since"):
                if v.get(k) is not None:
                    v[k] = v[k] + gap

    def cleanup(self, timestamp):
        RETENTION_WINDOW = 1800.0
        self.recently_exited = [
            item for item in self.recently_exited
            if timestamp - item["time"] < RETENTION_WINDOW
        ]

        INACTIVE_THRESHOLD = 60.0
        stale_tracks = [
            t_id for t_id, data in self.tracks.items()
            if timestamp - data["last_seen"] > INACTIVE_THRESHOLD
        ]
        for t_id in stale_tracks:
            self._detach_track(t_id)

        VEHICLE_PURGE_GRACE = REVISIT_RETENTION_EFF + 60.0
        stale_vehicle_ids = [
            v_id for v_id, v in self.vehicles.items()
            if v.get("session_closed")
            and v.get("time_out") is not None
            and timestamp - v["time_out"] > VEHICLE_PURGE_GRACE
        ]
        for v_id in stale_vehicle_ids:
            self.vehicles.pop(v_id, None)
            self.class_history.pop(v_id, None)

        GHOST_VEHICLE_PURGE_SECONDS = 90.0
        ghost_vehicle_ids = [
            v_id for v_id, v in self.vehicles.items()
            if not v.get("entered")
            and not v.get("active_tracks")
            and timestamp - v.get("last_seen", timestamp) > GHOST_VEHICLE_PURGE_SECONDS
        ]
        for v_id in ghost_vehicle_ids:
            self.vehicles.pop(v_id, None)
            self.class_history.pop(v_id, None)


_OUTBOX_LOCK = threading.Lock()
_OUTBOX_TRIES = {}


def _outbox_dead_letter(line, reason):
    """Event yang mustahil terkirim (ditolak 4xx / gagal terus) dipindah ke file *_dead.jsonl
    supaya TIDAK memblokir antrian; bisa dikirim ulang manual setelah masalahnya diperbaiki."""
    dead = os.path.splitext(N8N_OUTBOX_PATH)[0] + "_dead.jsonl"
    try:
        with open(dead, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(f"[N8N] !!! Event dipindah ke '{dead}' ({reason})")
    except Exception as e:
        print(f"[N8N] !!! Gagal menulis dead-letter: {e}")


def _post_n8n(payload, timeout=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        N8N_WEBHOOK,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout or N8N_TIMEOUT_SECONDS) as resp:
        code = getattr(resp, "status", 200)
        resp.read()
    if code >= 300:
        raise RuntimeError(f"HTTP {code}")


def _outbox_append(payload):
    """Simpan event yang gagal terkirim ke file (jsonl) supaya dikirim ulang nanti, bukan hilang."""
    try:
        with _OUTBOX_LOCK:
            if os.path.isfile(N8N_OUTBOX_PATH) and os.path.getsize(N8N_OUTBOX_PATH) > N8N_OUTBOX_MAX_MB * 1024 * 1024:
                print(f"[N8N] !!! Outbox penuh (> {N8N_OUTBOX_MAX_MB:.0f} MB), event DIBUANG: "
                      f"{payload.get('jenis_kejadian')} - {payload.get('vehicle_id')}")
                return
            with open(N8N_OUTBOX_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        print(f"[N8N] Event disimpan ke outbox '{N8N_OUTBOX_PATH}': "
              f"{payload.get('jenis_kejadian')} - {payload.get('vehicle_id')}")
    except Exception as e:
        print(f"[N8N] !!! Gagal menulis outbox: {e}")


def _outbox_flush(max_items=20):
    """Kirim ulang event di outbox (urut dari yang terlama). Berhenti di kegagalan pertama. Return jumlah terkirim."""
    sent = 0
    with _OUTBOX_LOCK:
        if not os.path.isfile(N8N_OUTBOX_PATH) or os.path.getsize(N8N_OUTBOX_PATH) == 0:
            return 0
        with open(N8N_OUTBOX_PATH, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        consumed = 0
        for line in lines[:max_items]:
            try:
                payload = json.loads(line)
            except Exception:
                consumed += 1  # baris rusak dibuang
                continue
            try:
                _post_n8n(payload)
            except Exception as e:
                key = payload.get("event_id") or line[:80]
                tries = _OUTBOX_TRIES.get(key, 0) + 1
                _OUTBOX_TRIES[key] = tries
                code = getattr(e, "code", None)
                if (isinstance(code, int) and 400 <= code < 500) or tries >= N8N_OUTBOX_MAX_TRIES:
                    _outbox_dead_letter(line, f"{e}; percobaan ke-{tries}")
                    _OUTBOX_TRIES.pop(key, None)
                    consumed += 1
                    continue
                print(f"[N8N] Outbox: n8n belum bisa dihubungi ({e}); {len(lines) - consumed} event tertunda")
                break
            _OUTBOX_TRIES.pop(payload.get("event_id") or line[:80], None)
            consumed += 1
            sent += 1
        remaining = lines[consumed:]
        if remaining:
            tmp = N8N_OUTBOX_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(remaining) + "\n")
            os.replace(tmp, N8N_OUTBOX_PATH)
        else:
            os.remove(N8N_OUTBOX_PATH)
    return sent


def _n8n_outbox_worker():
    while True:
        try:
            n = _outbox_flush()
            if n:
                print(f"[N8N] Outbox: {n} event tertunda berhasil dikirim ulang")
        except Exception as e:
            print(f"[N8N] Outbox worker error: {e}")
        time.sleep(N8N_OUTBOX_INTERVAL_SECONDS)


def _eksekusi_kirim_n8n(payload):
    last_err = None
    for attempt in range(1, N8N_RETRY_ATTEMPTS + 1):
        try:
            _post_n8n(payload)
            extra = f" (percobaan ke-{attempt})" if attempt > 1 else ""
            print(f"[N8N] Terkirim: {payload.get('kamera_id')} | "
                  f"{payload.get('jenis_kejadian')} - {payload.get('vehicle_id')}{extra}")
            return
        except Exception as e:
            last_err = e
            if attempt < N8N_RETRY_ATTEMPTS:
                time.sleep(N8N_RETRY_DELAY_SECONDS * attempt)
    print(f"[N8N] Gagal kirim setelah {N8N_RETRY_ATTEMPTS} percobaan ({last_err})")
    _outbox_append(payload)


def kirim_ke_n8n(payload):
    t = threading.Thread(target=_eksekusi_kirim_n8n, args=(payload,), daemon=True)
    t.start()
    with _N8N_THREADS_LOCK:
        _N8N_THREADS[:] = [x for x in _N8N_THREADS if x.is_alive()]  # jangan menumpuk thread selesai
        _N8N_THREADS.append(t)


def tunggu_semua_kirim_n8n(timeout_total=15):
    with _N8N_THREADS_LOCK:
        pending = [t for t in _N8N_THREADS if t.is_alive()]
    if not pending:
        return
    print(f"> Menunggu {len(pending)} pengiriman N8N yang masih berjalan...")
    deadline = time.time() + timeout_total
    for t in pending:
        sisa = max(0, deadline - time.time())
        t.join(timeout=sisa)


def build_payload(config, jenis_kejadian, cls, conf, bbox, status_muatan,
                  crop_enhanced, track_id, vehicle_id, event_id,
                  arah_kendaraan="TIDAK_JELAS", arah_confidence=0.0,
                  plat_nomor=None, plat_confidence=0.0,
                  warna="Tidak Terdeteksi", warna_rgb=(128, 128, 128), event_ts=None):
    x1r, y1r, x2r, y2r = bbox
    # [FIX-CLOCK] SEMUA event (MASUK / AKTIVITAS / SIKLUS_SELESAI) harus memakai jam yang SAMA: jam pipeline `now`.
    # Dulu MASUK memakai datetime.now() (jam dinding) sedangkan AKTIVITAS & KELUAR memakai jam video. Pada uji video
    # (proses ~4x lebih lambat dari real-time) MASUK jadi tampil ~70 dtk lebih lambat dari seharusnya, sehingga Loading
    # tampil SEBELUM Masuk dan Keluar tampil terlalu cepat dibanding Masuk. Di kamera live jam dinding == jam pipeline.
    _ts = event_ts if event_ts is not None else time.time()
    return {
        "timestamp": datetime.fromtimestamp(_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "kamera_id": config["kamera_id"],
        "kamera_nama": config["kamera_nama"],
        "jenis_kendaraan": _jenis_label(cls, config),
        "warna": warna,
        "confidence": round(float(conf), 4),
        "bbox_x1": int(x1r), "bbox_y1": int(y1r),
        "bbox_x2": int(x2r), "bbox_y2": int(y2r),
        "rgb_r": int(warna_rgb[0]), "rgb_g": int(warna_rgb[1]), "rgb_b": int(warna_rgb[2]),
        "gambar_base64": encode_image_base64(crop_enhanced),
        "status_muatan": status_muatan,
        "jenis_kejadian": jenis_kejadian,
        "track_id": track_id,
        "vehicle_id": vehicle_id,
        "event_id": event_id,
        "arah_kendaraan": arah_kendaraan,
        "arah_confidence": round(float(arah_confidence), 3),
        "plat_nomor": plat_nomor,
        "plat_confidence": round(float(plat_confidence), 3),
    }


def open_video_source(source):
    if is_local_video_file(source):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


_CORRUPT_BLOCK_SIZE = 16
_CORRUPT_JUMP_THRESHOLD = 60
_CORRUPT_RATIO_THRESHOLD = 0.35


def _is_frame_corrupted(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        bs = _CORRUPT_BLOCK_SIZE
        if h < bs * 4 or w < bs * 4:
            return False
        small = cv2.resize(gray, (w // bs, h // bs), interpolation=cv2.INTER_AREA).astype(np.int16)
        dx = np.abs(np.diff(small, axis=1))
        dy = np.abs(np.diff(small, axis=0))
        total = dx.size + dy.size
        if total == 0:
            return False
        extreme = int(np.count_nonzero(dx > _CORRUPT_JUMP_THRESHOLD)) + \
                  int(np.count_nonzero(dy > _CORRUPT_JUMP_THRESHOLD))
        return (extreme / total) > _CORRUPT_RATIO_THRESHOLD
    except Exception:
        return False


_NOISE_BLOCK_SIZE = 8
_NOISE_VARIANCE_THRESHOLD = env("NOISE_VARIANCE_THRESHOLD", float, required=False) or 250.0

NOISE_CORRUPT_CHECK_ENABLED = env("NOISE_CORRUPT_CHECK_ENABLED", bool, required=False) or False


def _is_frame_noisy_corrupt(frame, block_size=_NOISE_BLOCK_SIZE,
                             min_variance_threshold=_NOISE_VARIANCE_THRESHOLD,
                             debug_label=None):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = gray.astype(np.float32)
        h, w = gray.shape[:2]
        if h < block_size * 4 or w < block_size * 4:
            return False

        mean = cv2.boxFilter(gray, ddepth=-1, ksize=(block_size, block_size), normalize=True)
        mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(block_size, block_size), normalize=True)
        local_var = np.clip(mean_sq - mean * mean, 0, None)

        grid = local_var[::block_size, ::block_size]
        if grid.size == 0:
            return False
        p25_variance = float(np.percentile(grid, 25))

        if debug_label and DEBUG_MODE:
            print(f"[NOISE-CHECK] {debug_label}: varians-lokal persentil-25 = "
                  f"{p25_variance:.1f} (ambang={min_variance_threshold})")

        return p25_variance > min_variance_threshold
    except Exception:
        return False

_NOISE_DEBUG_LOG_ALWAYS = env("NOISE_DEBUG_LOG_ALWAYS", bool, required=False) or False

RECONNECT_GRACE_FRAMES = 15
# [GAP-GUARD] Jeda antar frame valid yang lebih lama dari ini dianggap stream putus/frame rusak (bukan kendaraan hilang).
GAP_GUARD_SECONDS = float(os.environ.get("GAP_GUARD_SECONDS", "4.0"))
GAP_GUARD_GRACE_SECONDS = float(os.environ.get("GAP_GUARD_GRACE_SECONDS", "10.0"))


def extract_roi(frame, roi_config):
    h_full, w_full = frame.shape[:2]
    rx1 = int(w_full * roi_config["x_min"])
    ry1 = int(h_full * roi_config["y_min"])
    rx2 = int(w_full * roi_config["x_max"])
    ry2 = int(h_full * roi_config["y_max"])
    return frame[ry1:ry2, rx1:rx2].copy()


def run_inference(model, roi_frame, classes_allowed, confidence_min=CONFIDENCE_MIN):
    roi_resized, scale = prepare_roi_for_inference(roi_frame)

    results = model.track(
        roi_resized,
        imgsz=INFER_TARGET_WIDTH,
        conf=confidence_min,
        iou=IOU_THRESHOLD,
        classes=classes_allowed,
        persist=True,
        verbose=False,
        tracker=TRACKER_CONFIG,
    )
    result = results[0] if results else None
    return result, scale

def fetch_snapshot(channel, timeout=4):
    rstp_url = _build_rtsp_url(None, channel)
    cap = cv2.VideoCapture(rstp_url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            print(f"[snapshot-rstp] Gagal buka RTSP channel {channel}")
            return None

        frame = None
        for _ in range(RTSP_SNAPSHOT_WARMUP_FRAMES):
            ok, f = cap.read()
            if not ok or f is None:
                continue
            if _is_frame_corrupted(f):
                continue
            frame = f

        if frame is None:
            print(f"[snapshot-rtsp] Semua frame warm-up channel {channel} gagal/corrupt")
            return None

        return frame
    except Exception as e:
        print(f"[snapshot-rtsp] Gagal ambil snapshot channel {channel}: {e}")
        return None
    finally:
        cap.release()

def _plate_snapshot_worker(tracks, vehicle, snap_channel, roi_config, bbox, roi_w, roi_h, conf, timestamp, cam_name, debug_label, cargo_detection_enabled=False):

    snap_frame = fetch_snapshot(snap_channel)
    if snap_frame is None or _is_frame_corrupted(snap_frame):
        return

    fresh_bbox = vehicle.get("last_bbox", bbox)
    snap_crop, _ = crop_from_snapshot(snap_frame, roi_config, fresh_bbox, roi_w, roi_h)
    if snap_crop is None or _is_frame_corrupted(snap_crop):
        return

    if NOISE_CORRUPT_CHECK_ENABLED or _NOISE_DEBUG_LOG_ALWAYS:
        _snap_noisy = _is_frame_noisy_corrupt(
            snap_crop,
            debug_label=f"{cam_name}/snapshot/{vehicle.get('vehicle_id')}" if _NOISE_DEBUG_LOG_ALWAYS else None,
        )
        if NOISE_CORRUPT_CHECK_ENABLED and _snap_noisy:
            if DEBUG_MODE:
                print(f"[{cam_name}] Skip Snapshot (noise/dither lokal): "
                    f"{vehicle.get('vehicle_id')}, varians piksel lokal tidak wajar")
            return
    snap_h, snap_w = snap_frame.shape[:2]

    own_tracks_for_vehicle = set(vehicle.get("active_tracks") or [])
    own_bbox_snapshot = _map_roi_bbox_to_snapshot_coords(
        fresh_bbox, roi_config, roi_w, roi_h, snap_w, snap_h
    )
    neighbor_boxes_snapshot = []
    for tid, tinfo in tracks.tracks.items():
        if tid in own_tracks_for_vehicle:
            continue
        if timestamp - tinfo.get("last_seen", 0) > 2.0 or not tinfo.get("last_bbox"):
            continue
        mapped = _map_roi_bbox_to_snapshot_coords(
            tinfo["last_bbox"], roi_config, roi_w, roi_h, snap_w, snap_h
        )
        if mapped is not None:
            neighbor_boxes_snapshot.append(mapped)
    y_ratio = None
    if roi_h:
        y_ratio = max(0.0, min(1.0, fresh_bbox[3] / roi_h))

    usable_cargo = cargo_detection_enabled and _is_usable_cargo_crop(snap_crop)

    tracks.add_plate_snapshot_crop(vehicle, snap_crop, conf, timestamp,
                                    snap_frame_area=(snap_w * snap_h),
                                    y_ratio=y_ratio,
                                    usable_cargo=usable_cargo,
                                    own_bbox=own_bbox_snapshot,
                                    neighbor_boxes=neighbor_boxes_snapshot)
    if DEBUG_MODE:
        if cargo_detection_enabled:
            print(f"[{cam_name}] snapshot {debug_label}: usable_cargo={usable_cargo} "
                  f"(crop {snap_crop.shape[1]}x{snap_crop.shape[0]} dari mainstream, "
                  f"vs video substream yang jauh lebih kecil)")
        print(f"[{cam_name}] snapshot plat {debug_label} berhasil diambil "
              f"({snap_crop.shape[1]}x{snap_crop.shape[0]}) dari snap_frame "
              f"penuh {snap_w}x{snap_h} (channel={snap_channel})")
        print(
            f"[DIAG-PLATE-RACE-v1] {vehicle.get('vehicle_id')}: "
            f"id(vehicle)={id(vehicle)}, len(recent_crops setelah tambah)="
            f"{len(vehicle.get('recent_crops') or [])}"
        )


def _map_roi_bbox_to_snapshot_coords(bbox, roi_config, roi_w, roi_h, snap_w, snap_h, pad_ratio=0.02):
    try:
        x1r, y1r, x2r, y2r = bbox

        fx1, fy1 = x1r / roi_w, y1r / roi_h
        fx2, fy2 = x2r / roi_w, y2r / roi_h

        rx0, rx1 = roi_config["x_min"], roi_config["x_max"]
        ry0, ry1 = roi_config["y_min"], roi_config["y_max"]

        full_x1 = rx0 + fx1 * (rx1 - rx0) - pad_ratio
        full_x2 = rx0 + fx2 * (rx1 - rx0) + pad_ratio
        full_y1 = ry0 + fy1 * (ry1 - ry0) - pad_ratio
        full_y2 = ry0 + fy2 * (ry1 - ry0) + pad_ratio

        sx1 = max(0, int(full_x1 * snap_w))
        sy1 = max(0, int(full_y1 * snap_h))
        sx2 = min(snap_w, int(full_x2 * snap_w))
        sy2 = min(snap_h, int(full_y2 * snap_h))

        if sx2 <= sx1 or sy2 <= sy1:
            return None
        return (sx1, sy1, sx2, sy2)
    except Exception:
        return None


def crop_from_snapshot(snap_frame, roi_config, bbox, roi_w, roi_h, pad_ratio=0.02):
    snap_h, snap_w = snap_frame.shape[:2]
    mapped = _map_roi_bbox_to_snapshot_coords(bbox, roi_config, roi_w, roi_h, snap_w, snap_h, pad_ratio)
    if mapped is None:
        return None, None
    sx1, sy1, sx2, sy2 = mapped
    return snap_frame[sy1:sy2, sx1:sx2].copy(), (sx1, sy1)


def make_crop(roi_frame, roi_w, roi_h, bbox, pad=20, extra_top_ratio=0.10):
    x1r, y1r, x2r, y2r = bbox

    height = max(0.0, y2r - y1r)
    top_pad = pad + (height * extra_top_ratio)

    x1_c = int(max(0, x1r - pad))
    y1_c = int(max(0, y1r - top_pad))
    x2_c = int(min(roi_w, x2r + pad))
    y2_c = int(min(roi_h, y2r + pad))
    if x2_c <= x1_c or y2_c <= y1_c:
        return None, None
    return roi_frame[y1_c:y2_c, x1_c:x2_c].copy(), (x1_c, y1_c)


def is_in_excluded_zone(center_x, center_y, roi_w, roi_h, exclude_zones):
    if not exclude_zones:
        return False
    cx_ratio = center_x / roi_w if roi_w else 0
    cy_ratio = center_y / roi_h if roi_h else 0
    for zone in exclude_zones:
        if (zone["x_min"] <= cx_ratio <= zone["x_max"] and
                zone["y_min"] <= cy_ratio <= zone["y_max"]):
            return True
    return False


def get_voted_class_from_history(class_history, vehicle_id, default_cls=2, vehicle=None):
    if vehicle is not None:
        locked = vehicle.get("locked_cls")
        if locked is not None:
            return locked
        weights = vehicle.get("cls_weights")
        if weights:
            return max(weights, key=weights.get)

    if class_history.get(vehicle_id):
        counter = Counter(class_history[vehicle_id])
        return counter.most_common(1)[0][0]
    return default_cls


_COCO_VEHICLE_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_FK_GATE_LOGGED = set()
_FK_CONFIRMED_LOGGED = set()
_FK_T0 = None
FORKLIFT_TRUCK_ASPECT_MIN = float(os.environ.get("FORKLIFT_TRUCK_ASPECT_MIN", "1.65"))
# [FIX-GRANMAX] Granmax/pickup/van dari belakang berbentuk hampir KOTAK (w/h 1.1-1.2) sehingga lolos aturan aspek di atas, lalu
# forklift_yolo.pt memberi conf 0.87-0.88 dengan cover 0.97 -> dikunci sebagai forklift dan dibuang (log 07 & 08 Okt: TRK_0132
# 541x487 dan TRK_0001 952x789). Forklift asli di cam1 kecil (bbox log: 53x90, 67x67, 75x64 px = 0.2-0.6% area ROI), sedangkan
# Granmax 11-31% area ROI. Bbox sebesar >= rasio ini TIDAK dicek sebagai forklift (gerbang dilewati). 0 = matikan aturan.
FORKLIFT_GATE_MAX_AREA_RATIO = float(os.environ.get("FORKLIFT_GATE_MAX_AREA_RATIO", "0.06"))
_FK_BIG_LOGGED = set()
# [FIX-OVERSIZE] Kotak yang LEBIH BESAR dari kendaraan nyata mana pun di cam1 hampir pasti kotak GABUNGAN (forklift di depan
# + pickup/van di belakangnya terbaca satu "truk"). Gerbang forklift sengaja dilewati untuk kotak >= 6% (agar Granmax tidak
# terbuang), dan warna oranye diencerkan oleh area kotak yang luas, sehingga kotak gabungan lolos jadi "Truk" (dashboard
# #4166: 1302x1031 px = 55% ROI, conf 0.64, oranye hanya ~1.7%). Data log: truk 6-8% ROI, Granmax maksimum 31%. Kotak
# >= rasio ini: track BARU diabaikan; track yang sudah punya sesi hanya frame-nya yang dilewati (sesi tetap hidup).
# Isi 0 untuk mematikan.
VEHICLE_MAX_AREA_RATIO = float(os.environ.get("VEHICLE_MAX_AREA_RATIO", "0.45"))
_FK_OVERSIZE_LOGGED = set()

# [FIX-ORANGE] Forklift di lokasi ini berwarna ORANYE (HELI). forklift_yolo.pt tidak punya kelas truk sebagai
# pembanding: truk asli ikut terbaca forklift, sedangkan forklift kadang lolos jadi "kendaraan" lalu tergabung
# ke sesi truk lewat RE-LINK (ID pindah ke forklift). Warna dipakai sebagai penjaga tambahan.
#   FORKLIFT_ORANGE_NEW_PCT : track BARU (belum punya sesi) dengan oranye >= nilai ini dianggap forklift,
#                             diblok seumur track (tidak dibuat sesi baru & tidak boleh RE-LINK ke truk).
#   FORKLIFT_ORANGE_LIVE_PCT: track yang SUDAH punya sesi, frame dengan oranye >= nilai ini dilewati
#                             (crop didominasi forklift, bukan truk). Set 0 untuk mematikan.
# [FIX-ORANGE-NEW] default 2.5 -> 1.5: crop forklift di kotak lebar hanya ~1.7% oranye (diencerkan area kotak), sedangkan crop
# truk tanpa forklift 0% (ukuran di activity_estimator: ACT_ORANGE_CROP_PCT=1.0). Kalau ada truk berwarna oranye yang jadi
# diblok, naikkan lagi lewat .env.
FORKLIFT_ORANGE_NEW_PCT = float(os.environ.get("FORKLIFT_ORANGE_NEW_PCT", "1.5"))
FORKLIFT_ORANGE_LIVE_PCT = float(os.environ.get("FORKLIFT_ORANGE_LIVE_PCT", "5.0"))
_ORANGE_BLOCKED = set()
# [FIX-ORANGE-KEEPALIVE] Track milik sesi yang tiba-tiba "berubah jadi forklift" (tracker tertukar) dilewati,
# tapi vannya biasanya masih ada (tertutup forklift). Tanpa ini sesi van dianggap hilang lalu ditutup dan van
# yang muncul lagi menjadi ID baru. Selama batas ini, track yang dilewati tetap dianggap "masih terlihat".
# [FIX-KEEPALIVE-LOADING] Dulu batas 90 detik dihitung dari frame-skip PERTAMA seumur track dan tidak pernah
# di-reset, padahal loading bisa berlangsung berkali-kali selama beberapa menit dengan forklift menempel di truk.
# Akibatnya sesi truk dianggap hilang lalu ditutup (KELUAR) padahal truk masih di tempat dan belum selesai dimuat.
# Sekarang: batas dihitung PER EPISODE (episode baru bila ada jeda > FORKLIFT_KEEPALIVE_EPISODE_GAP detik tanpa
# frame yang dilewati) dan default dinaikkan jadi 900 detik. Isi 0 untuk mematikan.
FORKLIFT_ORANGE_KEEPALIVE_SECONDS = float(os.environ.get("FORKLIFT_ORANGE_KEEPALIVE_SECONDS", "900"))
FORKLIFT_KEEPALIVE_EPISODE_GAP = float(os.environ.get("FORKLIFT_KEEPALIVE_EPISODE_GAP", "5"))
_ORANGE_KEEPALIVE_T0 = {}
_ORANGE_KEEPALIVE_LAST = {}


def _note_gate_skip(tracks, track_id, reason):
    """Hitung frame milik sesi yang dibuang gerbang forklift -> muncul di [ACTIVITY] agar kelihatan
    kalau sampel kendaraan hilang justru saat loading (forklift menempel)."""
    try:
        _vid = tracks.track_to_vehicle.get(track_id)
        _veh = tracks.vehicles.get(_vid) if _vid else None
        if _veh is not None and activity_estimator is not None:
            activity_estimator.note_skip(_veh, reason)
    except Exception:
        pass


def _keepalive_session_track(tracks, track_id, now):
    """Track milik sesi yang frame-nya dilewati karena forklift (warna/YOLO): tetap dianggap 'masih terlihat'
    selama FORKLIFT_ORANGE_KEEPALIVE_SECONDS (per episode) supaya sesi tidak ditutup lalu pecah jadi RE-LINK/ID baru."""
    if FORKLIFT_ORANGE_KEEPALIVE_SECONDS <= 0:
        return
    _otr = tracks.tracks.get(track_id)
    if _otr is None:
        return
    _last = _ORANGE_KEEPALIVE_LAST.get(track_id)
    if _last is None or now - _last > FORKLIFT_KEEPALIVE_EPISODE_GAP:
        _ORANGE_KEEPALIVE_T0[track_id] = now
    _ORANGE_KEEPALIVE_LAST[track_id] = now
    _ot0 = _ORANGE_KEEPALIVE_T0.setdefault(track_id, now)
    if now - _ot0 <= FORKLIFT_ORANGE_KEEPALIVE_SECONDS:
        _otr["last_seen"] = now

# [FIX-DOOR] Pintu van dibuka/ditutup membuat kotak deteksi berubah/pecah (bodi + pintu/bak terbaca terpisah),
# lalu track baru dianggap "kendaraan lain" karena muncul barengan track lama. Aturan ini: track baru yang
# kotaknya hampir seluruhnya berada DI DALAM kotak sesi terbuka (atau sebaliknya) = bagian dari kendaraan yang sama.
# Nilai = porsi irisan terhadap kotak yang lebih kecil (0-1). Set 0 untuk mematikan.
DOOR_MERGE_MIN_CONTAIN = float(os.environ.get("DOOR_MERGE_MIN_CONTAIN", "0.60"))
# Ambang keterkungkungan utk menolak RETURN VISIT palsu (lebih ketat dari DOOR_MERGE). 0 = matikan.
RETURN_FRAGMENT_MIN_CONTAIN = float(os.environ.get("RETURN_FRAGMENT_MIN_CONTAIN", "0.85"))
_ORANGE_LOGGED = set()


def orange_pct(crop):
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    try:
        small = cv2.resize(crop, (160, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (3, 140, 120), (20, 255, 255))
        return float(m.mean()) / 255.0 * 100.0
    except Exception:
        return 0.0

# Simpan hasil crop yang diperiksa gate forklift supaya bisa dilihat/dikalibrasi.
# Subfolder: FORKLIFT (dikunci forklift), TAHAN (ditahan saat warmup), LOLOS (diteruskan sebagai kendaraan).
# Nonaktifkan dengan FORKLIFT_DEBUG_CROPS_DIR= (kosong).
_DBG_ARTIFACTS = os.environ.get("DEBUG_ARTIFACTS", "0").strip().lower() in ("1", "true", "yes", "on")
FORKLIFT_DEBUG_CROPS_DIR = os.environ.get("FORKLIFT_DEBUG_CROPS_DIR", "forklift_debug" if _DBG_ARTIFACTS else "")
FORKLIFT_DEBUG_CROPS_PER_TRACK = int(os.environ.get("FORKLIFT_DEBUG_CROPS_PER_TRACK", "6"))
FORKLIFT_DEBUG_CROPS_MIN_GAP = float(os.environ.get("FORKLIFT_DEBUG_CROPS_MIN_GAP", "2.0"))
_FK_DEBUG_STATE = {}


def _save_fk_debug_crop(crop, track_id, cls, conf, now, label, src, score):
    if not FORKLIFT_DEBUG_CROPS_DIR or crop is None or crop.size == 0:
        return
    try:
        st = _FK_DEBUG_STATE.setdefault(track_id, {"n": 0, "t": -1e9, "label": None})
        changed = st["label"] != label
        if st["n"] >= FORKLIFT_DEBUG_CROPS_PER_TRACK + 2:
            return
        if not changed and (st["n"] >= FORKLIFT_DEBUG_CROPS_PER_TRACK
                            or now - st["t"] < FORKLIFT_DEBUG_CROPS_MIN_GAP):
            return
        st["n"] += 1
        st["t"] = now
        st["label"] = label
        t_rel = now - (_FK_T0 if _FK_T0 is not None else now)
        cname = _COCO_VEHICLE_NAMES.get(cls, f"cls{cls}")
        folder = os.path.join(FORKLIFT_DEBUG_CROPS_DIR, label)
        os.makedirs(folder, exist_ok=True)
        img = crop.copy()
        banner = f"{label} {track_id} t+{t_rel:.0f}s {cname} {conf:.2f} | {src or '-'} {score:.2f}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, banner, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        fn = f"{track_id}_{st['n']:02d}_t{t_rel:04.0f}s_{cname}_{conf:.2f}.jpg"
        cv2.imwrite(os.path.join(folder, fn), img)
    except Exception as e:
        print(f"[GATE] gagal simpan crop debug: {e}")


# ---- Video debug: tulis video beranotasi (kotak + label keputusan per track) ----
# Aktif secara default -> forklift_debug/debug_<cam>.mp4 . Matikan dengan DEBUG_VIDEO_PATH= (kosong).
# Warna: MERAH=FORKLIFT (dikunci), KUNING=TAHAN (warmup), HIJAU=KENDARAAN (lolos gate), ABU=SKIP/belum diproses.
DEBUG_VIDEO_PATH = os.environ.get(
    "DEBUG_VIDEO_PATH", os.path.join("forklift_debug", "debug_{cam}.mp4") if _DBG_ARTIFACTS else "")
DEBUG_VIDEO_FPS = float(os.environ.get("DEBUG_VIDEO_FPS", "25"))
DEBUG_VIDEO_MAX_WIDTH = int(os.environ.get("DEBUG_VIDEO_MAX_WIDTH", "1280"))
_DBG_VIDEO = {}
# Ukur waktu per bagian supaya kelihatan apa yang bikin lambat. Matikan: PERF_LOG_EVERY=0
PERF_LOG_EVERY = int(os.environ.get("PERF_LOG_EVERY", "100"))
_PERF = {"n": 0, "handle": 0.0, "gate": 0.0, "video": 0.0, "act": 0.0}
_DBG_COLORS = {
    "FORKLIFT": (0, 0, 255),
    "TAHAN": (0, 215, 255),
    "KENDARAAN": (0, 200, 0),
    "SKIP": (160, 160, 160),
}


def _debug_video_write(cam_name, roi_frame, ann, now):
    if not DEBUG_VIDEO_PATH or roi_frame is None:
        return
    try:
        st = _DBG_VIDEO.get(cam_name)
        h0, w0 = roi_frame.shape[:2]
        f = min(1.0, DEBUG_VIDEO_MAX_WIDTH / float(max(1, w0)))
        ow, oh = int(w0 * f) // 2 * 2, int(h0 * f) // 2 * 2
        if st is None:
            path = DEBUG_VIDEO_PATH.format(cam=cam_name)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), DEBUG_VIDEO_FPS, (ow, oh))
            st = {"vw": vw, "size": (ow, oh), "n": 0, "path": path}
            _DBG_VIDEO[cam_name] = st
            print(f"[DEBUG-VIDEO] menulis video beranotasi ke '{path}' ({ow}x{oh} @ {DEBUG_VIDEO_FPS:.0f}fps)")
        img = cv2.resize(roi_frame, st["size"]) if f < 1.0 else roi_frame.copy()
        for a in ann:
            x1, y1, x2, y2 = [int(v * f) for v in a["bbox"]]
            col = _DBG_COLORS.get(a["label"], (160, 160, 160))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            txt = f"{a['label']} {a['text']}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(th + 6, y1)
            cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), col, -1)
            cv2.putText(img, txt, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2, cv2.LINE_AA)
        st["n"] += 1
        t_rel = now - (_FK_T0 if _FK_T0 is not None else now)
        cv2.putText(img, f"{cam_name} frame#{st['n']} t+{t_rel:.0f}s", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        st["vw"].write(img)
    except Exception as e:
        print(f"[DEBUG-VIDEO] gagal menulis frame: {e}")


def _debug_video_close():
    for st in _DBG_VIDEO.values():
        try:
            st["vw"].release()
            print(f"[DEBUG-VIDEO] selesai: '{st['path']}' ({st['n']} frame)")
        except Exception:
            pass
    _DBG_VIDEO.clear()


import atexit as _atexit

_atexit.register(_debug_video_close)


ACTIVITY_LABEL_IN_JENIS = os.environ.get("ACTIVITY_LABEL_IN_JENIS", "1").strip().lower() not in ("0", "false", "no", "off")


def _short_activity(txt):
    """'Mengambil Muatan (Loading) x3, Bongkar Muatan (Unloading)' -> 'Loading x3, Unloading'."""
    if not txt:
        return ""
    return (str(txt).replace("Mengambil Muatan (Loading)", "Loading").replace("Bongkar Muatan (Unloading)", "Unloading")
            .replace("Aktivitas Forklift (arah belum pasti)", "Aktivitas Forklift"))


def _jenis_with_activity(base, activity):
    # [DASHBOARD] Dashboard/n8n yang ada hanya menampilkan jenis_kendaraan di teks deskripsi dan belum membaca
    # status_aktivitas, jadi status ikut ditulis di jenis_kendaraan agar langsung terlihat. Matikan: ACTIVITY_LABEL_IN_JENIS=0
    short = _short_activity(activity)
    if not ACTIVITY_LABEL_IN_JENIS or not short or short in ("Lewat Kosong", "Tidak Dapat Dipastikan"):
        return base
    return f"{base} - {short}"


_CAM_CONFIG = {}
ACTIVITY_EVENTS_SEND = os.environ.get("ACTIVITY_EVENTS_SEND", "1").strip().lower() not in ("0", "false", "no", "off")
# [MERGE-AKTIVITAS] Satu sesi memuat bisa menghasilkan puluhan kunjungan forklift (tiap track forklift baru = satu event),
# padahal truknya diam. Event AKTIVITAS yang AKTIVITASNYA SAMA dengan event sebelumnya dan berjarak < N detik dianggap
# kelanjutan dari proses yang sama dan TIDAK dikirim lagi (tetap tercatat di log & ringkasan "Loading xN" pada SIKLUS_SELESAI).
# Event baru dikirim bila aktivitasnya berganti (mis. Loading -> Unloading) atau jeda sejak event sebelumnya >= N detik.
# 0 = matikan penggabungan (kirim semua).
ACTIVITY_EVENT_MERGE_SECONDS = float(os.environ.get("ACTIVITY_EVENT_MERGE_SECONDS", "600") or 0)
# [FIX-FORKLIFT-HANTU] Event dari "lintasan forklift" (dock_events) hanya bersandar pada deteksi YOLO forklift, TANPA perubahan skor
# muatan. Model forklift_yolo.pt juga menganggap truk/pickup (dan bagian-bagiannya) sebagai forklift, sehingga pada video tanpa
# forklift pun muncul "forklift #N menempel ke kendaraan" -> event Loading palsu saat tidak terjadi apa-apa (log 08 Okt: skor muatan
# 0.07-0.43 sepanjang video, tetap 7 event). Default: event semacam itu DIABAIKAN (tidak dikirim & tidak masuk ringkasan).
# Nyalakan lagi hanya bila forklift memang terlihat jelas oleh kamera: ACTIVITY_ALLOW_FORKLIFT_ONLY=1
ACTIVITY_ALLOW_FORKLIFT_ONLY = os.environ.get("ACTIVITY_ALLOW_FORKLIFT_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")


def _is_forklift_only_event(ev):
    try:
        return str(ev.get("source", "")).startswith("lintasan")
    except Exception:
        return False


def _emit_activity_event(config, tracks, vehicle, ev):
    """Kirim SATU kejadian Loading/Unloading (jenis_kejadian='AKTIVITAS'). Satu sesi bisa punya banyak."""
    try:
        vid = vehicle["vehicle_id"]
        event_id = tracks.make_event_id(vehicle, f"ACT_{ev['no']:03d}")
        final_cls = vehicle.get("locked_cls")
        if final_cls is None:
            final_cls = vehicle.get("final_cls", -1)
        voted_cls = get_voted_class_from_history(tracks.class_history, vid, final_cls, vehicle=vehicle)
        warna, rgb = vehicle.get("locked_color") or _voted_color(vehicle)
        bbox = vehicle.get("last_bbox") or (0.0, 0.0, 0.0, 0.0)
        after = ev.get("score_after")
        payload = {
            "timestamp": datetime.fromtimestamp(ev["t_end"]).strftime("%Y-%m-%d %H:%M:%S"),
            "kamera_id": config["kamera_id"],
            "kamera_nama": config["kamera_nama"],
            "jenis_kendaraan": _jenis_with_activity(_jenis_label(voted_cls, config, vehicle), ev["activity"]),
            "warna": warna,
            "rgb_r": int(rgb[0]), "rgb_g": int(rgb[1]), "rgb_b": int(rgb[2]),
            "confidence": round(float(vehicle.get("final_conf", 0.0)), 4),
            "bbox_x1": int(bbox[0]), "bbox_y1": int(bbox[1]), "bbox_x2": int(bbox[2]), "bbox_y2": int(bbox[3]),
            "gambar_base64": encode_image_base64(
                photo_at_time(vehicle, min(float(ev["t_end"]) + PHOTO_AFTER_EVENT_SECONDS,
                                           ((vehicle.get("photo_hist") or [{"ts": float(ev["t_end"])}])[-1]["ts"])),
                              tag=f"AKTIVITAS #{ev['no']}")
                if vehicle.get("photo_hist") else best_recent_crop(vehicle)),
            "status_muatan": ((CARGO_LOADED if "Loading" in ev["activity"] else CARGO_EMPTY if "Unloading" in ev["activity"] else CARGO_UNKNOWN)
                              if after is None else (CARGO_LOADED if after >= 0.5 else CARGO_EMPTY)),
            "jenis_kejadian": "AKTIVITAS",
            "track_id": vehicle.get("last_track_id", "UNKNOWN"),
            "vehicle_id": vid,
            "event_id": event_id,
            "event_no": ev["no"],
            "arah_kendaraan": "TIDAK_JELAS", "arah_confidence": 0.0,
            "plat_nomor": None, "plat_confidence": 0.0,
            "status_aktivitas": ev["activity"],
            "aktivitas_alasan": ev["reason"],
            "aktivitas_sumber": ev["source"],
            "aktivitas_confidence": ev["confidence"],
            "skor_muatan_awal": ev.get("score_before"),
            "skor_muatan_akhir": ev.get("score_after"),
            "pelaku": ev.get("pelaku"),
            "area_kejadian": ev.get("area"),
            "waktu_mulai": datetime.fromtimestamp(ev["t_start"]).strftime("%Y-%m-%d %H:%M:%S"),
            "waktu_selesai": datetime.fromtimestamp(ev["t_end"]).strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(f"[AKTIVITAS] {config['kamera_id']} {vid} #{ev['no']}: {ev['activity']} | {ev.get('area')} | "
              f"pelaku={ev.get('pelaku')} | sumber={ev['source']} conf={ev['confidence']} | {ev['reason']}")
        if _is_forklift_only_event(ev) and not ACTIVITY_ALLOW_FORKLIFT_ONLY:
            if ACTIVITY_FORKLIFT_ONLY_CONFIRM == "bak" and _bed_change_supports(vehicle, ev):
                payload["aktivitas_sumber"] = "lintasan_forklift+bak_berubah"
                payload["aktivitas_alasan"] = f"{ev['reason']} [DIKONFIRMASI: isi bak berubah pada jendela kejadian]"
                payload["aktivitas_confidence"] = max(float(ev.get("confidence") or 0.0), 0.6)
                print(f"[AKTIVITAS] #{ev['no']} DIKONFIRMASI oleh perubahan isi bak -> dikirim")
            else:
                vehicle["_aktivitas_ditolak"] = vehicle.get("_aktivitas_ditolak", 0) + 1
                print(f"[AKTIVITAS] #{ev['no']} DIABAIKAN: lintasan forklift TANPA bukti perubahan skor muatan maupun isi bak "
                      f"(skor {ev.get('score_before')} -> {ev.get('score_after')}); tidak dikirim. "
                      f"ACTIVITY_ALLOW_FORKLIFT_ONLY=1 untuk mengirim semua.")
                return
        if ACTIVITY_EVENTS_SEND and tracks.can_send_event(event_id):
            _prev = vehicle.get("_aktivitas_prev")  # (aktivitas, t_end) event TERAKHIR (terkirim atau digabung)
            vehicle["_aktivitas_prev"] = (ev["activity"], ev["t_end"])
            if (ACTIVITY_EVENT_MERGE_SECONDS > 0 and _prev is not None and _prev[0] == ev["activity"]
                    and (ev["t_end"] - _prev[1]) < ACTIVITY_EVENT_MERGE_SECONDS):
                vehicle["_aktivitas_digabung"] = vehicle.get("_aktivitas_digabung", 0) + 1
                print(f"[AKTIVITAS] #{ev['no']} digabung ke event sebelumnya (aktivitas sama, jeda "
                      f"{ev['t_end'] - _prev[1]:.0f}s < {ACTIVITY_EVENT_MERGE_SECONDS:.0f}s) -> tidak dikirim ke n8n")
            else:
                kirim_ke_n8n(payload)
    except Exception as e:
        print(f"[AKTIVITAS] gagal membuat event: {e}")
        traceback.print_exc()


def handle_detections(config, result, scale, roi_frame, tracks, now, cam_name):
    global _FK_T0
    ann = []
    _t0 = time.perf_counter()
    try:
        _handle_detections_impl(config, result, scale, roi_frame, tracks, now, cam_name, ann)
        if config.get("dock_events") and dock_events is not None and ACTIVITY_MODE:
            try:
                _CAM_CONFIG[cam_name] = config
                dock_events.step(cam_name, roi_frame, tracks, now,
                                 lambda v, ev: _emit_activity_event(config, tracks, v, ev))
                ann.extend(dock_events.annotations(cam_name, now))
            except Exception as _e:
                print(f"[DOCK] error (diabaikan, deteksi utama tetap jalan): {_e}")
                traceback.print_exc()
    finally:
        _t1 = time.perf_counter()
        if _FK_T0 is None:
            _FK_T0 = now
        _debug_video_write(cam_name, roi_frame, ann, now)
        _t2 = time.perf_counter()
        _PERF["n"] += 1
        _PERF["handle"] += _t1 - _t0
        _PERF["video"] += _t2 - _t1
        if PERF_LOG_EVERY > 0 and _PERF["n"] % PERF_LOG_EVERY == 0:
            n = float(PERF_LOG_EVERY)
            print(f"[PERF] {_PERF['n']} frame | rata-rata per frame: handle_detections={_PERF['handle'] / n * 1000:.0f}ms "
                  f"(di dalamnya gate-forklift-YOLO={_PERF['gate'] / n * 1000:.0f}ms, "
                  f"sampling-muatan={_PERF['act'] / n * 1000:.0f}ms) "
                  f"tulis-video-debug={_PERF['video'] / n * 1000:.0f}ms")
            _PERF["handle"] = _PERF["gate"] = _PERF["video"] = _PERF["act"] = 0.0


def _handle_detections_impl(config, result, scale, roi_frame, tracks, now, cam_name, _ann):
    if not (result and result.boxes is not None and result.boxes.id is not None):
        return

    roi_h, roi_w = roi_frame.shape[:2]
    boxes = result.boxes
    ids = boxes.id.int().cpu().tolist()

    _this_frame_boxes = {}
    for _i in range(len(boxes)):
        _tid = f"TRK_{ids[_i]:04d}"
        _x1r, _y1r, _x2r, _y2r = boxes.xyxy[_i].cpu().tolist()
        _this_frame_boxes[_tid] = (
            _x1r * scale, _y1r * scale, _x2r * scale, _y2r * scale
        )
    _this_frame_track_ids = set(_this_frame_boxes.keys())

    def _neighbor_boxes_for(track_id, vehicle):
        own_tracks = set(vehicle.get("active_tracks") or [])
        own_tracks.add(track_id)
        result_boxes = [
            b for tid, b in _this_frame_boxes.items() if tid not in own_tracks
        ]
        for tid, tinfo in tracks.tracks.items():
            if tid in own_tracks or tid in _this_frame_boxes:
                continue
            if now - tinfo.get("last_seen", 0) <= 2.0 and tinfo.get("last_bbox"):
                result_boxes.append(tinfo["last_bbox"])
        return result_boxes

    global _FK_T0

    for i in range(len(boxes)):
        cls = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        yolo_track_id = ids[i]
        track_id = f"TRK_{yolo_track_id:04d}"

        counter_key = f"{cam_name}_{track_id}"

        _FRAME_COUNTERS[counter_key] = _FRAME_COUNTERS.get(counter_key, 0) + 1
        if _FRAME_COUNTERS[counter_key] < 6:
            continue

        x1r, y1r, x2r, y2r = boxes.xyxy[i].cpu().tolist()
        x1r, y1r, x2r, y2r = (
            x1r * scale, y1r * scale, x2r * scale, y2r * scale
        )
        bbox = (x1r, y1r, x2r, y2r)
        _a = {"bbox": bbox, "label": "SKIP",
              "text": f"{track_id} {_COCO_VEHICLE_NAMES.get(cls, cls)} {conf:.2f}"}
        _ann.append(_a)

        width, height = x2r - x1r, y2r - y1r
        area = width * height

        roi_area = roi_w * roi_h
        min_ratio = config.get("min_bbox_area_ratio", 0)
        if min_ratio > 0 and roi_area > 0 and (area / roi_area) < min_ratio:
            if DEBUG_MODE:
                print(f"[{cam_name}] SKIP (terlalu kecil): track={track_id} "
                      f"area_ratio={area / roi_area:.4f} < min={min_ratio:.4f}")
            continue

        if MIN_AREA_NOISE > 0 and area < MIN_AREA_NOISE:
            continue

        # [FIX-OVERSIZE] kotak gabungan forklift+kendaraan (lihat VEHICLE_MAX_AREA_RATIO)
        if (config.get("forklift_gate") and VEHICLE_MAX_AREA_RATIO > 0 and roi_area > 0
                and (area / roi_area) >= VEHICLE_MAX_AREA_RATIO):
            if tracks.track_to_vehicle.get(track_id):
                _keepalive_session_track(tracks, track_id, now)
                _note_gate_skip(tracks, track_id, "oversize")
            _a["text"] += " oversize"
            if track_id not in _FK_OVERSIZE_LOGGED:
                _FK_OVERSIZE_LOGGED.add(track_id)
                print(f"[GATE] {cam_name} {track_id} bbox={int(width)}x{int(height)} = {100.0 * area / roi_area:.1f}% area ROI "
                      f"(>= {100.0 * VEHICLE_MAX_AREA_RATIO:.0f}%) -> kotak gabungan/terlalu besar untuk satu kendaraan, DIABAIKAN "
                      f"(sesi={'ya' if tracks.track_to_vehicle.get(track_id) else 'belum'})")
            continue

        _truck_min_conf = config.get("truck_min_confidence", 0.0)
        if (
            _truck_min_conf
            and cls == config.get("truck_class_id", 7)
            and conf < _truck_min_conf
        ):
            if DEBUG_MODE:
                print(f"[{cam_name}] SKIP (truck conf rendah, kemungkinan "
                      f"forklift/bukan truk): track={track_id} conf={conf:.2f} "
                      f"< {_truck_min_conf:.2f}")
            continue

        center_x = (x1r + x2r) / 2
        center_y = (y1r + y2r) / 2

        if is_in_excluded_zone(center_x, center_y, roi_w, roi_h,
                               config.get("exclude_zones", [])):
            continue

        crop, crop_origin = make_crop(roi_frame, roi_w, roi_h, bbox)

        if crop is not None and _is_frame_corrupted(crop):
            if DEBUG_MODE:
                print(f"[{cam_name}] SKIP crop (corrupt lokal): track={track_id}, "
                      f"frame sekitarnya lolos filter global tapi area kendaraan rusak")
            crop = None

        if crop is not None and (NOISE_CORRUPT_CHECK_ENABLED or _NOISE_DEBUG_LOG_ALWAYS):
            _noisy = _is_frame_noisy_corrupt(
                crop, debug_label=f"{cam_name}/crop/{track_id}" if _NOISE_DEBUG_LOG_ALWAYS else None
            )
            if NOISE_CORRUPT_CHECK_ENABLED and _noisy:
                if DEBUG_MODE:
                    print(f"[{cam_name}] SKIP crop (noise/dither lokal): track={track_id}, "
                          f"varians piksel lokal tidak wajar -- kemungkinan corrupt "
                          f"jenis dither/speckle (lolos cek jump biasa)")
                crop = None

        crop_enhanced = crop  # [NO-FILTER] tanpa peningkatan tampilan; nama variabel dipertahankan agar kode lain tidak berubah

        # [FIX-ORANGE] penjaga warna forklift (lihat definisi FORKLIFT_ORANGE_NEW_PCT)
        if crop_enhanced is not None and (FORKLIFT_ORANGE_NEW_PCT > 0 or FORKLIFT_ORANGE_LIVE_PCT > 0):
            _has_session = bool(tracks.track_to_vehicle.get(track_id))
            _orp = None
            if track_id in _ORANGE_BLOCKED:
                _skip_orange = True
            else:
                _orp = orange_pct(crop_enhanced)
                if not _has_session and FORKLIFT_ORANGE_NEW_PCT > 0 and _orp >= FORKLIFT_ORANGE_NEW_PCT:
                    _ORANGE_BLOCKED.add(track_id)
                    _skip_orange = True
                elif _has_session and FORKLIFT_ORANGE_LIVE_PCT > 0 and _orp >= FORKLIFT_ORANGE_LIVE_PCT:
                    _skip_orange = True
                else:
                    _skip_orange = False
            if _skip_orange:
                if _has_session:
                    _keepalive_session_track(tracks, track_id, now)
                    _note_gate_skip(tracks, track_id, "oranye")
                _a["label"] = "FORKLIFT"
                if _orp is not None:
                    _a["text"] += f" oranye={_orp:.1f}%"
                # [FIX-FK-SELF-NOTE] Track yang sudah punya sesi = kendaraan itu sendiri (ada forklift di crop),
                # bbox-nya BUKAN bbox forklift. Mencatatnya membuat kendaraan "dekat forklift" dengan dirinya sendiri.
                if activity_estimator is not None and not _has_session:
                    activity_estimator.note_forklift(bbox, now)
                if track_id not in _ORANGE_LOGGED:
                    _ORANGE_LOGGED.add(track_id)
                    print(f"[ORANGE] {cam_name} {track_id} t+{now - (_FK_T0 or now):.0f}s => dianggap FORKLIFT "
                          f"(oranye={(_orp if _orp is not None else 0.0):.1f}%, sesi={'ya' if _has_session else 'belum'}); "
                          f"tidak dibuat/digabung ke sesi truk")
                continue

        # Kotak LEBAR (w/h >= FORKLIFT_TRUCK_ASPECT_MIN) = bentuk truk, bukan forklift.
        # Dari log: truk memberi conf forklift 0.82-0.93 (sama dengan forklift asli), jadi conf saja
        # tidak bisa memisahkan; bentuk kotak bisa. Gate dilewati total (hemat CPU juga).
        # Matikan aturan ini dengan FORKLIFT_TRUCK_ASPECT_MIN=0.
        _wide_truck_shape = (
            FORKLIFT_TRUCK_ASPECT_MIN > 0
            and (width / max(1.0, height)) >= FORKLIFT_TRUCK_ASPECT_MIN
        )
        _big_vehicle_shape = (
            FORKLIFT_GATE_MAX_AREA_RATIO > 0 and roi_area > 0
            and (area / roi_area) >= FORKLIFT_GATE_MAX_AREA_RATIO
        )
        if _big_vehicle_shape and config.get("forklift_gate") and track_id not in _FK_BIG_LOGGED:
            _FK_BIG_LOGGED.add(track_id)
            print(f"[GATE] {cam_name} {track_id} bbox={int(width)}x{int(height)} = {100.0 * area / roi_area:.1f}% area ROI "
                  f"(>= {100.0 * FORKLIFT_GATE_MAX_AREA_RATIO:.0f}%) -> terlalu besar untuk forklift, gerbang forklift dilewati")
        if config.get("forklift_gate") and crop_enhanced is not None and not _wide_truck_shape and not _big_vehicle_shape:
            _is_forklift, _forklift_sim, _forklift_src = False, 0.0, ""

            # Log sekali per track: kelas dari detektor UTAMA (yolo11n) + ukuran + waktu relatif,
            # supaya bisa dicocokkan dengan keputusan forklift_filter di bawah.
            if track_id not in _FK_GATE_LOGGED:
                _FK_GATE_LOGGED.add(track_id)
                if _FK_T0 is None:
                    _FK_T0 = now
                _cname = _COCO_VEHICLE_NAMES.get(cls, f"cls{cls}")
                print(f"[GATE] {cam_name} {track_id} t+{now - _FK_T0:.0f}s "
                      f"kelas_utama={_cname}({cls}) conf_utama={conf:.2f} "
                      f"bbox={int(width)}x{int(height)}")

            if forklift_filter is not None and forklift_filter.yolo_available():
                _tg = time.perf_counter()
                _is_forklift, _forklift_sim = forklift_filter.is_forklift_by_detector(
                    track_id, crop_enhanced)
                _PERF["gate"] += time.perf_counter() - _tg
                _forklift_src = "detektor YOLO"

            if not _is_forklift and cargo_classifier is not None and forklift_filter is not None:
                _crop_fk = crop_enhanced if CARGO_DISABLE_BED_CROP else _crop_to_cargo_bed(crop_enhanced)
                _is_forklift, _forklift_sim = forklift_filter.is_forklift_by_model(
                    track_id, _crop_fk, cargo_classifier.forklift_probability)
                _forklift_src = "model"

            if not _is_forklift and forklift_filter is not None and forklift_filter.is_available():
                _ff_fn = getattr(forklift_filter, "is_forklift_tracked", None)
                if _ff_fn is not None:
                    _is_forklift, _forklift_sim = _ff_fn(track_id, crop_enhanced)
                else:
                    _is_forklift, _forklift_sim = forklift_filter.is_forklift(crop_enhanced)
                _forklift_src = "tampilan"

            _fk_confirmed = _is_forklift and not (
                _forklift_src == "detektor YOLO"
                and not forklift_filter.yolo_is_locked(track_id)
            )
            _save_fk_debug_crop(
                crop_enhanced, track_id, cls, conf, now,
                "FORKLIFT" if _fk_confirmed else ("TAHAN" if _is_forklift else "LOLOS"),
                _forklift_src, _forklift_sim)
            _a["label"] = "FORKLIFT" if _fk_confirmed else ("TAHAN" if _is_forklift else "KENDARAAN")
            if _forklift_src:
                _a["text"] += f" fk={_forklift_sim:.2f}"

            if _is_forklift:
                # Saat warmup detektor YOLO, track hanya ditahan (belum pasti forklift),
                # jadi jangan dicatat sebagai forklift dulu.
                _fk_has_session = bool(tracks.track_to_vehicle.get(track_id))
                if _fk_confirmed and _fk_has_session:
                    _keepalive_session_track(tracks, track_id, now)
                    _note_gate_skip(tracks, track_id, "yolo")
                if activity_estimator is not None and _fk_confirmed and not _fk_has_session:
                    activity_estimator.note_forklift(bbox, now)
                if _fk_confirmed and track_id not in _FK_CONFIRMED_LOGGED:
                    _FK_CONFIRMED_LOGGED.add(track_id)
                    print(f"[GATE] {cam_name} {track_id} t+{now - (_FK_T0 or now):.0f}s "
                          f"=> FORKLIFT oleh {_forklift_src} (skor={_forklift_sim:.2f}); "
                          f"kelas_utama={_COCO_VEHICLE_NAMES.get(cls, cls)} conf_utama={conf:.2f}")
                if DEBUG_MODE:
                    print(f"[{cam_name}] SKIP (forklift dikenali dari {_forklift_src}): "
                          f"track={track_id} skor={_forklift_sim:.2f}")
                continue

        if SHADOW_FILTER_ENABLE and conf < SHADOW_FILTER_CONF_GATE:
            already_tracked = tracks.track_to_vehicle.get(track_id)
            if not already_tracked and looks_like_shadow(crop_enhanced):
                if DEBUG_MODE:
                    print(f"[{cam_name}] SKIP (bayangan): track={track_id} conf={conf:.2f}")
                continue

        if _a["label"] == "SKIP":
            _a["label"] = "KENDARAAN"

        vehicle, is_new_track, was_relinked, was_return = tracks.get_or_create_vehicle(
            track_id, cls, bbox, now, crop=crop_enhanced,
            this_frame_track_ids=_this_frame_track_ids
        )

        if vehicle is None:
            continue

        tracks.update_vehicle(vehicle, track_id, cls, conf, bbox, now, crop_enhanced,
                              frame_area=(roi_w * roi_h),
                              has_plate_snapshot=bool(config.get("plate_snapshot_channel")),
                              roi_h=roi_h,
                              crop_origin=crop_origin,
                              neighbor_boxes=_neighbor_boxes_for(track_id, vehicle))

        _periodic_voted_cls = get_voted_class_from_history(
            tracks.class_history, vehicle["vehicle_id"], cls, vehicle=vehicle
        )
        if (
            config.get("cargo_detection")
            and _is_cargo_class(_periodic_voted_cls, config)
            and not vehicle.get("session_closed")
        ):
            last_periodic_check = vehicle.get("last_cargo_periodic_check", 0.0)
            if now - last_periodic_check >= CARGO_PERIODIC_CHECK_INTERVAL_SECONDS:
                vehicle["last_cargo_periodic_check"] = now
                if ACTIVITY_MODE:
                    activity_estimator.set_t0(_FK_T0 if _FK_T0 is not None else now)
                    _ta = time.perf_counter()
                    activity_estimator.add_samples(
                        vehicle, now, _soft_loaded_score, _activity_box_fn
                    )
                    _PERF["act"] = _PERF.get("act", 0.0) + (time.perf_counter() - _ta)
                    _maybe_mark_pickup(vehicle, _periodic_voted_cls, config)
                    continue_legacy_cargo = False
                else:
                    continue_legacy_cargo = True
                    best_cargo_detection(vehicle, debug_tag="PERIODIC")

                if continue_legacy_cargo and not vehicle.get("entered") and vehicle.get("cargo_in_latched") is None:
                    _mv = vehicle.get("last_moment_cargo")
                    if (
                        _mv
                        and _mv["status"] != CARGO_UNKNOWN
                        and _mv["agreement"] >= CARGO_MOMENT_MIN_AGREEMENT
                        and _mv["votes"] >= CARGO_MOMENT_MIN_VOTES
                        and _mv["confidence"] >= CARGO_MIN_AVG_CONFIDENCE
                    ):
                        vehicle["cargo_in_latched"] = (_mv["status"], _mv["confidence"], now)
                        print(
                            f"[CARGO-LATCH] {vehicle['vehicle_id']}: kondisi saat datang "
                            f"dikunci = {_mv['status']} (conf={_mv['confidence']:.2f}, "
                            f"sepakat={_mv['agreement']:.2f}, "
                            f"{now - vehicle['first_seen']:.1f}s sejak pertama terlihat)"
                        )

        snap_channel = config.get("plate_snapshot_channel")
        want_plate_snapshot = (
            config.get("plate_detection", False)
            and config.get("read_own_plate", True)
            and not vehicle.get("plate_confirmed")
        )
        want_cargo_snapshot = (
            config.get("cargo_detection", False)
            and not vehicle.get("session_closed")
        )
        if snap_channel and (want_plate_snapshot or want_cargo_snapshot):
            area_ratio = (area / roi_area) if roi_area else 0.0
            if PLATE_SNAPSHOT_MIN_AREA_RATIO > 0 and area_ratio < PLATE_SNAPSHOT_MIN_AREA_RATIO:
                if DEBUG_MODE:
                    _alasan = "plat/cargo" if (want_plate_snapshot and want_cargo_snapshot) else (
                        "plat" if want_plate_snapshot else "cargo")
                    print(f"[{cam_name}] TUNDA snapshot ({_alasan}): track={track_id} "
                          f"area_ratio={area_ratio:.4f} < min={PLATE_SNAPSHOT_MIN_AREA_RATIO:.4f} "
                          f"(kendaraan masih terlalu jauh/kecil)")
            else:
                snap_count = vehicle.get("plate_snapshot_count", 0)
                last_snap_attempt = vehicle.get("last_plate_snapshot_attempt", 0.0)
                interval = (PLATE_SNAPSHOT_INTERVAL_SECONDS
                            if snap_count < PLATE_SNAPSHOT_MAX_PER_VEHICLE
                            else PLATE_SNAPSHOT_EXTENDED_INTERVAL_SECONDS)
                if (snap_count < PLATE_SNAPSHOT_MAX_TOTAL_ATTEMPTS
                        and now - last_snap_attempt >= interval):
                    vehicle["last_plate_snapshot_attempt"] = now
                    vehicle["plate_snapshot_count"] = snap_count + 1
                    threading.Thread(
                        target=_plate_snapshot_worker,
                        args=(tracks, vehicle, snap_channel, config["roi"], bbox,
                              roi_w, roi_h, conf, now, cam_name,
                              f"#{snap_count + 1} ({vehicle['vehicle_id']})"),
                        kwargs={"cargo_detection_enabled": config.get("cargo_detection", False)},
                        daemon=True,
                    ).start()

        if is_new_track and DEBUG_MODE:
            cx_ratio = center_x / roi_w if roi_w else 0
            cy_ratio = center_y / roi_h if roi_h else 0
            w_ratio = (x2r - x1r) / roi_w if roi_w else 0
            h_ratio = (y2r - y1r) / roi_h if roi_h else 0
            area_ratio = (area / roi_area) if roi_area else 0
            label = "car" if cls == 2 else ("truck" if cls == config.get("truck_class_id", 7) else f"cls{cls}")
            print(f"[{cam_name}] TRACK BARU {track_id} -> "
                  f"{vehicle['vehicle_id']} ({label}, conf={conf:.2f}) "
                  f"pos=(x={cx_ratio:.3f}, y={cy_ratio:.3f}) "
                  f"size=(w={w_ratio:.3f}, h={h_ratio:.3f}, area_ratio={area_ratio:.4f})")

        if was_return and not vehicle.get("entry_event_sent"):
            voted_cls = get_voted_class_from_history(
                tracks.class_history, vehicle["vehicle_id"], cls, vehicle=vehicle
            )

            if ACTIVITY_MODE:
                status_muatan_return = activity_estimator.arrival_label(vehicle) or ACTIVITY_PENDING_LABEL
            elif (
                config["cargo_detection"]
                and crop_enhanced is not None
                and _is_truck_class(voted_cls, config)
            ):
                status_muatan_return, _ = detect_cargo_status(crop_enhanced)
            else:
                status_muatan_return = CARGO_UNKNOWN

            event_id = tracks.make_event_id(vehicle, f"MASUK_{vehicle.get('visit_no', 1):04d}")

            if config.get("sync_plate_to"):
                vehicle["entry_event_sent"] = True
            elif tracks.can_send_event(event_id):
                arah, arah_score = classify_vehicle_direction(vehicle, config)
                plat_text, plat_conf = None, 0.0
                payload = build_payload(
                    config=config,
                    jenis_kejadian="MASUK",
                    cls=voted_cls,
                    conf=conf,
                    bbox=bbox,
                    status_muatan=status_muatan_return,
                    crop_enhanced=crop_enhanced,
                    track_id=track_id,
                    vehicle_id=vehicle["vehicle_id"],
                    event_id=event_id,
                    arah_kendaraan=arah,
                    arah_confidence=arah_score,
                    plat_nomor=plat_text,
                    plat_confidence=plat_conf,
                    warna=(vehicle.get("locked_color") or _voted_color(vehicle))[0],
                    warna_rgb=(vehicle.get("locked_color") or _voted_color(vehicle))[1],
                    event_ts=vehicle.get("first_seen") or now,
                )
                kirim_ke_n8n(payload)
                vehicle["entry_event_sent"] = True
                print(f"[{cam_name}] MASUK KEMBALI: {vehicle['vehicle_id']} | "
                      f"visit={vehicle.get('visit_no')} | "
                      f"arah={arah} (score={arah_score:.2f})")

        if not vehicle.get("entered"):
            time_since_first_seen = now - vehicle["first_seen"]

            has_usable_crop = _has_usable_confirm_crop(vehicle, config)

            has_usable_crop = has_usable_crop and bool(vehicle.get("cls_votes"))

            if has_usable_crop and not _cls_vote_is_decisive(vehicle, config):
                has_usable_crop = False
                if DEBUG_MODE and not vehicle.get("_cls_indecisive_logged"):
                    vehicle["_cls_indecisive_logged"] = True
                    _w = vehicle.get("cls_weights") or {}
                    _tot = sum(_w.values()) or 1.0
                    print(f"[{cam_name}] TUNDA MASUK: {vehicle['vehicle_id']} jenis "
                          f"kendaraan belum jelas -- bobot vote="
                          f"{ {k: round(v / _tot, 2) for k, v in _w.items()} }, "
                          f"butuh salah satu >= {CLS_VOTE_MIN_DOMINANCE:.2f}")

            _stationary_since = vehicle.get("stationary_since")
            _stationary_big_enough = (
                vehicle.get("best_area_ratio", 0.0) >= MIN_STATIONARY_CONFIRM_AREA_RATIO
            )
            _parked_long_enough = (
                ENTRY_STATIONARY_CONFIRM_SECONDS > 0
                and _stationary_since is not None
                and (now - _stationary_since) >= ENTRY_STATIONARY_CONFIRM_SECONDS
                and _stationary_big_enough
            )
            if (
                DEBUG_MODE
                and ENTRY_STATIONARY_CONFIRM_SECONDS > 0
                and _stationary_since is not None
                and (now - _stationary_since) >= ENTRY_STATIONARY_CONFIRM_SECONDS
                and not _stationary_big_enough
                and not vehicle.get("_stationary_too_small_logged")
            ):
                vehicle["_stationary_too_small_logged"] = True
                print(f"[{cam_name}] ABAIKAN auto-confirm 'sudah DIAM': "
                      f"{vehicle['vehicle_id']} sudah diam {now - _stationary_since:.1f}s "
                      f"tapi best_area_ratio={vehicle.get('best_area_ratio', 0.0):.4f} < "
                      f"{MIN_STATIONARY_CONFIRM_AREA_RATIO} -- kemungkinan besar bukan "
                      f"kendaraan sungguhan (noise/objek statis latar), tetap menunggu "
                      f"sampai batas {ENTRY_MAX_EXTRA_WAIT_SECONDS:.0f}s.")

            _park_need = config.get("park_confirm_seconds", 0) or 0
            if _park_need > 0:
                # hanya kendaraan yang diam berturut-turut >= _park_need dtk yang dikonfirmasi (bukan yang sekadar lewat)
                ready_to_confirm = (
                    time_since_first_seen >= CONFIRM_SECONDS
                    and has_usable_crop
                    and _stationary_since is not None
                    and (now - _stationary_since) >= _park_need
                    and _stationary_big_enough
                )
            else:
                ready_to_confirm = time_since_first_seen >= CONFIRM_SECONDS and (
                    has_usable_crop
                    or _parked_long_enough
                    or time_since_first_seen >= CONFIRM_SECONDS + ENTRY_MAX_EXTRA_WAIT_SECONDS
                )

            if (
                DEBUG_MODE
                and ready_to_confirm
                and not has_usable_crop
                and _parked_long_enough
                and not vehicle.get("_parked_confirm_logged")
            ):
                vehicle["_parked_confirm_logged"] = True
                print(f"[{cam_name}] MASUK dikonfirmasi karena kendaraan sudah DIAM "
                      f"{now - _stationary_since:.1f}s (>= {ENTRY_STATIONARY_CONFIRM_SECONDS:.1f}s) "
                      f"-- tidak menunggu sampai batas "
                      f"{ENTRY_MAX_EXTRA_WAIT_SECONDS:.0f}s, karena pemandangan tidak akan "
                      f"membaik lagi: {vehicle['vehicle_id']}")
            if DEBUG_MODE and ready_to_confirm and not vehicle.get("_ready_to_confirm_logged"):
                vehicle["_ready_to_confirm_logged"] = True
                print(f"[DIAG-ENTRY-PHOTO-AREA-v1] {vehicle.get('vehicle_id', '?')}: "
                      f"ready_to_confirm=True -- time_since_first_seen={time_since_first_seen:.2f}s "
                      f"(CONFIRM_SECONDS={CONFIRM_SECONDS}s), has_usable_crop={has_usable_crop}, "
                      f"via_extra_wait={time_since_first_seen >= CONFIRM_SECONDS + ENTRY_MAX_EXTRA_WAIT_SECONDS}")

            if ready_to_confirm and _looks_like_static_fixture(vehicle):
                if DEBUG_MODE and not vehicle.get("_static_fixture_warned"):
                    vehicle["_static_fixture_warned"] = True
                    print(f"[{cam_name}] CATATAN: {vehicle['vehicle_id']} bbox-nya "
                          f"nyaris tidak bergerak/berubah ukuran sejak awal -- "
                          f"KEMUNGKINAN objek diam (pagar/fixture), tapi TETAP "
                          f"dikonfirmasi seperti biasa (hotfix v2, tidak lagi "
                          f"diblokir otomatis karena resiko salah tolak "
                          f"kendaraan asli yang jauh/lambat mendekat).")

            if ready_to_confirm:
                if not has_usable_crop and DEBUG_MODE:
                    print(f"[{cam_name}] MASUK dikonfirmasi TANPA crop usable "
                          f"(sudah nunggu {time_since_first_seen:.1f}s, lewat batas "
                          f"ekstra {ENTRY_MAX_EXTRA_WAIT_SECONDS:.1f}s) -- "
                          f"{vehicle['vehicle_id']}")
                crop_final = best_recent_crop(vehicle)
                if crop_final is not None:
                    vehicle["entry_display_crop"] = crop_final  # cadangan foto bersih untuk KELUAR

                if vehicle.get("cls_votes"):
                    final_cls = _weighted_cls(vehicle, cls)
                    final_confs = vehicle["cls_confs"].get(final_cls, [conf])
                    final_conf = sum(final_confs) / len(final_confs) if final_confs else conf
                else:
                    final_cls = cls
                    final_conf = conf

                vehicle["final_cls"] = final_cls
                vehicle["final_conf"] = final_conf

                vehicle["locked_cls"] = final_cls

                vehicle["locked_color"] = _voted_color(vehicle)

                if ACTIVITY_MODE:
                    status_muatan = activity_estimator.arrival_label(vehicle) or ACTIVITY_PENDING_LABEL
                elif config["cargo_detection"] and _is_truck_class(final_cls, config):
                    _latched = vehicle.get("cargo_in_latched")
                    if _latched is not None:
                        status_muatan = _latched[0]
                        print(
                            f"[{cam_name}] MUATAN AWAL dari kunci kedatangan: "
                            f"{status_muatan} (diambil {now - _latched[2]:.1f}s sebelum "
                            f"MASUK dikonfirmasi)"
                        )
                    else:
                        status_muatan, _ = best_cargo_detection(vehicle, debug_tag="MASUK")
                        if status_muatan == CARGO_UNKNOWN and not vehicle.get("is_box_body"):
                            _mv = vehicle.get("last_moment_cargo")
                            if (
                                _mv
                                and _mv["status"] != CARGO_UNKNOWN
                                and _mv["agreement"] >= CARGO_MOMENT_MIN_AGREEMENT
                                and _mv["votes"] >= CARGO_MOMENT_MIN_VOTES
                                and _mv["confidence"] >= CARGO_MIN_AVG_CONFIDENCE
                            ):
                                status_muatan = _mv["status"]
                                print(
                                    f"[{cam_name}] MUATAN AWAL dari voting momen "
                                    f"(riwayat temporal belum penuh): {status_muatan} "
                                    f"conf={_mv['confidence']:.2f} "
                                    f"sepakat={_mv['agreement']:.2f}"
                                )
                else:
                    status_muatan = CARGO_UNKNOWN

                tracks.mark_entered(vehicle, status_muatan, now)

                voted_cls = get_voted_class_from_history(
                    tracks.class_history, vehicle["vehicle_id"], final_cls, vehicle=vehicle
                )

                event_id = tracks.make_event_id(vehicle, f"MASUK_{vehicle.get('visit_no', 1):04d}")

                if config.get("sync_plate_to"):
                    vehicle["entry_event_sent"] = True
                elif not vehicle.get("entry_event_sent") and tracks.can_send_event(event_id):
                    arah, arah_score = classify_vehicle_direction(vehicle, config)
                    plat_text, plat_conf = None, 0.0
                    payload = build_payload(
                        config=config,
                        jenis_kejadian="MASUK",
                        cls=voted_cls,
                        conf=final_conf,
                        bbox=bbox,
                        status_muatan=status_muatan,
                        crop_enhanced=crop_final,
                        track_id=track_id,
                        vehicle_id=vehicle["vehicle_id"],
                        event_id=event_id,
                        arah_kendaraan=arah,
                        arah_confidence=arah_score,
                        plat_nomor=plat_text,
                        plat_confidence=plat_conf,
                        warna=vehicle["locked_color"][0],
                        warna_rgb=vehicle["locked_color"][1],
                        event_ts=vehicle.get("first_seen") or now,
                    )
                    kirim_ke_n8n(payload)
                    vehicle["entry_event_sent"] = True
                    print(f"[{cam_name}] MASUK: {vehicle['vehicle_id']} | "
                          f"cls={voted_cls} | muatan={status_muatan} | "
                          f"arah={arah} (score={arah_score:.2f})")
                else:
                    print(f"[{cam_name}] SKIP DUPLIKAT MASUK: {vehicle['vehicle_id']}")


def _has_usable_confirm_crop(vehicle, config):
    recent = vehicle.get("recent_crops") or []
    if not recent:
        return False
    min_area = config.get("min_confirm_crop_area", 0)
    if min_area <= 0:
        return True
    areas = [
        c["crop"].shape[0] * c["crop"].shape[1]
        for c in recent if c.get("crop") is not None
    ]
    result = any(a >= min_area for a in areas)
    if DEBUG_MODE and result and not vehicle.get("_usable_crop_area_logged"):
        vehicle["_usable_crop_area_logged"] = True
        print(f"[DIAG-ENTRY-PHOTO-AREA-v1] {vehicle.get('vehicle_id', '?')}: "
              f"has_usable_crop=True -- area terbesar={max(areas) if areas else 0}px^2 "
              f"(min_confirm_crop_area={min_area}px^2), total {len(recent)} crop di recent_crops, "
              f"areas={sorted(areas, reverse=True)[:5]}")
    return result


def check_lost_tracks(config, tracks, now, cam_name):
    if now < getattr(tracks, "lost_grace_until", 0.0):
        return  # [GAP-GUARD] masa tenang setelah stream putus
    for vehicle_id, vehicle in list(tracks.vehicles.items()):
        if not vehicle.get("entered"):
            continue
        if vehicle.get("session_closed"):
            continue

        active_last_seen = [
            tracks.tracks[tid]["last_seen"]
            for tid in vehicle.get("active_tracks", [])
            if tid in tracks.tracks
        ]
        if active_last_seen:
            latest_seen = max(active_last_seen)
        else:
            latest_seen = vehicle.get("last_seen", now)

        arah_cek, arah_score_cek = classify_vehicle_direction(vehicle, config)
        if arah_cek == "KELUAR" and arah_score_cek <= -DIRECTION_SCORE_THRESHOLD and not vehicle.get("is_stationary"):
            effective_lost_seconds = LOST_SECONDS
        else:
            effective_lost_seconds = STATIONARY_LOST_SECONDS

        session_start_ts = vehicle.get("time_in") or vehicle.get("first_seen") or now
        session_age = now - session_start_ts
        session_hard_max = config.get("session_hard_max_seconds", SESSION_HARD_MAX_SECONDS)
        force_close = (
            session_hard_max > 0
            and session_age > session_hard_max
        )

        if not force_close and now - latest_seen < effective_lost_seconds:
            continue

        if force_close:
            print(f"[{cam_name}] WATCHDOG: {vehicle['vehicle_id']} dipaksa ditutup -- "
                  f"sesi sudah terbuka {session_age / 60.0:.1f} menit "
                  f"(batas {session_hard_max / 60.0:.1f} menit), terakhir terlihat "
                  f"{now - latest_seen:.1f}s lalu. CEK: kemungkinan track sesi ini "
                  f"tercemar objek lain sehingga tidak pernah dianggap hilang.")

        crop_enhanced = best_recent_crop(vehicle)
        try:
            _exit_photo = None if _plate_priority(vehicle) else photo_before_exit(vehicle, float(latest_seen))
            if _exit_photo is not None:
                crop_enhanced = _exit_photo
        except Exception as _e:
            print(f"[FOTO] gagal memilih foto keluar, pakai foto terbaik biasa: {_e}")
        bbox = vehicle.get("last_bbox") or (0.0, 0.0, 0.0, 0.0)
        final_cls = vehicle.get("locked_cls")
        if final_cls is None:
            final_cls = vehicle.get("final_cls", -1)
        final_conf = vehicle.get("final_conf", 0.0)

        if ACTIVITY_MODE:
            status_muatan_keluar = ACTIVITY_PENDING_LABEL
        elif config["cargo_detection"] and _is_truck_class(final_cls, config):
            status_muatan_keluar, _ = best_cargo_detection(vehicle, debug_tag="KELUAR")
        else:
            status_muatan_keluar = CARGO_UNKNOWN

        # [FIX-EXIT-TIME] `now` = saat sesi DINYATAKAN hilang (terlambat LOST_SECONDS, atau STATIONARY_LOST_SECONDS=150 dtk
        # untuk truk yang dianggap diam). Waktu KELUAR yang dilaporkan = terakhir kali kendaraan benar-benar terlihat.
        # vehicle["time_out"] sengaja TIDAK diubah: logika RESUME/RETURN VISIT bergantung padanya.
        vehicle["time_out_seen"] = min(float(now), float(latest_seen))
        if tracks.finalize_exit(vehicle, status_muatan_keluar, now):
            if config.get("sync_plate_to"):
                if DEBUG_MODE:
                    print(f"[{cam_name}] (helper plat) sesi ditutup lokal, "
                          f"tidak kirim event -- {vehicle['vehicle_id']}")
            else:
                arah, arah_score = classify_vehicle_direction(vehicle, config)

                voted_cls = get_voted_class_from_history(
                    tracks.class_history, vehicle_id, final_cls, vehicle=vehicle
                )

                plat_text, plat_conf = None, 0.0

                _warna_keluar, _warna_rgb_keluar = (
                    vehicle.get("locked_color") or _voted_color(vehicle)
                )

                _t_masuk = vehicle.get("first_seen") or vehicle.get("time_in")
                _t_keluar = vehicle.get("time_out_seen") or vehicle["time_out"]
                if _t_masuk:
                    vehicle["duration_seconds"] = round(max(0.0, _t_keluar - _t_masuk), 2)
                payload = {
                    "timestamp_keluar": datetime.fromtimestamp(_t_keluar).strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_masuk": datetime.fromtimestamp(_t_masuk).strftime("%Y-%m-%d %H:%M:%S") if _t_masuk else "-",
                    "kamera_id": config["kamera_id"],
                    "kamera_nama": config["kamera_nama"],
                    "jenis_kendaraan": _jenis_with_activity(_jenis_label(voted_cls, config, vehicle),
                                                            vehicle.get("activity_type")),
                    "warna": _warna_keluar,
                    "rgb_r": int(_warna_rgb_keluar[0]), "rgb_g": int(_warna_rgb_keluar[1]), "rgb_b": int(_warna_rgb_keluar[2]),
                    "confidence": round(float(final_conf), 4),
                    "bbox_x1": int(bbox[0]), "bbox_y1": int(bbox[1]),
                    "bbox_x2": int(bbox[2]), "bbox_y2": int(bbox[3]),
                    "gambar_base64": encode_image_base64(crop_enhanced),
                    "vehicle_id": vehicle["vehicle_id"],
                    "status_muatan_awal": vehicle.get("cargo_in"),
                    "status_muatan_akhir": vehicle.get("cargo_out"),
                    "durasi_detik": vehicle.get("duration_seconds", 0),
                    "durasi_menit": round(vehicle.get("duration_seconds", 0) / 60.0, 2),
                    "status_aktivitas": vehicle.get("activity_type"),
                    "aktivitas_alasan": (vehicle.get("activity_info") or {}).get("reason"),
                    # "skor" = dari perubahan skor muatan; "asumsi" = aturan forklift/pintu tertutup; "tidak_pasti" = data kurang
                    "aktivitas_sumber": (vehicle.get("activity_info") or {}).get("source"),
                    "aktivitas_confidence": (vehicle.get("activity_info") or {}).get("confidence"),
                    "skor_muatan_awal": (vehicle.get("activity_info") or {}).get("score_in"),
                    "skor_muatan_akhir": (vehicle.get("activity_info") or {}).get("score_out"),
                    "aktivitas_sampel_awal": (vehicle.get("activity_info") or {}).get("n_in"),
                    "aktivitas_sampel_akhir": (vehicle.get("activity_info") or {}).get("n_out"),
                    "jenis_kejadian": "SIKLUS_SELESAI",
                    "track_id": vehicle.get("last_track_id", "UNKNOWN"),
                    "event_id": tracks.make_event_id(vehicle, f"SESSION_SUMMARY_{vehicle.get('visit_no', 1):04d}"),
                    "arah_kendaraan": arah,
                    "arah_confidence": round(float(arah_score), 3),
                    "plat_nomor": plat_text,
                    "plat_confidence": round(float(plat_conf), 3),
                }

                tracks.pending_exits[vehicle_id] = {
                    "payload": payload,
                    "visit_no": vehicle.get("visit_no", 1),
                    "close_timestamp": now,
                    "hold": PARKED_HOLD_SECONDS if vehicle.get("is_stationary") else MIN_GENUINE_REVISIT_GAP_SECONDS,
                }
                print(f"[{cam_name}] PENDING EXIT: {vehicle['vehicle_id']} ditutup, "
                      f"menunggu konfirmasi {tracks.pending_exits[vehicle_id]['hold']:.0f}s sebelum kirim webhook "
                      f"(dibatalkan otomatis kalau truk RESUME)")


def confirm_pending_exits(tracks, now, cam_name):
    for vehicle_id, pending in list(tracks.pending_exits.items()):
        vehicle = tracks.vehicles.get(vehicle_id)

        if vehicle is not None and not vehicle.get("session_closed") \
                and vehicle.get("visit_no", 1) == pending["visit_no"]:
            del tracks.pending_exits[vehicle_id]
            continue

        if now - pending["close_timestamp"] < pending.get("hold", MIN_GENUINE_REVISIT_GAP_SECONDS):
            continue

        payload = pending["payload"]
        if tracks.can_send_event(payload["event_id"]):
            kirim_ke_n8n(payload)
            print(f"[{cam_name}] SESI SELESAI (terkonfirmasi): {vehicle_id} | "
                  f"Aktivitas: {payload.get('status_aktivitas')} | "
                  f"Muatan Masuk: {payload['status_muatan_awal']} -> "
                  f"Muatan Keluar: {payload['status_muatan_akhir']} | "
                  f"Durasi: {payload['durasi_detik']}s")
        del tracks.pending_exits[vehicle_id]


def camera_loop(config):
    cam_name = config["kamera_id"]

    # [CAM1-ZONA] hanya kamera yang config-nya "pipeline": "zona" (cam1). cam2 lanjut ke kode lama di bawah, tidak diubah.
    # CAM1 memakai pipeline aktivitas di cam1.py; CAM2 tetap memakai loop aw.py di bawah.
    if cam_name == "cam1":
        try:
            import cam1
        except Exception as _e:
            print(f"[{cam_name}] !!! cam1.py gagal diimpor ({_e}) -> cam1 TIDAK berjalan")
            traceback.print_exc()
            return
        print(f"[{cam_name}] memakai cam1.py (pipeline baru)")
        return cam1.run_forever()


    tracks = TruckTrackState(
        cam_name,
        single_lane_mode=config.get("single_lane_mode", False),
        cargo_detection=config.get("cargo_detection", True),
        plate_detection=config.get("plate_detection", False),
        read_own_plate=config.get("read_own_plate", True),
    )

    while True:
        try:
            _camera_loop_body(config, tracks)
            print(f"[{cam_name}] WARNING: camera_loop_body berhenti. Restart dalam 5 detik...")
        except Exception:
            print(f"[{cam_name}] !!! CRASH di camera_loop:")
            traceback.print_exc()
            print(f"[{cam_name}] Restart dalam 5 detik... (status kendaraan aktif tetap disimpan)")
        time.sleep(5)


def _camera_loop_body(config, tracks):
    cam_name = config["kamera_id"]
    video_source = config["video_source"]
    roi_config = config["roi"]
    classes_allowed = config["classes_allowed"]
    is_file_source = is_local_video_file(video_source)

    print(f"\n> [{cam_name}] Memuat model: {config['model_path']}")
    model = YOLO(config["model_path"], task='detect')

    print(f"> [{cam_name}] Membuka sumber video: {_mask_url(video_source)}")
    cap = open_video_source(video_source)

    frame_count = 0
    consecutive_fail = 0
    frames_since_connect = 0

    consecutive_corrupt = 0
    MAX_CONSECUTIVE_CORRUPT = 45

    confidence_min = config.get("confidence_min", CONFIDENCE_MIN)
    print(f"> [{cam_name}] Mulai berjalan... classes={classes_allowed} "
          f"conf={confidence_min:.2f} tracker={TRACKER_CONFIG}")
    last_ok_now = None  # [GAP-GUARD]
    last_cam2_diagnostic = 0.0

    while True:
        ok = cap.grab()
        if not ok:
            if is_file_source:
                print(f"[{cam_name}] Video file habis, mengulang...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            consecutive_fail += 1
            if consecutive_fail > 10:
                print(f"[{cam_name}] WARNING: Reconnecting RTSP...")
                cap.release()
                time.sleep(3)
                cap = open_video_source(video_source)
                consecutive_fail = 0
                frames_since_connect = 0
            continue
        consecutive_fail = 0

        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue

        frames_since_connect += 1
        if frames_since_connect <= RECONNECT_GRACE_FRAMES:
            continue

        if _is_frame_corrupted(frame):
            consecutive_corrupt += 1
            if DEBUG_MODE:
                print(f"[{cam_name}] SKIP frame#{frame_count}: terdeteksi corrupt "
                      f"(blocky/noise), kemungkinan gangguan jaringan RTSP "
                      f"(beruntun={consecutive_corrupt})")
            if consecutive_corrupt > MAX_CONSECUTIVE_CORRUPT:
                print(f"[{cam_name}] WARNING: {consecutive_corrupt} frame corrupt "
                      f"beruntun -- stream kemungkinan degradasi kronis, "
                      f"reconnect paksa...")
                cap.release()
                time.sleep(3)
                cap = open_video_source(video_source)
                consecutive_fail = 0
                consecutive_corrupt = 0
                frames_since_connect = 0
            continue
        consecutive_corrupt = 0

        if frame_count == 0:
            print(f"[{cam_name}] Frame pertama berhasil! shape={frame.shape}")

        now = time.time()

        if last_ok_now is not None and (now - last_ok_now) > GAP_GUARD_SECONDS:
            _gap = now - last_ok_now
            tracks.apply_gap(_gap, now, GAP_GUARD_GRACE_SECONDS)
            print(f"[{cam_name}] [GAP] tidak ada frame valid {_gap:.1f}s (stream putus/frame rusak) -> "
                  f"sesi tidak ditutup selama {GAP_GUARD_GRACE_SECONDS:.0f}s, hitung konfirmasi MASUK diundur {_gap:.1f}s")
        last_ok_now = now

        cam_frame_skip = config.get("frame_skip", FRAME_SKIP)
        raw_frame_seen = frame_count + 1
        if raw_frame_seen % cam_frame_skip != 0:
            check_lost_tracks(config, tracks, now, cam_name)
            confirm_pending_exits(tracks, now, cam_name)
            tracks.cleanup(now)
            frame_count = raw_frame_seen
            continue

        roi_frame = extract_roi(frame, roi_config)
        if roi_frame.size == 0:
            continue
        _FULL_FRAME_CTX[cam_name] = (frame, int(frame.shape[1] * roi_config["x_min"]), int(frame.shape[0] * roi_config["y_min"]))

        if _is_frame_corrupted(roi_frame):
            if DEBUG_MODE:
                print(f"[{cam_name}] SKIP frame#{frame_count}: ROI terdeteksi "
                      f"corrupt (lolos cek frame penuh, tapi rusak di area ROI)")
            check_lost_tracks(config, tracks, now, cam_name)
            confirm_pending_exits(tracks, now, cam_name)
            tracks.cleanup(now)
            continue

        if NOISE_CORRUPT_CHECK_ENABLED or _NOISE_DEBUG_LOG_ALWAYS:
            _roi_noisy = _is_frame_noisy_corrupt(
                roi_frame, debug_label=f"{cam_name}/roi" if _NOISE_DEBUG_LOG_ALWAYS else None
            )
            if NOISE_CORRUPT_CHECK_ENABLED and _roi_noisy:
                if DEBUG_MODE:
                    print(f"[{cam_name}] SKIP frame#{frame_count}: ROI terdeteksi "
                          f"noise/dither (varians lokal tidak wajar)")
                check_lost_tracks(config, tracks, now, cam_name)
                confirm_pending_exits(tracks, now, cam_name)
                tracks.cleanup(now)
                continue

        if SAVE_DEBUG_IMAGES and frame_count < SAVE_DEBUG_ROI_FRAMES:
            cv2.imwrite(f"debug_{cam_name}_roi_{frame_count}.jpg", roi_frame)

        last_snap = _LAST_SNAPSHOT_TIME.get(cam_name, 0)
        if SAVE_DEBUG_IMAGES and now - last_snap >= SNAPSHOT_INTERVAL_SECONDS:
            try:
                os.makedirs("debug_snapshots", exist_ok=True)
                ts_label = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"debug_snapshots/{cam_name}_{ts_label}.jpg", roi_frame)
            except Exception:
                pass
            _LAST_SNAPSHOT_TIME[cam_name] = now

        result, scale = run_inference(model, roi_frame, classes_allowed, confidence_min)

        frame_count += 1

        if cam_name == "cam2" and now - last_cam2_diagnostic >= 10.0:
            last_cam2_diagnostic = now
            diagnostic_boxes = result.boxes if result and result.boxes is not None else None
            if diagnostic_boxes is None or len(diagnostic_boxes) == 0:
                detection_summary = "tidak ada deteksi"
            else:
                detection_ids = (
                    diagnostic_boxes.id.int().cpu().tolist()
                    if diagnostic_boxes.id is not None
                    else [None] * len(diagnostic_boxes)
                )
                roi_h, roi_w = roi_frame.shape[:2]
                detections = []
                for index, detection_id in enumerate(detection_ids):
                    x1, y1, x2, y2 = diagnostic_boxes.xyxy[index].cpu().tolist()
                    cx = ((x1 + x2) * 0.5 * scale) / max(roi_w, 1)
                    cy = ((y1 + y2) * 0.5 * scale) / max(roi_h, 1)
                    cls = int(diagnostic_boxes.cls[index])
                    conf = float(diagnostic_boxes.conf[index])
                    detections.append(
                        f"{_COCO_VEHICLE_NAMES.get(cls, cls)}:{conf:.2f}"
                        f"@({cx:.2f},{cy:.2f})"
                        f"#{detection_id if detection_id is not None else '-'}"
                    )
                detection_summary = " ; ".join(detections)
            print(f"[cam2] DIAG conf_min={confidence_min:.2f} "
                  f"tracks={len(tracks.tracks)} sesi_aktif="
                  f"{sum(not v.get('session_closed') for v in tracks.vehicles.values())} "
                  f"deteksi={detection_summary}")

        if DEBUG_MODE and frame_count % DEBUG_LOG_EVERY_N_FRAMES == 0:
            n_raw = len(result.boxes) if (result and result.boxes is not None) else 0
            has_id = bool(result and result.boxes is not None and result.boxes.id is not None)
            print(f"[{cam_name}] frame={frame_count} detections={n_raw} has_track_id={has_id}")

        handle_detections(config, result, scale, roi_frame, tracks, now, cam_name)
        check_lost_tracks(config, tracks, now, cam_name)
        confirm_pending_exits(tracks, now, cam_name)
        tracks.cleanup(now)

    cap.release()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def close(self):
        for s in self.streams:
            if s is sys.__stdout__ or s is sys.__stderr__:
                continue
            try:
                s.close()
            except Exception:
                pass

    def isatty(self):
        return False


class _RotatingLog:
    """File log yang otomatis ganti file saat melewati LOG_MAX_MB dan hanya menyimpan LOG_KEEP_FILES file terbaru."""

    def __init__(self, directory, prefix, max_bytes, keep):
        self.dir, self.prefix, self.max_bytes, self.keep = directory, prefix, max_bytes, keep
        self.fh, self.size, self.part, self.path = None, 0, 0, None
        self._bol = True  # [LOG-TS] awal baris -> beri jam
        self._lock = threading.Lock()
        self._open()

    def _open(self):
        self.part += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "" if self.part == 1 else f"_p{self.part}"
        self.path = os.path.join(self.dir, f"{self.prefix}{ts}{suffix}.txt")
        self.fh = open(self.path, "w", encoding="utf-8")
        self.size = 0
        self._prune()

    def _prune(self):
        try:
            files = [os.path.join(self.dir, f) for f in os.listdir(self.dir)
                     if f.startswith(self.prefix) and f.endswith(".txt")]
            files.sort(key=os.path.getmtime, reverse=True)
            for old in files[self.keep:]:
                if old != self.path:
                    try:
                        os.remove(old)
                    except Exception:
                        pass
        except Exception:
            pass

    def _stamp(self, data):
        # [LOG-TS] awalan jam [HH:MM:SS] di tiap baris log FILE (layar tidak diubah)
        out = []
        for chunk in data.splitlines(keepends=True):
            if self._bol:
                out.append(datetime.now().strftime("[%H:%M:%S] "))
            out.append(chunk)
            self._bol = chunk.endswith("\n")
        return "".join(out)

    def write(self, data):
        with self._lock:
            if self.fh is None:
                return
            data = self._stamp(data)
            n = len(data.encode("utf-8", "replace"))
            if self.size + n > self.max_bytes:
                try:
                    self.fh.close()
                except Exception:
                    pass
                self._open()
            self.fh.write(data)
            self.size += n

    def flush(self):
        with self._lock:
            if self.fh is not None:
                self.fh.flush()

    def close(self):
        with self._lock:
            if self.fh is not None:
                try:
                    self.fh.close()
                except Exception:
                    pass
                self.fh = None


LOG_MAX_MB = env("LOG_MAX_MB", float, required=False) or 100.0
LOG_KEEP_FILES = env("LOG_KEEP_FILES", int, required=False) or 14


def _setup_run_log():
    if isinstance(sys.stdout, _Tee):
        return None

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_handle = _RotatingLog(log_dir, "run_log_", int(LOG_MAX_MB * 1024 * 1024), LOG_KEEP_FILES)
    sys.stdout = _Tee(sys.__stdout__, log_file_handle)
    sys.stderr = _Tee(sys.__stderr__, log_file_handle)
    return log_file_handle.path


def _deep_model_check():
    """Cek ISI model, bukan hanya file ada. Return list peringatan (juga dipakai video.py)."""
    warn = []
    try:
        if forklift_filter is not None and forklift_filter.yolo_available():
            if forklift_filter._load_yolo() is None:
                warn.append("model YOLO forklift ADA tapi GAGAL dimuat (lihat pesan [forklift_filter] di atas) "
                            "-> gate YOLO forklift MATI, forklift akan lolos jadi 'kendaraan'")
    except Exception as e:
        warn.append(f"cek model YOLO forklift error: {e}")
    try:
        if cargo_classifier is not None and cargo_classifier.is_available():
            _b = cargo_classifier._load()
            if not _b:
                warn.append("cargo_model.pkl ada tapi GAGAL dimuat/ekstrak fitur -> semua skor muatan kosong")
            else:
                _cls = [str(c) for c in _b["model"].classes_]
                _miss = [c for c in ("forklift", "bukan_truk", "tertutup") if c not in _cls]
                if _miss:
                    warn.append(f"cargo_model.pkl hanya punya kelas {_cls}, TIDAK ada {_miss} -> forklift/boks "
                                f"tertutup dipaksa jadi Kosong/Bermuatan dan status Loading/Unloading jadi salah. "
                                f"Latih ulang: python train_classifier.py (isi dataset/forklift, tertutup, bukan_truk)")
    except Exception as e:
        warn.append(f"cek cargo_model.pkl error: {e}")
    return warn


def _preflight_check():
    """Cek file wajib SEBELUM kamera jalan. Return True bila aman; FATAL (file wajib hilang) -> False."""
    fatal, warn = [], []
    if not os.path.exists(TRACKER_CONFIG):
        fatal.append(f"file tracker '{TRACKER_CONFIG}' tidak ada di {os.getcwd()} (tanpa ini tracking gagal terus)")
    for key, cfg in CAMERAS.items():
        mp = cfg.get("model_path")
        if not mp or not os.path.exists(mp):
            fatal.append(f"{key}: model deteksi '{mp}' tidak ditemukan")
        vs = cfg.get("video_source")
        if vs and is_local_video_file(vs) and not os.path.isfile(vs):
            fatal.append(f"{key}: file video '{vs}' tidak ditemukan")
        if cfg.get("forklift_gate"):
            if forklift_filter is None:
                warn.append(f"{key}: forklift_filter.py gagal diimpor -> gate forklift MATI")
            elif not forklift_filter.yolo_available():
                warn.append(f"{key}: FORKLIFT_YOLO_PATH tidak diisi/tidak ditemukan -> gate YOLO forklift MATI")
    if cargo_classifier is None or not cargo_classifier.is_available():
        warn.append("cargo_model.pkl tidak ditemukan -> skor muatan pakai rule-based")
    warn.extend(_deep_model_check())
    if ACTIVITY_MODE is False:
        warn.append("ACTIVITY_MODE nonaktif (activity_estimator tidak ter-import atau ACTIVITY_MODE=false)")
    for w in warn:
        print(f"[PREFLIGHT][PERINGATAN] {w}")
    for f in fatal:
        print(f"[PREFLIGHT][FATAL] {f}")
    return not fatal


def main():
    cv2.setNumThreads(2)

    log_path = _setup_run_log()
    if log_path:
        print(f"[INFO] Semua output juga disimpan ke: {log_path}")

    print("=" * 60)
    print("Sistem Deteksi Kendaraan Dual Camera - v4")
    print(f"Cam1: {CAMERAS['cam1']['kamera_nama']} - {CAMERAS['cam1']['classes_allowed']}")
    print(f"Cam2: {CAMERAS['cam2']['kamera_nama']} - {CAMERAS['cam2']['classes_allowed']}")
    print("=" * 60)

    if not _preflight_check():
        print("!! Preflight gagal -- perbaiki file yang hilang di atas lalu jalankan ulang.")
        sys.exit(2)

    threading.Thread(target=_n8n_outbox_worker, daemon=True).start()

    threads = []
    for cam_key in CAMERAS:
        t = threading.Thread(target=camera_loop, args=(CAMERAS[cam_key],), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)

    print("\n> Semua kamera berjalan. Tekan Ctrl+C untuk berhenti.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n> Dihentikan oleh user.")
    finally:
        tunggu_semua_kirim_n8n()
        if isinstance(sys.stdout, _Tee):
            tee = sys.stdout
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            tee.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"> FATAL ERROR: {e}")
        traceback.print_exc()