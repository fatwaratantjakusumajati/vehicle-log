# cargo_classifier.py - wrapper model muatan (cargo_model.pkl).
# Tambahan: forklift_probability() untuk P(forklift); None bila model belum dilatih dengan kelas forklift.

import os
import pickle

import numpy as np

CARGO_MODEL_PATH = os.environ.get("CARGO_MODEL_PATH", "cargo_model.pkl")

LABEL_TO_STATUS = {
    "muatan": "Bermuatan",
    "muatan_terpal": "Bermuatan",
    "kosong": "Kosong",
}

# "tertutup" = bak/boks/pintu tertutup sehingga isi tidak terlihat: sampel diabaikan (bukan ditebak).
REJECT_LABELS = {"forklift", "bukan_truk", "tertutup"}

FORKLIFT_LABELS = {"forklift"}

CONFIDENCE_THRESHOLD = float(os.environ.get("CARGO_ML_CONF", "0.70"))
MARGIN_THRESHOLD = float(os.environ.get("CARGO_ML_MARGIN", "0.15"))
VERBOSE = os.environ.get("CARGO_ML_VERBOSE", "0") not in ("0", "", "false", "False")
# [FIX-SKOR-JENUH] Skor bak (loaded_probability) bisa jenuh di ~1.00 sehingga "banyak palet" dan "sedikit palet" tampak sama.
# CARGO_SOFTEN_T > 1 membagi logit dengan T sebelum sigmoid (skala tetap 0..1, urutan tetap): skor tidak lagi menempel di 1.00.
# 1.0 = perilaku lama.
SOFTEN_T = max(1.0, float(os.environ.get("CARGO_SOFTEN_T", "1.0")))

try:
    from feature_extractor import extract_features
    _FEATURE_AVAILABLE = True
except Exception as _e:
    _FEATURE_AVAILABLE = False
    print(f"[cargo_classifier] feature_extractor tidak tersedia ({_e})")

_bundle = None
_load_attempted = False


def is_available():
    return os.path.isfile(CARGO_MODEL_PATH)


def _load():
    global _bundle, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        if _FEATURE_AVAILABLE and os.path.isfile(CARGO_MODEL_PATH):
            try:
                with open(CARGO_MODEL_PATH, "rb") as f:
                    _bundle = pickle.load(f)
                print(f"[cargo_classifier] Model dimuat dari '{CARGO_MODEL_PATH}' "
                      f"(conf>={CONFIDENCE_THRESHOLD}, margin>={MARGIN_THRESHOLD}, soften_T={SOFTEN_T:g})")
            except Exception as e:
                print(f"[cargo_classifier] Gagal load model: {e}")
                _bundle = None
    return _bundle


def aggregate_by_status(proba, classes):
    agg = {}
    for label, p in zip(classes, proba):
        label = str(label)
        key = "ditolak" if label in REJECT_LABELS else LABEL_TO_STATUS.get(label, label)
        agg[key] = agg.get(key, 0.0) + float(p)
    return agg


def _validate_prediction(proba, classes):
    proba = np.asarray(proba, dtype=float)
    if proba.size == 0:
        return None

    agg = aggregate_by_status(proba, classes)
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    top_status, top_p = ranked[0]
    second_p = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_status == "ditolak":
        if VERBOSE:
            print(f"[cargo_classifier] ditolak (kelas non-muatan unggul): {agg}")
        return None

    if top_p < CONFIDENCE_THRESHOLD:
        if VERBOSE:
            print(f"[cargo_classifier] confidence rendah {top_p:.2f} < "
                  f"{CONFIDENCE_THRESHOLD}: {agg}")
        return None

    if (top_p - second_p) < MARGIN_THRESHOLD:
        if VERBOSE:
            print(f"[cargo_classifier] margin kecil {top_p - second_p:.2f} < "
                  f"{MARGIN_THRESHOLD}: {agg}")
        return None

    return top_status, top_p


def predict(crop_bgr):
    if not _FEATURE_AVAILABLE or crop_bgr is None or crop_bgr.size == 0:
        return None

    bundle = _load()
    if bundle is None:
        return None

    try:
        feat = extract_features(crop_bgr).reshape(1, -1)
        model = bundle["model"]
        proba = model.predict_proba(feat)[0]
        return _validate_prediction(proba, model.classes_)
    except Exception as e:
        print(f"[cargo_classifier] Gagal prediksi: {e}")
        return None


def loaded_probability(crop_bgr):
    if not _FEATURE_AVAILABLE or crop_bgr is None or crop_bgr.size == 0:
        return None
    bundle = _load()
    if bundle is None:
        return None
    try:
        feat = extract_features(crop_bgr).reshape(1, -1)
        model = bundle["model"]
        agg = aggregate_by_status(model.predict_proba(feat)[0], model.classes_)
        if agg.get("ditolak", 0.0) > 0.5:
            return None
        p_load = agg.get("Bermuatan", 0.0)
        p_empty = agg.get("Kosong", 0.0)
        total = p_load + p_empty
        if total <= 0:
            return None
        p = p_load / total
        if SOFTEN_T > 1.0:
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            p = 1.0 / (1.0 + float(np.exp(-np.log(p / (1.0 - p)) / SOFTEN_T)))
        return p
    except Exception as e:
        print(f"[cargo_classifier] Gagal loaded_probability: {e}")
        return None


def forklift_probability(crop_bgr):
    if not _FEATURE_AVAILABLE or crop_bgr is None or crop_bgr.size == 0:
        return None
    bundle = _load()
    if bundle is None:
        return None
    try:
        model = bundle["model"]
        classes = [str(c) for c in model.classes_]
        if not any(c in FORKLIFT_LABELS for c in classes):
            return None
        proba = model.predict_proba(extract_features(crop_bgr).reshape(1, -1))[0]
        return float(sum(p for c, p in zip(classes, proba) if c in FORKLIFT_LABELS))
    except Exception as e:
        print(f"[cargo_classifier] Gagal forklift_probability: {e}")
        return None