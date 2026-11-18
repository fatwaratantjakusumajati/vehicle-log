# person_carry.py - classifier "orang membawa barang (dus/palet) atau tangan kosong".
# Memakai backbone MobileNetV2 yang sama dengan forklift_carry/cargo_classifier (feature_extractor.py).
# Model dilatih lewat: python train_classifier.py --data data_orang --out person_carry_model.pkl

import os
import pickle

PERSON_CARRY_MODEL_PATH = os.environ.get("PERSON_CARRY_MODEL_PATH", "person_carry_model.pkl")
POSITIVE_LABEL = "orang_bawa"
NEGATIVE_LABEL = "orang_kosong"

try:
    from feature_extractor import extract_features
    _FEATURE_AVAILABLE = True
except Exception as _e:
    _FEATURE_AVAILABLE = False
    print(f"[person_carry] feature_extractor tidak tersedia ({_e})")

_bundle = None
_load_attempted = False


def is_available():
    return os.path.isfile(PERSON_CARRY_MODEL_PATH)


def _load():
    global _bundle, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        if _FEATURE_AVAILABLE and os.path.isfile(PERSON_CARRY_MODEL_PATH):
            try:
                with open(PERSON_CARRY_MODEL_PATH, "rb") as f:
                    _bundle = pickle.load(f)
                print(f"[person_carry] Model dimuat dari '{PERSON_CARRY_MODEL_PATH}'")
            except Exception as e:
                print(f"[person_carry] Gagal load model: {e}")
                _bundle = None
    return _bundle


def carry_probability(crop_bgr):
    """P(orang membawa barang) 0..1, atau None bila model/crop tidak tersedia."""
    if not _FEATURE_AVAILABLE or crop_bgr is None or crop_bgr.size == 0:
        return None
    bundle = _load()
    if bundle is None:
        return None
    try:
        model = bundle["model"]
        classes = [str(c) for c in model.classes_]
        if POSITIVE_LABEL not in classes:
            return None
        proba = model.predict_proba(extract_features(crop_bgr).reshape(1, -1))[0]
        return float(proba[classes.index(POSITIVE_LABEL)])
    except Exception as e:
        print(f"[person_carry] Gagal prediksi: {e}")
        return None
