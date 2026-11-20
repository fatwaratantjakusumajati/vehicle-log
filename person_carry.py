# person_carry.py - "orang membawa barang atau kosong". Drop-in pengganti person_carry.py lama.
# Prioritas: model fine-tune person_carry_cls.pt (YOLO11-cls). Bila tidak ada, jatuh ke model lama person_carry_model.pkl.
import os
import pickle

# Urutan: PERSON_CARRY_CLS_PATH (bila diisi) -> folder OpenVINO person_carry_cls_openvino_model -> person_carry_cls.pt
_OV_DIR = "person_carry_cls_openvino_model"
CLS_PATH = os.environ.get("PERSON_CARRY_CLS_PATH") or (_OV_DIR if os.path.isdir(_OV_DIR) else "person_carry_cls.pt")
PERSON_CARRY_MODEL_PATH = os.environ.get("PERSON_CARRY_MODEL_PATH", "person_carry_model.pkl")
POSITIVE_LABEL = "orang_bawa"

try:
    from feature_extractor import extract_features
    _FEATURE_AVAILABLE = True
except Exception as _e:
    _FEATURE_AVAILABLE = False

_cls = None
_cls_idx = None
_cls_tried = False
_bundle = None
_bundle_tried = False


def _load_cls():
    global _cls, _cls_idx, _cls_tried
    if not _cls_tried:
        _cls_tried = True
        if os.path.exists(CLS_PATH):
            try:
                from ultralytics import YOLO
                m = YOLO(CLS_PATH, task="classify")
                idx = [k for k, v in m.names.items() if v == "bawa"]
                if idx:
                    _cls, _cls_idx = m, idx[0]
                    print(f"[person_carry] Model fine-tune dimuat dari '{CLS_PATH}'")
                else:
                    print(f"[person_carry] '{CLS_PATH}' tidak punya kelas 'bawa'")
            except Exception as e:
                print(f"[person_carry] Gagal load model fine-tune: {e}")
    return _cls


def _load_pkl():
    global _bundle, _bundle_tried
    if not _bundle_tried:
        _bundle_tried = True
        if _FEATURE_AVAILABLE and os.path.isfile(PERSON_CARRY_MODEL_PATH):
            try:
                with open(PERSON_CARRY_MODEL_PATH, "rb") as f:
                    _bundle = pickle.load(f)
                print(f"[person_carry] Model lama dimuat dari '{PERSON_CARRY_MODEL_PATH}'")
            except Exception as e:
                print(f"[person_carry] Gagal load model lama: {e}")
    return _bundle


def is_available():
    return os.path.exists(CLS_PATH) or os.path.isfile(PERSON_CARRY_MODEL_PATH)


def carry_probability(crop_bgr):
    """P(orang membawa barang) 0..1, atau None bila model/crop tidak tersedia."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    m = _load_cls()
    if m is not None:
        try:
            r = m.predict(crop_bgr, imgsz=224, verbose=False)[0]
            return float(r.probs.data[_cls_idx])
        except Exception as e:
            print(f"[person_carry] Gagal prediksi (fine-tune): {e}")
            return None
    bundle = _load_pkl()
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