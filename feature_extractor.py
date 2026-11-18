"""
Modul bersama untuk ekstrak "fitur" (vektor 1280) dari crop gambar memakai MobileNetV2 ImageNet (transfer learning).
Backbone DIBEKUKAN; classifier ringan (cargo_model.pkl, forklift_carry_model.pkl) dilatih di atas vektor ini.

Dipakai oleh: train_classifier.py, train_fk_carry.py, cargo_classifier.py, forklift_carry.py -> cam1.py / aw.py

BACKEND (otomatis):
  1. OpenVINO  bila folder  mobilenetv2_features_openvino_model/  ada (buat dengan  python convert_feature_extractor.py).
               Tidak butuh torch. Lokasi bisa diubah dengan FEATURE_OV_PATH (folder atau file .xml).
  2. torch     bila folder itu tidak ada (perilaku lama, bobot ImageNet dari cache torch).
FEATURE_BACKEND=torch memaksa torch; FEATURE_BACKEND=openvino memaksa OpenVINO (error bila tidak ada).
Kedua backend memberi fitur yang sama (diperiksa saat konversi), jadi file .pkl tidak perlu dilatih ulang.
"""
import os
import threading

import cv2
import numpy as np

FEATURE_DIM = 1280
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
_BASE = os.path.dirname(os.path.abspath(__file__))

_lock = threading.Lock()      # cam1 & cam2 bisa memanggil bersamaan
_ov = None                    # compiled model OpenVINO
_torch = None                 # (model, torch) bila memakai torch
_backend = None


def preprocess(image_bgr):
    """BGR uint8 HxWx3 -> (1,3,224,224) float32. Identik dengan torchvision: ToPILImage -> Resize((224,224), bilinear) -> ToTensor -> Normalize."""
    from PIL import Image
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray((x - _MEAN) / _STD, np.float32)


def _ov_xml():
    p = os.environ.get("FEATURE_OV_PATH", "").strip().strip('"') or os.path.join(_BASE, "mobilenetv2_features_openvino_model")
    if os.path.isdir(p):
        p = os.path.join(p, "mobilenetv2_features.xml")
    return p if os.path.exists(p) else None


def _load():
    global _ov, _torch, _backend
    if _backend:
        return
    with _lock:
        if _backend:
            return
        want = os.environ.get("FEATURE_BACKEND", "").strip().lower()
        xml = _ov_xml() if want != "torch" else None
        if xml:
            import openvino as ov
            core = ov.Core()
            # 1 thread seperti versi torch lama: kamera lain (RTSP, YOLO) tidak boleh diganggu.
            _ov = core.compile_model(xml, "CPU", {"INFERENCE_NUM_THREADS": 1, "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"})
            _backend = "openvino"
        elif want == "openvino":
            raise RuntimeError("FEATURE_BACKEND=openvino tetapi mobilenetv2_features.xml tidak ditemukan "
                               "(jalankan python convert_feature_extractor.py)")
        else:
            import torch
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            torch.set_num_threads(1)
            m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
            m.classifier = torch.nn.Identity()
            m.eval()
            _torch = (m, torch)
            _backend = "torch"
        try:
            import logging
            logging.getLogger("cam1").info("feature_extractor: backend=%s", _backend)
        except Exception:
            pass
        print(f"[feature_extractor] backend={_backend}")


def backend():
    _load()
    return _backend


def extract_features(image_bgr):
    """image_bgr: gambar OpenCV (BGR, uint8, HxWx3). Return: np.array shape (1280,)"""
    _load()
    x = preprocess(image_bgr)
    if _backend == "openvino":
        with _lock:                       # satu infer request bersama -> serialkan
            out = _ov(x)[0]
        return np.asarray(out, np.float32).reshape(-1)
    m, torch = _torch
    with torch.no_grad():
        return m(torch.from_numpy(x)).squeeze(0).numpy().astype(np.float32)
