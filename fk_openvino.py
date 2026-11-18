"""
fk_openvino.py - detektor forklift OpenVINO TANPA postprocess ultralytics.

Kenapa: postprocess ultralytics menebak format keluaran model (mis. dari metadata / jumlah kolom). Untuk model
dengan 2 kelas (4 kotak + 2 kelas = 6 kolom) tebakan itu bisa keliru -> ratusan deteksi palsu. Di sini keluaran mentah
OpenVINO dibaca langsung dan di-decode sendiri (letterbox -> decode -> NMS -> koordinat asli), jadi hasilnya bisa
diperiksa dan tidak bergantung pada versi ultralytics.

Dipakai oleh cam1.py (ForkliftDetector) dan convert_forklift.py (verifikasi).
Yang dibutuhkan: openvino, opencv, numpy, pyyaml (semua ikut terpasang bersama ultralytics).
"""
import os

import cv2
import numpy as np


def letterbox(img, size, color=114):
    """Skala + pad ke size x size (sama dengan ultralytics). Return (gambar, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) / 2.0, (size - nh) / 2.0
    top, bottom = int(round(py - 0.1)), int(round(py + 0.1))
    left, right = int(round(px - 0.1)), int(round(px + 0.1))
    img = cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, float(left), float(top)


def decode(out, class_ids, conf, iou, max_det=100):
    """
    out: keluaran mentah OpenVINO. Format yang dikenali:
      (1, 4+nc, N)  YOLOv8 standar: xywh (piksel input) + skor kelas (sudah sigmoid)
      (1, N, 4+nc)  versi transposisi
      (1, 300, 6)   end-to-end: x1,y1,x2,y2,conf,cls
    Return: list [x1, y1, x2, y2, conf, cls] dalam piksel gambar letterbox.
    """
    a = np.asarray(out)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"bentuk keluaran tak dikenali: {np.asarray(out).shape}")
    rows, cols = a.shape
    # end-to-end: (<=1000, 6 atau 7) -> x1,y1,x2,y2,conf,cls. (Standar punya N >= ~2000 anchor, jadi tidak tertukar.)
    if cols in (6, 7) and rows <= 1000:
        keep = [r for r in a if r[4] >= conf and int(r[5]) in class_ids]
        return [[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[5])] for r in keep][:max_det]
    if rows > cols:                               # (N, 4+nc) -> (4+nc, N)
        a = a.T
        rows, cols = a.shape
    if not class_ids or max(class_ids) + 4 >= rows:
        raise ValueError(f"bentuk keluaran {a.shape} tidak cocok dengan kelas {class_ids}")
    scores = a[4:, :]                             # (nc, N)
    sel = scores[class_ids, :]                    # hanya kelas forklift
    best = sel.max(axis=0)
    cls_local = sel.argmax(axis=0)
    m = best >= conf
    if not np.any(m):
        return []
    xywh, best, cls_local = a[:4, m].T, best[m], cls_local[m]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), best.astype(float).tolist(), float(conf), float(iou))
    idx = np.array(idx).reshape(-1)[:max_det]
    res = []
    for i in idx:
        x, y, w, h = boxes[i]
        res.append([float(x), float(y), float(x + w), float(y + h), float(best[i]), int(class_ids[int(cls_local[i])])])
    return res


def _read_meta(folder):
    import yaml
    p = os.path.join(folder, "metadata.yaml")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"metadata.yaml tidak ada di {folder}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class DirectForklift:
    """Muat folder OpenVINO (hasil convert_forklift.py), kembalikan kotak forklift untuk satu gambar BGR."""

    def __init__(self, folder, conf=0.35, iou=0.45, name_contains="forklift", threads=0):
        import openvino as ov
        meta = _read_meta(folder)
        names = {int(k): str(v) for k, v in (meta.get("names") or {}).items()}
        self.class_ids = [k for k, v in names.items() if name_contains in v.lower()]
        if not self.class_ids:
            raise ValueError(f"tidak ada kelas '{name_contains}' di model (kelas: {names})")
        isz = meta.get("imgsz", 640)
        self.size = int(isz[0] if isinstance(isz, (list, tuple)) else isz)
        self.names, self.conf, self.iou = names, float(conf), float(iou)
        xml = [f for f in os.listdir(folder) if f.endswith(".xml")]
        if not xml:
            raise FileNotFoundError(f"tidak ada .xml di {folder}")
        core = ov.Core()
        model = core.read_model(os.path.join(folder, xml[0]))
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = int(threads)
        self.compiled = core.compile_model(model, "CPU", cfg)
        self.out = self.compiled.output(0)

    def raw(self, img_bgr):
        lb, r, px, py = letterbox(img_bgr, self.size)
        x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return self.compiled(x)[self.out], r, px, py

    def __call__(self, img_bgr, conf=None):
        out, r, px, py = self.raw(img_bgr)
        h, w = img_bgr.shape[:2]
        dets = decode(out, self.class_ids, self.conf if conf is None else conf, self.iou)
        res = []
        for x1, y1, x2, y2, c, k in dets:
            x1, x2 = (x1 - px) / r, (x2 - px) / r
            y1, y2 = (y1 - py) / r, (y2 - py) / r
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            if x2 - x1 > 1 and y2 - y1 > 1:
                res.append((x1, y1, x2, y2, c, k))
        return res
