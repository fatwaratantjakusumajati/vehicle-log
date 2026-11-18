"""
extract_crops.py - potong crop kendaraan (dan forklift) dari rekaman NVR untuk bahan latih classifier.

Contoh (di folder plate-det, Windows):
  python extract_crops.py "D:\\rekaman_nvr" --out dataset_raw --zones cam1_zones.json
  python extract_crops.py video1.mp4 video2.mp4 --every 6 --fk-model forklift_openvino_model

Hasil:
  dataset_raw/<nama_video>/<video>_<detik>_<kelas>_<conf>_<no>.jpg   crop kendaraan (car/truck)
  dataset_raw/forklift/...                                           crop forklift (jika --fk-model)
  dataset_raw/manifest.csv                                           daftar semua crop (video, detik, posisi, ukuran)
MODE ORANG (untuk melatih classifier "orang membawa barang"):
  python extract_crops.py rekaman_manual.mp4 --person --zones cam1_zones.json --out dataset_raw_orang
  -> crop orang (1 frame/dtk), dikelompokkan per video per blok 5 menit: dataset_raw_orang/<video>_blk00/...
  Lalu pilah cepat dengan:  python sortir_crop.py --src dataset_raw_orang
Setelah itu PILAH MANUAL: pindahkan crop ke dataset/kosong, dataset/muatan, dataset/tertutup,
dataset/bukan_truk, dataset/forklift (sesuai train_classifier.py).
Filter ukuran & zona dibuat sama dengan cam1.py supaya crop mirip yang dilihat classifier saat live.
"""
import argparse
import csv
import os
import sys

import cv2

VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".ts", ".flv", ".wmv", ".mpg", ".mpeg")


# ----------------------------------------------------------------------------- zona
def inside(pt, poly):
    x, y = pt
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


class Zones:
    """Zona normal 0..1 dari cam1_zones.json. in_area = di dalam 'area' dan di luar semua 'exclude' (sama seperti cam1.py)."""

    def __init__(self, path):
        import json
        d = json.load(open(path, encoding="utf-8"))
        self.area = d.get("area") or []
        self.exclude = [p for p in (d.get("exclude") or []) if len(p) >= 3]
        xs = [p[0] for p in self.area]
        ys = [p[1] for p in self.area]
        # luas ROI = bbox poligon area (normal), dipakai sebagai pembagi rasio ukuran seperti cam1.py
        self.roi_area = max(1e-6, (max(xs) - min(xs)) * (max(ys) - min(ys))) if self.area else 1.0

        self.stage = d.get("stage") or []

    def in_area_only(self, nx, ny):
        """Di dalam poligon 'area' saja (exclude TIDAK berlaku untuk orang: orang boleh berjalan di ramp)."""
        return (not self.area) or inside((nx, ny), self.area)

    def in_stage(self, nx, ny):
        return len(self.stage) >= 3 and inside((nx, ny), self.stage)

    def in_area(self, nx, ny):
        if self.area and not inside((nx, ny), self.area):
            return False
        return not any(inside((nx, ny), p) for p in self.exclude)


# ----------------------------------------------------------------------------- detektor
class YoloDet:
    """Pembungkus ultralytics. detect(frame) -> [(x1,y1,x2,y2,conf,cls)]"""

    def __init__(self, path, conf, iou, classes, imgsz):
        from ultralytics import YOLO
        self.m = YOLO(path, task="detect")
        self.conf, self.iou, self.classes, self.imgsz = conf, iou, classes, imgsz

    def _predict(self, frame):
        return self.m.predict(frame, conf=self.conf, iou=self.iou, classes=self.classes, imgsz=self.imgsz, verbose=False)[0]

    def detect(self, frame):
        try:
            r = self._predict(frame)
        except RuntimeError as ex:
            # model OpenVINO punya ukuran input TETAP (mis. 640): ambil dari pesan error lalu ulangi dengan ukuran itu
            import re
            m = re.search(r"expecting \[1,\s*3,\s*(\d+),\s*(\d+)\]", str(ex))
            if not m:
                raise
            self.imgsz = int(m.group(1))
            print(f"  (ukuran input model otomatis diset ke {self.imgsz})", flush=True)
            r = self._predict(frame)
        out = []
        if r.boxes is None:
            return out
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append((x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
        return out


# ----------------------------------------------------------------------------- inti
def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def crop_pad(frame, box, pad):
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0, int(box[0] - pad * bw))
    y1 = max(0, int(box[1] - pad * bh))
    x2 = min(w, int(box[2] + pad * bw))
    y2 = min(h, int(box[3] + pad * bh))
    return frame[y1:y2, x1:x2]


class Saver:
    """Simpan crop + cegah banyak duplikat dari kendaraan yang parkir diam (posisi sama dalam jeda waktu pendek)."""

    def __init__(self, out_dir, manifest, static_gap, static_dist, max_per_video):
        self.out_dir, self.manifest = out_dir, manifest
        self.static_gap, self.static_dist, self.max_per_video = static_gap, static_dist, max_per_video
        self.recent = []      # (nx, ny, t, kind)
        self.n = 0

    def reset(self):
        self.recent, self.n = [], 0

    def try_save(self, crop, kind, stem, t, cls_name, conf, nx, ny, ratio, folder, no, min_side, min_pixels, pad_box):
        if self.max_per_video and self.n >= self.max_per_video:
            return False
        h, w = crop.shape[:2]
        if min(h, w) < min_side or h * w < min_pixels:
            return False
        for (px, py, pt, pk) in self.recent:
            if pk == kind and abs(px - nx) < self.static_dist and abs(py - ny) < self.static_dist and (t - pt) < self.static_gap:
                return False
        os.makedirs(folder, exist_ok=True)
        name = f"{stem}_{int(t):06d}_{cls_name}_{conf:.2f}_{no}.jpg"
        path = os.path.join(folder, name)
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.recent.append((nx, ny, t, kind))
        self.recent = self.recent[-80:]
        self.n += 1
        self.manifest.writerow([stem, f"{t:.1f}", kind, cls_name, f"{conf:.2f}", f"{nx:.3f}", f"{ny:.3f}",
                                "" if ratio is None else f"{ratio:.4f}", w, h, os.path.relpath(path, self.out_dir)])
        return True


def process_video(path, veh_det, fk_det, zones, args, saver):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ! tidak bisa dibuka: {path}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 120:
        fps = 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    step = max(1, int(round(fps * args.every)))
    stem = os.path.splitext(os.path.basename(path))[0]
    veh_folder = os.path.join(args.out, stem)
    fk_folder = os.path.join(args.out, "forklift")
    saver.reset()
    idx = 0
    saved = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                idx += 1
                continue
            t = idx / fps
            H, W = frame.shape[:2]
            if args.person:
                for i, d in enumerate(veh_det.detect(frame)):
                    x1, y1, x2, y2, conf, cls = d
                    if cls != 0 or conf < args.conf:
                        continue
                    fx, fy = (x1 + x2) / 2 / W, y2 / H            # titik kaki orang
                    if zones is not None and not zones.in_area_only(fx, fy):
                        continue
                    if args.stage_only and not (zones is not None and zones.in_stage(fx, fy)):
                        continue
                    blk = int(t // (60 * args.block_min))
                    folder = os.path.join(args.out, f"{stem}_blk{blk:02d}")
                    if saver.try_save(crop_pad(frame, d, args.pad), "orang", stem, t, "orang", conf, fx, (y1 + y2) / 2 / H,
                                      None, folder, i, args.min_side, args.min_pixels, d):
                        saved += 1
                idx += 1
                continue
            fks = []
            if fk_det is not None:
                fks = [d for d in fk_det.detect(frame) if d[4] >= args.fk_conf]
                for i, d in enumerate(fks):
                    if saver.try_save(crop_pad(frame, d, args.pad), "forklift", stem, t, "forklift", d[4],
                                      (d[0] + d[2]) / 2 / W, (d[1] + d[3]) / 2 / H, None, fk_folder, i,
                                      args.min_side, args.min_pixels, d):
                        saved += 1
            for i, d in enumerate(veh_det.detect(frame)):
                x1, y1, x2, y2, conf, cls = d
                if conf < args.conf or cls not in (2, 7):
                    continue
                nx, ny = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                if zones is not None and not zones.in_area(nx, ny):
                    continue
                ratio = box_area(d) / (W * H) / zones.roi_area if zones is not None else box_area(d) / (W * H)
                if ratio < args.min_ratio:
                    continue
                # kotak yang sebagian besar adalah forklift = bukan kendaraan (sama seperti cam1.py)
                if any(inter_area(d, f) / max(1e-6, box_area(d)) >= 0.6 for f in fks):
                    continue
                if saver.try_save(crop_pad(frame, d, args.pad), "kendaraan", stem, t, "car" if cls == 2 else "truck",
                                  conf, nx, ny, ratio, veh_folder, i, args.min_side, args.min_pixels, d):
                    saved += 1
            if args.progress and total and (idx // step) % 50 == 0:
                print(f"  {stem}: {idx / total * 100:4.0f}%  tersimpan {saved}", flush=True)
        idx += 1
    cap.release()
    return saved


def collect_videos(inputs):
    vids = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.lower().endswith(VIDEO_EXT):
                        vids.append(os.path.join(root, f))
        elif os.path.isfile(p):
            vids.append(p)
        else:
            print(f"! tidak ditemukan: {p}")
    return vids


def main(argv=None, veh_det=None, fk_det=None):
    ap = argparse.ArgumentParser(description="Potong crop kendaraan dari rekaman NVR.")
    ap.add_argument("inputs", nargs="+", help="file video atau folder berisi video")
    ap.add_argument("--out", default="dataset_raw")
    ap.add_argument("--model", default="yolo11n_openvino_model", help="detektor kendaraan (sama dengan MODEL_PATH)")
    ap.add_argument("--fk-model", default="", help="detektor forklift (mis. forklift_openvino_model); kosong = lewati forklift")
    ap.add_argument("--zones", default="", help="cam1_zones.json; saring kendaraan di luar area / dalam exclude")
    ap.add_argument("--every", type=float, default=None, help="ambil 1 frame tiap sekian detik video")
    ap.add_argument("--conf", type=float, default=None, help="conf minimal (kendaraan 0.35 / orang 0.30)")
    ap.add_argument("--fk-conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-ratio", type=float, default=0.012, help="luas kotak / luas ROI minimal (sama dengan CAM1_MIN_AREA_RATIO)")
    ap.add_argument("--min-side", type=int, default=None, help="sisi terpendek crop (CARGO_MIN_CROP_SIDE)")
    ap.add_argument("--min-pixels", type=int, default=None, help="piksel crop minimal (CARGO_MIN_CROP_PIXELS)")
    ap.add_argument("--pad", type=float, default=None, help="perlebar crop (pecahan ukuran kotak per sisi)")
    ap.add_argument("--static-gap", type=float, default=None, help="detik: crop di posisi sama dilewati selama jeda ini (anti duplikat)")
    ap.add_argument("--static-dist", type=float, default=None, help="jarak normal dianggap 'posisi sama'")
    ap.add_argument("--max-per-video", type=int, default=0, help="0 = tanpa batas")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--person", action="store_true", help="MODE ORANG: potong crop orang (untuk classifier orang bawa/kosong)")
    ap.add_argument("--stage-only", action="store_true", help="mode orang: hanya orang yang titik kakinya di poligon 'stage'")
    ap.add_argument("--block-min", type=float, default=5.0, help="mode orang: panjang blok menit per subfolder (jadi 'grup' evaluasi)")
    args = ap.parse_args(argv)
    # nilai bawaan tergantung mode
    dflt = dict(every=1.0, conf=0.30, min_side=40, min_pixels=3000, pad=0.15, static_gap=1.5, static_dist=0.015) if args.person \
        else dict(every=5.0, conf=0.35, min_side=70, min_pixels=8000, pad=0.05, static_gap=30.0, static_dist=0.04)
    for k_, v_ in dflt.items():
        if getattr(args, k_) is None:
            setattr(args, k_, v_)

    vids = collect_videos(args.inputs)
    if not vids:
        print("Tidak ada video.")
        return 1
    zones = Zones(args.zones) if args.zones else None
    if veh_det is None:
        veh_det = YoloDet(args.model, args.conf, args.iou, [0] if args.person else [2, 7], args.imgsz)
    if fk_det is None and args.fk_model and not args.person:
        fk_det = YoloDet(args.fk_model, args.fk_conf, args.iou, None, args.imgsz)

    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "manifest.csv")
    new = not os.path.exists(mpath)
    with open(mpath, "a", newline="", encoding="utf-8") as fh:
        mw = csv.writer(fh)
        if new:
            mw.writerow(["video", "detik", "jenis", "kelas_yolo", "conf", "x", "y", "rasio", "lebar", "tinggi", "file"])
        saver = Saver(args.out, mw, args.static_gap, args.static_dist, args.max_per_video)
        total = 0
        for i, v in enumerate(vids, 1):
            print(f"[{i}/{len(vids)}] {v}", flush=True)
            n = process_video(v, veh_det, fk_det, zones, args, saver)
            print(f"  -> {n} crop", flush=True)
            total += n
            fh.flush()
    print(f"Selesai: {total} crop di '{args.out}'. Pilah manual ke kosong / muatan / tertutup / bukan_truk / forklift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())