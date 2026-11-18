"""Latih detektor forklift (satu kelas) dari dataset Roboflow (format YOLOv8) + gambar negatif dermaga.

Pakai:
  pip install ultralytics
  python latih_forklift.py --data "C:\\data\\ForkliftProject" --neg "C:\\data\\negatif" --own "C:\\data\\dermaga_berlabel" --weights bobot_lama.pt --epochs 60

--data     folder hasil unduhan Roboflow (berisi train/ valid/ test/ dan data.yaml)
--neg      (opsional) folder berisi gambar JPG/PNG truk, container, van dari dermaga TANPA forklift (jadi contoh "bukan forklift")
--own      (opsional) folder berisi gambar dermaga berlabel YOLO (images/ + labels/) yang ada forkliftnya
--weights  (opsional) bobot forklift lama (.pt) untuk di-fine-tune; kosong = mulai dari yolo11n.pt
Semua kelas di dataset dijadikan satu kelas: forklift (id 0).
Hasil: runs/forklift/weights/best.pt, lalu diekspor ke OpenVINO (folder best_openvino_model).
"""
import argparse, shutil, random, glob, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--neg", default="")
ap.add_argument("--own", default="")
ap.add_argument("--weights", default="yolo11n.pt")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--out", default="dataset_gabungan")
a = ap.parse_args()

out = Path(a.out)
if out.exists():
    shutil.rmtree(out)
for sp in ("train", "val"):
    (out / "images" / sp).mkdir(parents=True)
    (out / "labels" / sp).mkdir(parents=True)

IMG = (".jpg", ".jpeg", ".png", ".bmp")

def add(img, lab_lines, split, tag):
    name = f"{tag}_{img.stem}"
    shutil.copy(img, out / "images" / split / (name + img.suffix.lower()))
    (out / "labels" / split / (name + ".txt")).write_text("\n".join(lab_lines) + ("\n" if lab_lines else ""))

def remap(lab_path):
    """semua kelas -> 0"""
    if not lab_path.exists():
        return []
    res = []
    for ln in lab_path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            res.append("0 " + " ".join(p[1:]))
    return res

n = {"roboflow": 0, "neg": 0, "own": 0}
root = Path(a.data)
for sp_src, sp in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "train")):
    d = root / sp_src / "images"
    if not d.exists():
        continue
    for img in d.iterdir():
        if img.suffix.lower() in IMG:
            add(img, remap(root / sp_src / "labels" / (img.stem + ".txt")), sp, "rf"); n["roboflow"] += 1

if a.neg:
    for img in Path(a.neg).iterdir():
        if img.suffix.lower() in IMG:
            add(img, [], "val" if random.random() < 0.15 else "train", "neg"); n["neg"] += 1

if a.own:
    o = Path(a.own)
    for img in (o / "images").iterdir():
        if img.suffix.lower() in IMG:
            for _ in range(3):   # diulang 3x agar contoh dermaga tidak tenggelam oleh dataset publik
                add(img, remap(o / "labels" / (img.stem + ".txt")), "train", f"own{_}"); 
            n["own"] += 1

print("jumlah gambar:", n)
(Path("data_forklift.yaml")).write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: forklift\n")

from ultralytics import YOLO
m = YOLO(a.weights)
m.train(data="data_forklift.yaml", epochs=a.epochs, imgsz=a.imgsz, project="runs", name="forklift", exist_ok=True,
        hsv_h=0.25, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0, patience=20)   # hsv_h besar: warna forklift jangan jadi petunjuk
best = YOLO("runs/forklift/weights/best.pt")
best.export(format="openvino", imgsz=a.imgsz)
print("SELESAI. Bobot: runs/forklift/weights/best.pt | OpenVINO: runs/forklift/weights/best_openvino_model")