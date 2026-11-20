# export_openvino.py - ubah person_carry_cls.pt menjadi OpenVINO, lalu pastikan hasil prediksinya sama.
# Pakai:  python export_openvino.py            (setelah pelatihan selesai; memakai dataset_cls\val untuk perbandingan)
import glob
import os
import sys

from ultralytics import YOLO

PT = "person_carry_cls.pt"
if not os.path.isfile(PT):
    sys.exit(f"{PT} tidak ada. Selesaikan pelatihan dulu.")
out = YOLO(PT).export(format="openvino", imgsz=224, half=False)      # -> person_carry_cls_openvino_model/
print("hasil ekspor:", out)

a = YOLO(PT)
b = YOLO(out, task="classify")
ia = [k for k, v in a.names.items() if v == "bawa"][0]
ib = [k for k, v in b.names.items() if v == "bawa"][0]
files = sorted(glob.glob(os.path.join("dataset_cls", "val", "*", "*.jpg")))[:60]
if not files:
    sys.exit("dataset_cls\\val kosong; tidak bisa membandingkan.")
mx, flip = 0.0, 0
for f in files:
    pa = float(a.predict(f, imgsz=224, device="cpu", verbose=False)[0].probs.data[ia])
    pb = float(b.predict(f, imgsz=224, verbose=False)[0].probs.data[ib])
    mx = max(mx, abs(pa - pb))
    flip += (pa >= 0.5) != (pb >= 0.5)
print(f"dibandingkan {len(files)} gambar | selisih probabilitas terbesar={mx:.4f} | keputusan berbeda={flip}")
print("AMAN dipakai" if mx < 0.02 and flip == 0 else "PERIKSA: hasil berbeda, jangan dipasang dulu")