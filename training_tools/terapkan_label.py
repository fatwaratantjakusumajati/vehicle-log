# terapkan_label.py - salin crop yang sudah dilabeli (label_tambahan.txt) dari person_crops ke data_orang.
#   python terapkan_label.py --src person_crops --dst data_orang --list label_tambahan.txt
# Nama berkas tujuan memakai pola "sesi__berkas" (sama seperti label_crops.py) supaya validasi silang per sesi tetap benar.
# Berkas asli tidak dipindah. Yang sudah ada dilewati. Setelah itu latih ulang:
#   python train_classifier.py --data data_orang --out person_carry_model.pkl

import argparse
import os
import re
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="person_crops")
ap.add_argument("--dst", default="data_orang")
ap.add_argument("--list", default="label_tambahan.txt")
a = ap.parse_args()

n_ok = n_skip = n_miss = 0
for ln in open(a.list, encoding="utf-8"):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    fn, _, lab = ln.partition("\t")
    m = re.match(r"(CAM1_\d{8}_\d{6}_\d{4})_", fn)
    if not m:
        print("lewati (nama tak dikenal):", fn)
        continue
    sesi = m.group(1)
    src = os.path.join(a.src, sesi, fn)
    dst_dir = os.path.join(a.dst, lab)
    dst = os.path.join(dst_dir, f"{sesi}__{fn}")
    if not os.path.isfile(src):
        n_miss += 1
        print("TIDAK ADA:", src)
        continue
    if os.path.isfile(dst):
        n_skip += 1
        continue
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)
    n_ok += 1
print(f"disalin {n_ok}, sudah ada {n_skip}, tidak ditemukan {n_miss}")
for lab in ("orang_bawa", "orang_kosong"):
    d = os.path.join(a.dst, lab)
    if os.path.isdir(d):
        print(f"  {lab}: {len(os.listdir(d))} berkas")