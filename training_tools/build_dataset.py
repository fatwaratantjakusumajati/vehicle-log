"""
build_dataset.py - susun dataset/ untuk train_classifier.py dari crop bak yang sudah disimpan cam1.py (folder sampel_bak_v2/<id_sesi>/).
Nama berkas crop: <id_sesi>_<no>_t<detik_sejak_masuk>_p<skor>.jpg  -> dipilah berdasarkan RENTANG DETIK.

  --ambil LABEL:MULAI-AKHIR   salin langsung ke dataset/LABEL/ (dipakai untuk yang pasti, mis. bak kosong sebelum forklift datang)
  --cek   LABEL:MULAI-AKHIR   salin ke dataset/_cek_LABEL/ untuk DIPERIKSA MANUAL; pindahkan yang benar ke dataset/LABEL/, hapus sisanya
  --maks-p-kosong 0.15        crop berlabel 'kosong' dengan skor lama > nilai ini dilewati (jaga-jaga salah label)
  --tiap N                    ambil tiap N-ya (kurangi crop yang nyaris kembar)

CONTOH (pickup 2-act_potong.mp4, sesi dari run terakhir):
  python build_dataset.py --sesi CAM1_20261104_070101_0001 --ambil kosong:80-290 --cek muatan:330-560 --ambil muatan:1-60 --tiap 2
Lihat ringkasan fase tiap sesi lebih dulu:
  python build_dataset.py --sesi CAM1_20261104_070101_0001 --ringkas
"""
import argparse
import os
import re
import shutil
import sys

PAT = re.compile(r"_t(\d+)_p(NA|[0-9.]+)\.(?:jpg|jpeg|png)$", re.I)


def list_crops(src, sesi):
    d = os.path.join(src, sesi)
    if not os.path.isdir(d):
        sys.exit(f"Folder '{d}' tidak ada.")
    out = []
    for f in sorted(os.listdir(d)):
        m = PAT.search(f)
        if m:
            p = None if m.group(2) == "NA" else float(m.group(2).rstrip("."))
            out.append((int(m.group(1)), p, os.path.join(d, f), f))
    return out


def parse(spec):
    m = re.match(r"^([\w\-]+):(\d+)-(\d+)$", spec)
    if not m:
        sys.exit(f"Format salah '{spec}'. Contoh: kosong:80-290")
    return m.group(1), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="sampel_bak_v2")
    ap.add_argument("--dst", default="dataset")
    ap.add_argument("--sesi", required=True, action="append", help="id sesi (boleh diulang)")
    ap.add_argument("--ambil", action="append", default=[])
    ap.add_argument("--cek", action="append", default=[])
    ap.add_argument("--maks-p-kosong", type=float, default=0.15)
    ap.add_argument("--tiap", type=int, default=1)
    ap.add_argument("--ringkas", action="store_true")
    a = ap.parse_args()

    for sesi in a.sesi:
        crops = list_crops(a.src, sesi)
        if not crops:
            print(f"[{sesi}] tidak ada crop bernama *_t<detik>_p<skor>.jpg")
            continue
        if a.ringkas:
            print(f"[{sesi}] {len(crops)} crop. Skor rata-rata per 30 dtk:")
            bins = {}
            for t, p, _, _ in crops:
                if p is not None:
                    bins.setdefault(t // 30, []).append(p)
            for b in sorted(bins):
                v = bins[b]
                print(f"   {b * 30:>4}-{b * 30 + 29:<4} dtk  n={len(v):>3}  rata2={sum(v) / len(v):.2f}  maks={max(v):.2f}")
            continue
        jobs = [(s, False) for s in a.ambil] + [(s, True) for s in a.cek]
        for spec, review in jobs:
            label, lo, hi = parse(spec)
            dst = os.path.join(a.dst, ("_cek_" if review else "") + label)
            os.makedirs(dst, exist_ok=True)
            pick = [c for c in crops if lo <= c[0] <= hi]
            pick = pick[::max(1, a.tiap)]
            n = skip = 0
            for t, p, path, fname in pick:
                if label == "kosong" and p is not None and p > a.maks_p_kosong:
                    skip += 1
                    continue
                shutil.copy2(path, os.path.join(dst, fname))
                n += 1
            print(f"[{sesi}] {label} {lo}-{hi} dtk -> {dst}: {n} gambar" + (f" ({skip} dilewati: skor > {a.maks_p_kosong})" if skip else ""))


if __name__ == "__main__":
    main()