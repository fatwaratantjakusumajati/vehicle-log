"""
build_forklift_reference.py
============================
Script bantu untuk bikin/nambah referensi forklift buat forklift_filter.py.

CARA PAKAI:
1. Siapkan folder isi crop forklift saja (bukan frame penuh) -- misal
   hasil crop manual dari beberapa screenshot debug_snapshots/ kamu,
   di kondisi/sudut/pencahayaan yang beda-beda (lagi diam, lagi jalan,
   kena bayangan, dari sudut agak lain). Minimal 5-10 crop supaya
   fingerprint-nya tidak terlalu sempit ke 1 kondisi pencahayaan saja.
   Format .jpg / .png, nama file bebas.

2. Jalankan:
       python build_forklift_reference.py contoh_forklift/

   Atau tentukan nama file output kalau mau beda dari default:
       python build_forklift_reference.py contoh_forklift/ forklift_reference.pkl

3. Taruh forklift_reference.pkl yang dihasilkan di folder yang sama
   dengan aw.py (atau set path-nya lewat FORKLIFT_REFERENCE_PATH di
   .env kalau mau taruh di lokasi lain).

4. Jalankan ulang aw.py -- forklift_filter otomatis aktif begitu file
   ini ketemu (lihat log "[forklift_filter] N referensi dimuat...").

Mau nambah referensi lagi belakangan (misal ternyata masih ada yang
lolos)? Tinggal ulangi langkah 1-2 dengan folder baru berisi crop
tambahan -- script ini akan GABUNG ke referensi lama yang sudah ada
di file output (bukan menimpa), asal path outputnya sama.
"""

import glob
import os
import sys

import cv2

import forklift_filter


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else forklift_filter.FORKLIFT_REFERENCE_PATH

    if not os.path.isdir(folder):
        print(f"[ERROR] Folder tidak ditemukan: '{folder}'")
        sys.exit(1)

    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
        + glob.glob(os.path.join(folder, "*.png"))
    )
    if not paths:
        print(f"[ERROR] Tidak ada file .jpg/.jpeg/.png di '{folder}'")
        sys.exit(1)

    # Gabung ke referensi lama kalau file output sudah ada.
    existing_sigs = []
    if os.path.isfile(out_path):
        import pickle
        try:
            with open(out_path, "rb") as f:
                existing_sigs = pickle.load(f)
            print(f"[INFO] {len(existing_sigs)} referensi lama ditemukan di "
                  f"'{out_path}', akan digabung dengan yang baru.")
        except Exception as e:
            print(f"[WARNING] Gagal baca referensi lama ({e}), mulai dari kosong.")

    new_sigs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [SKIP] Gagal baca gambar: {p}")
            continue
        sig = forklift_filter.add_reference(img)
        if sig is None:
            print(f"  [SKIP] Crop terlalu kecil (<5000px) atau gagal diproses: {p}")
            continue
        new_sigs.append(sig)
        print(f"  [OK]   {p}")

    if not new_sigs:
        print("\n[ERROR] Tidak ada referensi baru yang valid.")
        sys.exit(1)

    all_sigs = existing_sigs + new_sigs
    forklift_filter.save_references(all_sigs, out_path)
    print(f"\nSelesai. Total {len(all_sigs)} referensi "
          f"({len(existing_sigs)} lama + {len(new_sigs)} baru) tersimpan ke '{out_path}'.")
    print("Copy file .pkl ini ke folder yang sama dengan aw.py, lalu jalankan ulang aw.py.")


if __name__ == "__main__":
    main()