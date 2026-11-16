# train_classifier.py - latih classifier ringan di atas fitur MobileNetV2 (feature_extractor.py).
# Satu skrip untuk semua model: model muatan (cargo_model.pkl), forklift membawa barang (forklift_carry_model.pkl),
# dan orang membawa barang (person_carry_model.pkl). Nama kelas = nama sub-folder; formatnya sama dengan model lama:
#   {"model": Pipeline(StandardScaler + LogisticRegression), "labels": [...], "C": ...}
#
# Struktur folder data:
#   data_muatan/kosong/*.jpg   data_muatan/muatan/*.jpg   data_muatan/tertutup/*.jpg   (opsional: forklift, bukan_truk, muatan_terpal)
#   data_orang/orang_bawa/*.jpg   data_orang/orang_kosong/*.jpg
#
# Contoh:
#   python train_classifier.py --data data_muatan --out cargo_model.pkl
#   python train_classifier.py --data data_orang  --out person_carry_model.pkl
# Validasi silang dibagi PER SESI (nama berkas "sesi__berkas" dari label_crops.py), jadi crop yang hampir kembar dari satu sesi tidak
# bocor ke fold uji; angkanya lebih jujur. Berkas tanpa "__" dianggap satu sesi sendiri-sendiri (perilaku lama).
# Folder berawalan "_" diabaikan. Model lama dicadangkan ke <out>.bak.

import argparse
import os
import pickle
import shutil
import sys

import cv2
import numpy as np

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)          # aman untuk path Windows yang mengandung spasi/Unicode
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def collect(root):
    labels = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
    items = []
    for lab in labels:
        for dp, _, fns in os.walk(os.path.join(root, lab)):
            for fn in sorted(fns):
                if fn.lower().endswith(EXT):
                    items.append((os.path.join(dp, fn), lab))
    return labels, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder berisi sub-folder per kelas")
    ap.add_argument("--out", required=True, help="berkas keluaran .pkl")
    ap.add_argument("--C", type=float, default=0.3, help="kekuatan regularisasi LogisticRegression (lebih kecil = lebih halus)")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    from feature_extractor import extract_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels, items = collect(a.data)
    if len(labels) < 2:
        sys.exit(f"butuh minimal 2 sub-folder kelas di '{a.data}' (ada: {labels})")
    if len(labels) > 12:
        sys.exit(f"'{a.data}' berisi {len(labels)} sub-folder, terlalu banyak untuk kelas. Sub-folder harus nama LABEL "
                 f"(mis. orang_bawa, orang_kosong), bukan folder sesi. Contoh nama yang ditemukan: {labels[:3]}. "
                 f"Pisahkan gambar ke folder label dulu (lihat label_crops.py).")
    counts = {l: sum(1 for _, x in items if x == l) for l in labels}
    print("jumlah gambar per kelas:", counts)
    for l, n in counts.items():
        if n < 20:
            print(f"  PERINGATAN: kelas '{l}' hanya {n} gambar (disarankan >= 50)")

    X, y, grp, bad = [], [], [], 0
    for i, (p, lab) in enumerate(items, 1):
        im = read_image(p)
        if im is None or im.size == 0:
            bad += 1
            continue
        X.append(extract_features(im))
        y.append(lab)
        bn = os.path.basename(p)
        grp.append(bn.split("__")[0] if "__" in bn else p)
        if i % 50 == 0:
            print(f"  fitur {i}/{len(items)}")
    if bad:
        print(f"{bad} gambar tidak terbaca, dilewati")
    X, y, grp = np.vstack(X), np.array(y), np.array(grp)

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced"))
    k = min(a.folds, min(counts.values()))
    n_grp = len(set(grp.tolist()))
    if k >= 2:
        from sklearn.model_selection import StratifiedGroupKFold
        kg = min(k, n_grp)
        pred = None
        if kg >= 2:
            try:
                pred = cross_val_predict(clf, X, y, groups=grp, cv=StratifiedGroupKFold(kg, shuffle=True, random_state=0))
                print(f"\nvalidasi silang PER SESI {kg}-fold ({n_grp} kelompok sesi): akurasi {np.mean(pred == y):.3f}")
            except ValueError as ex:
                print(f"\nvalidasi per sesi gagal ({ex}); pakai validasi biasa")
        if pred is None:
            pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0))
            print(f"\nvalidasi silang {k}-fold (BIASA, kemungkinan terlalu optimis): akurasi {np.mean(pred == y):.3f}")
        if n_grp < 4:
            print(f"  PERINGATAN: hanya {n_grp} sesi berbeda; angka validasi belum bisa dipercaya")
        print("urutan kelas:", labels)
        print(confusion_matrix(y, pred, labels=labels))
    else:
        print("data terlalu sedikit untuk validasi silang")

    clf.fit(X, y)
    if os.path.isfile(a.out):
        shutil.copy2(a.out, a.out + ".bak")
        print(f"model lama dicadangkan ke {a.out}.bak")
    with open(a.out, "wb") as f:
        pickle.dump({"model": clf, "labels": [str(c) for c in clf.classes_], "C": a.C}, f)
    print(f"model disimpan: {a.out} | kelas: {[str(c) for c in clf.classes_]}")


if __name__ == "__main__":
    main()