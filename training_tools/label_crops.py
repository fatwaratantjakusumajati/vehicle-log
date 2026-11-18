# label_crops.py - pelabelan cepat crop orang: tekan tombol, gambar disalin ke folder kelasnya.
#   python label_crops.py --src person_crops --dst data_orang --n 600
#   python label_crops.py --src person_crops --dst data_orang --n 300 --rank-model person_carry_model.pkl --band 0.10 0.60 --exclude CAM1_20261106_152705 CAM1_20261107_071035
#     --band LO HI : (butuh --rank-model) tampilkan crop acak yang skor 'bawa'-nya di antara LO..HI = kasus yang model BELUM yakin (paling berguna untuk dilabeli)
#     --exclude    : lewati crop yang path-nya mengandung salah satu teks ini (mis. sesi klip uji, supaya tidak bocor ke data latih)
# Tombol:  b = orang_bawa (membawa dus/barang)   k = orang_kosong (tangan kosong / berdiri / jalan)
#          s = lewati (ragu / terhalang)         u = batalkan label terakhir        q = selesai
# Gambar diambil acak dari semua sub-folder sesi di --src (seed tetap), jadi bisa dilanjutkan: yang sudah dilabeli dilewati.
# File asli TIDAK dipindah/dihapus; hanya disalin ke <dst>/orang_bawa dan <dst>/orang_kosong.
# Target awal: >= 100 gambar per kelas dari beberapa sesi berbeda. Setelah itu:
#   python train_classifier.py --data data_orang --out person_carry_model.pkl

import argparse
import os
import random
import shutil

import cv2
import numpy as np

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
KEYS = {ord("b"): "orang_bawa", ord("k"): "orang_kosong"}


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def rank_by_model(model_path, paths, cache_path, with_scores=False):
    """Urutkan paths dari yang paling mungkin 'orang_bawa'. Skor di-cache supaya jalan berikutnya cepat."""
    import pickle
    from feature_extractor import extract_features
    with open(model_path, "rb") as f:
        model = pickle.load(f)["model"]
    classes = [str(c) for c in model.classes_]
    if "orang_bawa" not in classes:
        raise SystemExit(f"model {model_path} tidak punya kelas 'orang_bawa' (kelas: {classes})")
    j = classes.index("orang_bawa")
    cache = {}
    if os.path.isfile(cache_path):
        for ln in open(cache_path, encoding="utf-8"):
            p, _, s = ln.rstrip("\n").partition("\t")
            try:
                cache[p] = float(s)
            except ValueError:
                pass
    need = [p for p in paths if p not in cache]
    print(f"menghitung skor {len(need)} gambar (sisanya dari cache {len(paths) - len(need)})")
    with open(cache_path, "a", encoding="utf-8") as cf:
        for k, p in enumerate(need, 1):
            im = read_image(p)
            s = -1.0
            if im is not None and im.size:
                s = float(model.predict_proba(extract_features(im).reshape(1, -1))[0][j])
            cache[p] = s
            cf.write(f"{p}\t{s:.4f}\n")
            if k % 500 == 0:
                print(f"  skor {k}/{len(need)}")
    ranked = sorted(paths, key=lambda p: -cache.get(p, -1.0))
    return (ranked, cache) if with_scores else ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="person_crops")
    ap.add_argument("--dst", default="data_orang")
    ap.add_argument("--n", type=int, default=600, help="jumlah gambar acak yang ditampilkan")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--band", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="dengan --rank-model: hanya crop yang skor 'orang_bawa'-nya di antara LO dan HI (kasus yang model belum yakin), diacak")
    ap.add_argument("--exclude", nargs="*", default=[], help="lewati crop yang path-nya mengandung salah satu teks ini (mis. sesi klip uji)")
    ap.add_argument("--rank-model", default="", help="model awal (.pkl dari train_classifier.py): tampilkan dulu crop yang paling mirip 'orang_bawa'")
    a = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(a.src):
        files += [os.path.join(dp, f) for f in fns if f.lower().endswith(EXT)]
    files.sort()
    if a.exclude:
        n0 = len(files)
        files = [f for f in files if not any(x in f for x in a.exclude)]
        print(f"--exclude: {n0 - len(files)} crop dilewati, {len(files)} tersisa")
    all_files = files

    for lab in KEYS.values():
        os.makedirs(os.path.join(a.dst, lab), exist_ok=True)
    log_path = os.path.join(a.dst, "_dilabeli.txt")
    done = {}
    if os.path.isfile(log_path):
        for ln in open(log_path, encoding="utf-8"):
            p, _, lab = ln.rstrip("\n").partition("\t")
            done[p] = lab
    if a.band and not a.rank_model:
        raise SystemExit("--band butuh --rank-model")
    if a.rank_model and a.band:
        ranked, sc = rank_by_model(a.rank_model, [f for f in all_files if f not in done],
                                   os.path.join(a.dst, "_skor_cache.tsv"), with_scores=True)
        lo, hi = a.band
        files = [f for f in ranked if lo <= sc.get(f, -1.0) <= hi]
        print(f"--band {lo}..{hi}: {len(files)} crop di rentang itu")
        random.Random(a.seed).shuffle(files)
        files = files[:a.n]
    elif a.rank_model:
        files = rank_by_model(a.rank_model, [f for f in all_files if f not in done], os.path.join(a.dst, "_skor_cache.tsv"))[:a.n]
    else:
        random.Random(a.seed).shuffle(all_files)
        files = all_files[:a.n]
    todo = [f for f in files if f not in done]
    print(f"{len(files)} gambar dipilih, {len(done)} sudah dilabeli, {len(todo)} tersisa")

    def rewrite_log():
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(f"{k}\t{v}\n" for k, v in done.items())

    def tag_of(p):
        return os.path.basename(os.path.dirname(p)) + "__" + os.path.basename(p)   # nama unik: sesi__berkas

    history = []   # tiap aksi (label atau lewati): (path, label)
    i = 0
    while i < len(todo):
        p = todo[i]
        im = read_image(p)
        if im is None:
            i += 1
            continue
        h = 560
        view = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_CUBIC)
        n_b = len(os.listdir(os.path.join(a.dst, "orang_bawa")))
        n_k = len(os.listdir(os.path.join(a.dst, "orang_kosong")))
        cv2.putText(view, f"bawa={n_b} kosong={n_k} | {i + 1}/{len(todo)} | b/k/s/u/q", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("label_crops", view)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u"):
            if history:
                lp, lab = history.pop()
                if lab in KEYS.values():
                    f_ = os.path.join(a.dst, lab, tag_of(lp))
                    if os.path.isfile(f_):
                        os.remove(f_)
                done.pop(lp, None)
                rewrite_log()
                i -= 1
            continue
        if key in KEYS:
            lab = KEYS[key]
            shutil.copy2(p, os.path.join(a.dst, lab, tag_of(p)))
        elif key == ord("s"):
            lab = "lewati"
        else:
            continue
        done[p] = lab
        history.append((p, lab))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{p}\t{lab}\n")
        i += 1
    cv2.destroyAllWindows()
    print("selesai. bawa:", len(os.listdir(os.path.join(a.dst, "orang_bawa"))),
          "| kosong:", len(os.listdir(os.path.join(a.dst, "orang_kosong"))))


if __name__ == "__main__":
    main()