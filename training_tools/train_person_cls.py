# train_person_cls.py - melatih pengklasifikasi "orang bawa barang / kosong" dengan fine-tune YOLO11-cls (bukan fitur beku + logistic).
# Pakai:  python train_person_cls.py --raw dataset_raw --epochs 40
# Hasil:  person_carry_cls.pt (+ laporan akurasi pada sesi yang TIDAK dipakai melatih).
import argparse
import collections
import os
import random
import shutil

CLASSES = ("bawa", "kosong")


def session_of(name):
    p = name.split("_")
    return "_".join(p[:4]) if len(p) >= 4 else p[0]


def split_files(raw, val_frac, seed=7):
    """Bagi menurut SESI (bukan acak per gambar) supaya uji jujur. Bila sesi < 3, bagi per blok waktu."""
    by_cls = {c: sorted(os.listdir(os.path.join(raw, c))) for c in CLASSES}
    sessions = sorted({session_of(f) for c in CLASSES for f in by_cls[c]})
    train, val = [], []
    if len(sessions) >= 3:
        rnd = random.Random(seed)
        rnd.shuffle(sessions)
        total = sum(len(v) for v in by_cls.values())
        cnt = collections.Counter(session_of(f) for c in CLASSES for f in by_cls[c])
        val_s, acc = set(), 0
        for s in sessions:
            if acc >= val_frac * total:
                break
            val_s.add(s)
            acc += cnt[s]
        for c in CLASSES:
            for f in by_cls[c]:
                (val if session_of(f) in val_s else train).append((c, f))
        print(f"sesi uji: {sorted(val_s)}")
    else:
        print("PERINGATAN: sesi < 3, uji dibagi per blok waktu (kurang ketat). Tambahkan data dari lebih banyak sesi.")
        for c in CLASSES:
            fs = by_cls[c]
            k = max(1, int(len(fs) * (1 - val_frac)))
            train += [(c, f) for f in fs[:k]]
            val += [(c, f) for f in fs[k:]]
    return train, val


def build(raw, out, train, val, balance):
    if os.path.isdir(out):
        shutil.rmtree(out)
    for sp in ("train", "val"):
        for c in CLASSES:
            os.makedirs(os.path.join(out, sp, c), exist_ok=True)
    for sp, lst in (("train", train), ("val", val)):
        for c, f in lst:
            shutil.copy2(os.path.join(raw, c, f), os.path.join(out, sp, c, f))
    if balance:
        n = {c: len(os.listdir(os.path.join(out, "train", c))) for c in CLASSES}
        lo, hi = min(n, key=n.get), max(n, key=n.get)
        if n[lo] > 0 and n[hi] > n[lo]:
            src = sorted(os.listdir(os.path.join(out, "train", lo)))
            for j in range(n[hi] - n[lo]):
                f = src[j % len(src)]
                shutil.copy2(os.path.join(out, "train", lo, f), os.path.join(out, "train", lo, f"dup{j}_{f}"))
            print(f"seimbangkan: kelas '{lo}' diperbanyak {n[lo]} -> {n[hi]} (salinan; augmentasi membuatnya berbeda)")
    return {sp: {c: len(os.listdir(os.path.join(out, sp, c))) for c in CLASSES} for sp in ("train", "val")}


def save_errors(errors, out="kesalahan", cols=6, h=240):
    """Simpan gambar yang salah tebak + satu lembar gabungan kesalahan.jpg (kirim lembar ini untuk dianalisis)."""
    import cv2
    import numpy as np
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    tiles = []
    for c, p, path in errors:
        img = cv2.imread(path)
        if img is None:
            continue
        shutil.copy2(path, os.path.join(out, f"asli-{c}_p{p:.2f}_{os.path.basename(path)}"))
        w = max(80, int(img.shape[1] * h / img.shape[0]))
        t = cv2.resize(img, (w, h))
        cv2.rectangle(t, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(t, f"asli:{c} P(bawa)={p:.2f}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(t)
    if not tiles:
        print("tidak ada kesalahan untuk disimpan")
        return
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        width = sum(t.shape[1] for t in row)
        rows.append(np.hstack(row + [np.zeros((h, max(0, 1500 - width), 3), np.uint8)]) if width < 1500 else np.hstack(row))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((h, wmax - r.shape[1], 3), np.uint8)]) if r.shape[1] < wmax else r for r in rows]
    cv2.imwrite("kesalahan.jpg", np.vstack(rows))
    print(f"{len(tiles)} gambar salah tebak disimpan di folder '{out}' dan lembar gabungan kesalahan.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset_raw")
    ap.add_argument("--work", default="dataset_cls")
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=25, help="berhenti bila tak membaik selama N epoch")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cpu", help="cpu, atau 0 untuk GPU (Kaggle/Colab)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--eval-only", default="", help="lewati pelatihan; hanya uji bobot yang sudah ada, mis. runs\\classify\\runs_person\\cls\\weights\\best.pt")
    a = ap.parse_args()

    train, val = split_files(a.raw, a.val_frac)
    counts = build(a.raw, a.work, train, val, not a.no_balance)
    print("jumlah:", counts)
    if min(counts["train"].values()) < 30 or min(counts["val"].values()) < 10:
        print("PERINGATAN: data masih sedikit (disarankan >= 100 per kelas untuk latih, >= 30 per kelas untuk uji). Hasil belum bisa dipercaya.")

    from ultralytics import YOLO
    if a.eval_only:
        if os.path.abspath(a.eval_only) != os.path.abspath("person_carry_cls.pt"):
            shutil.copy2(a.eval_only, "person_carry_cls.pt")
    else:
        m = YOLO(a.model)
        m.train(data=os.path.abspath(a.work), epochs=a.epochs, imgsz=a.imgsz, batch=32, device=a.device, patience=a.patience, workers=a.workers,
                project="runs_person", name="cls", exist_ok=True, fliplr=0.5, degrees=5, scale=0.3, translate=0.1,
                hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, erasing=0.2, verbose=False)
        best = os.path.join(str(m.trainer.save_dir), "weights", "best.pt")   # lokasi bergantung versi ultralytics
        print("bobot terbaik:", best)
        shutil.copy2(best, "person_carry_cls.pt")

    # laporan jujur pada sesi uji
    m = YOLO("person_carry_cls.pt")
    names = m.names
    ib = [k for k, v in names.items() if v == "bawa"][0]
    tp = fp = tn = fn = 0
    probs_b = []
    errors = []
    for c in CLASSES:
        d = os.path.join(a.work, "val", c)
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))]
        for i in range(0, len(files), 32):
            for j, r in enumerate(m.predict(files[i:i + 32], imgsz=a.imgsz, device=a.device, verbose=False)):
                p = float(r.probs.data[ib])
                probs_b.append((c, p))
                pred = p >= 0.5
                if pred != (c == "bawa"):
                    errors.append((c, p, files[i + j]))
                if c == "bawa":
                    tp += pred; fn += (not pred)
                else:
                    fp += pred; tn += (not pred)
    n = max(1, tp + fp + tn + fn)
    print("\n=== HASIL PADA SESI UJI (tidak dipakai melatih) ===")
    print(f"akurasi={(tp + tn) / n:.1%} | bawa terdeteksi (recall)={tp / max(1, tp + fn):.1%} | "
          f"tebakan 'bawa' yang benar (precision)={tp / max(1, tp + fp):.1%}")
    print(f"matriks: bawa->bawa={tp} bawa->kosong={fn} kosong->bawa={fp} kosong->kosong={tn}")
    for th in (0.6, 0.7, 0.8, 0.9):
        t = sum(1 for c, p in probs_b if c == "bawa" and p >= th)
        f_ = sum(1 for c, p in probs_b if c == "kosong" and p >= th)
        print(f"  ambang {th}: bawa lolos={t}  kosong salah lolos={f_}")
    print(f"jumlah salah tebak pada ambang 0.5: {len(errors)}")
    save_errors(errors)
    print("Pasang ke sistem HANYA bila precision >= 90% dan data uji >= 30 per kelas.")


if __name__ == "__main__":
    main()