"""
train_carry.py - latih classifier "orang membawa barang" (orang_bawa) vs "orang_kosong".

  python train_carry.py --data dataset_person --out orang_carry_model.pkl

Struktur data (hasil sortir_crop.py):
  dataset_person/orang_bawa/<video>_blkNN/*.jpg
  dataset_person/orang_kosong/<video>_blkNN/*.jpg

- Memakai extract_features dari feature_extractor.py (backbone sama dengan cargo_classifier).
- Augmentasi (hanya data latih): flip, kecerahan, blur, resolusi turun, kompresi JPEG, geser/zoom.
  Kelas minoritas (bawa) diaugmentasi lebih banyak, meniru frame CCTV yang jelek.
- Evaluasi leave-one-video-out (grup = nama video, tanpa _blkNN), jadi tidak bocor antar frame mirip.
- Hasil: bundle pickle {"model": Pipeline, "positive": "orang_bawa", "threshold": ...}
"""
import argparse
import os
import pickle
import random
import re

import cv2
import numpy as np

from feature_extractor import extract_features

EXT = (".jpg", ".jpeg", ".png", ".bmp")
POS, NEG = "orang_bawa", "orang_kosong"


def load_dataset(root):
    items = []  # (path, label, video)
    for label in (POS, NEG):
        base = os.path.join(root, label)
        for dp, _, files in os.walk(base):
            for f in sorted(files):
                if f.lower().endswith(EXT):
                    grp = os.path.basename(dp)
                    video = re.sub(r"_blk\d+$", "", grp)
                    items.append((os.path.join(dp, f), label, video))
    return items


# ---------------------------------------------------------------- augmentasi
def aug_flip(im):
    return cv2.flip(im, 1)


def aug_bright(im):
    a = random.uniform(0.75, 1.25)
    b = random.uniform(-20, 20)
    return cv2.convertScaleAbs(im, alpha=a, beta=b)


def aug_blur(im):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(im, (k, k), 0)


def aug_lowres(im):
    h, w = im.shape[:2]
    s = random.uniform(0.3, 0.6)
    sm = cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def aug_jpeg(im):
    q = random.randint(12, 35)
    ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else im


def aug_shift_zoom(im):
    h, w = im.shape[:2]
    z = random.uniform(1.0, 1.2)
    nh, nw = int(h / z), int(w / z)
    y0 = random.randint(0, h - nh)
    x0 = random.randint(0, w - nw)
    return cv2.resize(im[y0:y0 + nh, x0:x0 + nw], (w, h))


def aug_bad_frame(im):
    """Kombinasi 'frame jelek': low-res + blur + JPEG (+ kecerahan)."""
    im = aug_lowres(im)
    if random.random() < 0.6:
        im = aug_blur(im)
    im = aug_jpeg(im)
    if random.random() < 0.5:
        im = aug_bright(im)
    return im


SINGLE = [aug_flip, aug_bright, aug_blur, aug_lowres, aug_jpeg, aug_shift_zoom]


def make_variants(im, n):
    out = []
    for i in range(n):
        v = aug_bad_frame(im) if i % 2 == 0 else random.choice(SINGLE)(im)
        if random.random() < 0.5:
            v = aug_flip(v)
        out.append(v)
    return out


def feat(im):
    return np.asarray(extract_features(im), dtype=np.float32).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_person")
    ap.add_argument("--out", default="orang_carry_model.pkl")
    ap.add_argument("--aug-pos", type=int, default=8, help="jumlah varian per crop bawa")
    ap.add_argument("--aug-neg", type=int, default=1, help="jumlah varian per crop kosong")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    items = load_dataset(args.data)
    if not items:
        print("Tidak ada gambar di", args.data)
        return 1
    n_pos = sum(1 for _, l, _ in items if l == POS)
    print(f"{len(items)} crop ({POS}={n_pos}, {NEG}={len(items) - n_pos}), "
          f"video={sorted({v for _, _, v in items})}")

    # ekstrak fitur sekali per crop: asli + varian
    rows = []  # (feat, label, video, is_aug)
    for k, (p, label, video) in enumerate(items, 1):
        im = cv2.imread(p)
        if im is None:
            continue
        rows.append((feat(im), label, video, False))
        n = args.aug_pos if label == POS else args.aug_neg
        for v in make_variants(im, n):
            rows.append((feat(v), label, video, True))
        if k % 20 == 0:
            print(f"  fitur {k}/{len(items)}", flush=True)

    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    vid = np.array([r[2] for r in rows])
    aug = np.array([r[3] for r in rows])

    def new_model():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))

    # ---- evaluasi leave-one-video-out (uji hanya pada crop asli)
    videos = sorted(set(vid))
    all_true, all_prob = [], []
    if len(videos) >= 2:
        for v in videos:
            tr = vid != v
            te = (vid == v) & (~aug)
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            m = new_model().fit(X[tr], y[tr])
            pr = m.predict_proba(X[te])[:, list(m.classes_).index(POS)]
            all_true += list(y[te] == POS)
            all_prob += list(pr)
            pred = np.where(pr >= 0.5, POS, NEG)
            acc = (pred == y[te]).mean()
            rec = ((pred == POS) & (y[te] == POS)).sum() / max(1, (y[te] == POS).sum())
            print(f"  uji video '{v}': akurasi={acc:.2f}  recall_bawa={rec:.2f}  (n={te.sum()})")
        t, pr = np.array(all_true), np.array(all_prob)
        print("\nGabungan leave-one-video-out (ambang 0.5):")
        print(classification_report(t, pr >= 0.5, target_names=[NEG, POS], zero_division=0))
        best_thr, best_f = 0.5, -1
        for thr in np.arange(0.2, 0.91, 0.05):
            p_ = pr >= thr
            tp = (p_ & t).sum()
            prec = tp / max(1, p_.sum())
            rec = tp / max(1, t.sum())
            f = 2 * prec * rec / max(1e-9, prec + rec)
            if f > best_f:
                best_thr, best_f = float(thr), f
        print(f"Ambang disarankan (F1 terbaik): {best_thr:.2f} (F1={best_f:.2f})")
    else:
        best_thr = 0.5
        print("Hanya 1 video -> evaluasi per video dilewati.")

    # ---- latih akhir pada semua data
    model = new_model().fit(X, y)
    with open(args.out, "wb") as f:
        pickle.dump({"model": model, "positive": POS, "negative": NEG, "threshold": best_thr}, f)
    print(f"\nModel disimpan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())