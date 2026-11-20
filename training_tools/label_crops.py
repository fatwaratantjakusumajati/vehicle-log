# label_crops.py - beri label crop orang dengan tombol keyboard (bahan melatih model "orang membawa barang").
# Pakai:  python label_crops.py --src person_crops --out dataset_raw
# Tombol: b = BAWA barang (dus/karung/palet/apa saja di tangan atau lengan)
#         k = KOSONG (tangan tidak memegang apa pun)
#         s = LEWATI (ragu / blur / tertutup / terpotong)    u = batalkan label terakhir    q = selesai
# Bisa dihentikan kapan saja dan dilanjutkan: yang sudah berlabel tidak muncul lagi.
import argparse
import os
import shutil

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="person_crops")
    ap.add_argument("--out", default="dataset_raw")
    ap.add_argument("--step", type=int, default=2, help="ambil 1 dari tiap N crop berurutan (crop berurutan hampir kembar)")
    ap.add_argument("--min-px", type=int, default=50, help="lewati crop yang sisi terpendeknya < ini")
    a = ap.parse_args()

    for c in ("bawa", "kosong"):
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
    skip_file = os.path.join(a.out, "dilewati.txt")
    done = set()
    for c in ("bawa", "kosong"):
        done |= set(os.listdir(os.path.join(a.out, c)))
    if os.path.isfile(skip_file):
        with open(skip_file, encoding="utf-8") as f:
            done |= {ln.strip() for ln in f if ln.strip()}

    items = []
    for sess in sorted(os.listdir(a.src)):
        d = os.path.join(a.src, sess)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg") and "_f" not in f.split("_")[-3:-1])
        for i, f in enumerate(files):
            if i % max(1, a.step) == 0 and f not in done:
                items.append((sess, os.path.join(d, f), f))
    print(f"{len(items)} crop untuk diberi label. Tombol: b=bawa  k=kosong  s=lewati  u=batal  q=selesai")

    history = []
    n_b = len(os.listdir(os.path.join(a.out, "bawa")))
    n_k = len(os.listdir(os.path.join(a.out, "kosong")))
    i = 0
    while i < len(items):
        sess, path, name = items[i]
        img = cv2.imread(path)
        if img is None or min(img.shape[:2]) < a.min_px:
            i += 1
            continue
        h = 640
        s = h / img.shape[0]
        vis = cv2.resize(img, (max(200, int(img.shape[1] * s)), h))
        cv2.putText(vis, f"{i + 1}/{len(items)}  bawa={n_b} kosong={n_k}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, "b=BAWA  k=KOSONG  s=lewati  u=batal  q=selesai", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("label", vis)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("u") and history:
            kind, nm = history.pop()
            if kind in ("bawa", "kosong"):
                os.remove(os.path.join(a.out, kind, nm))
                n_b -= kind == "bawa"
                n_k -= kind == "kosong"
            i = max(0, i - 1)
            continue
        if k == ord("b"):
            shutil.copy2(path, os.path.join(a.out, "bawa", name)); history.append(("bawa", name)); n_b += 1
        elif k == ord("k"):
            shutil.copy2(path, os.path.join(a.out, "kosong", name)); history.append(("kosong", name)); n_k += 1
        elif k == ord("s"):
            with open(skip_file, "a", encoding="utf-8") as f:
                f.write(name + "\n")
            history.append(("skip", name))
        else:
            continue
        i += 1
    cv2.destroyAllWindows()
    print(f"selesai. bawa={n_b} kosong={n_k}  -> {a.out}")


if __name__ == "__main__":
    main()