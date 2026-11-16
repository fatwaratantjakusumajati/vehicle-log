"""Train forklift carry-state classifier from labeled full-frame images.

Usage: python train_fk_carry.py --data dataset_fk --out forklift_carry_model_crops.pkl
"""
import argparse
import os
import pickle
import re

import cv2
import numpy as np

from cam1 import Cfg, ForkliftDetector, Zones, crop_vehicle
from feature_extractor import extract_features

POSITIVE = "forklift_bawa"
NEGATIVE = "forklift_kosong"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _session_group(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"(CAM1_\d{8}_\d{6}_\d{4})", stem)
    if match:
        return match.group(1)
    return re.sub(r"__t\d+(?:\.\d+)?$", "", stem)


def _new_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_fk")
    parser.add_argument("--out", default="forklift_carry_model_crops.pkl")
    args = parser.parse_args()

    cfg = Cfg()
    zones = Zones.load(cfg.zones_path)
    detector = ForkliftDetector(cfg, zones)
    if not detector.ok:
        raise RuntimeError("forklift detector tidak tersedia; crop training tidak dapat disamakan dengan runtime")

    items = []
    for label in (POSITIVE, NEGATIVE):
        root = os.path.join(args.data, label)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"folder label tidak ditemukan: {root}")
        for directory, _, filenames in os.walk(root):
            for filename in sorted(filenames):
                if filename.lower().endswith(EXTENSIONS):
                    path = os.path.join(directory, filename)
                    items.append((path, label, _session_group(path)))
    if not items:
        raise ValueError(f"tidak ada gambar berlabel di {args.data}")

    features, labels, groups = [], [], []
    skipped = {"invalid_image": 0, "no_forklift": 0, "multiple_forklifts": 0}
    for index, (path, label, group) in enumerate(items, 1):
        frame = cv2.imread(path)
        if frame is None:
            skipped["invalid_image"] += 1
            continue
        boxes = detector(frame)
        if not boxes:
            skipped["no_forklift"] += 1
            continue
        if len(boxes) != 1:
            skipped["multiple_forklifts"] += 1
            continue
        crop = crop_vehicle(frame, boxes[0], cfg.fk_crop_pad)
        if crop.size == 0:
            skipped["invalid_image"] += 1
            continue
        features.append(np.asarray(extract_features(crop), dtype=np.float32).ravel())
        labels.append(label)
        groups.append(group)
        if index % 20 == 0:
            print(f"  crop {index}/{len(items)}", flush=True)

    if len(set(labels)) != 2:
        raise ValueError(f"deteksi menyisakan satu kelas saja: {dict((label, labels.count(label)) for label in set(labels))}")

    x = np.stack(features)
    y = np.asarray(labels)
    g = np.asarray(groups)
    print(
        f"{len(y)} crop runtime-compatible "
        f"(bawa={int(np.sum(y == POSITIVE))}, kosong={int(np.sum(y == NEGATIVE))}, "
        f"sesi={len(set(g))}); dilewati={skipped}"
    )

    if len(set(g)) >= 2:
        actual, predicted = [], []
        for group in sorted(set(g)):
            test = g == group
            train = ~test
            if len(set(y[train])) != 2:
                continue
            model = _new_model().fit(x[train], y[train])
            probabilities = model.predict_proba(x[test])
            positive_index = list(model.classes_).index(POSITIVE)
            actual.extend(y[test] == POSITIVE)
            predicted.extend(probabilities[:, positive_index] >= 0.5)
        if actual:
            actual, predicted = np.asarray(actual), np.asarray(predicted)
            tp = int(np.sum(actual & predicted))
            precision = tp / max(1, int(predicted.sum()))
            recall = tp / max(1, int(actual.sum()))
            accuracy = float(np.mean(actual == predicted))
            print(
                f"leave-one-session-out (ambang 0.5): "
                f"akurasi={accuracy:.2f} precision_bawa={precision:.2f} recall_bawa={recall:.2f}"
            )
        else:
            print("leave-one-session-out dilewati: tidak ada fold dengan kedua kelas pada data latih")

    model = _new_model().fit(x, y)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as file:
        pickle.dump(
            {
                "model": model,
                "positive": POSITIVE,
                "negative": NEGATIVE,
                "threshold": 0.5,
                "input": "forklift_crop",
                "crop_pad": cfg.fk_crop_pad,
            },
            file,
        )
    print(f"model disimpan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
