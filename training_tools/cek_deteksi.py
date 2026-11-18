"""
cek_deteksi.py - cek KENAPA sebuah kendaraan tidak terdeteksi oleh cam1.py, di tahap mana ia dibuang.

Meniru cam1.py: YOLO dijalankan pada ROI (kotak pembatas poligon 'area'), lalu hanya kelas 2 (car) dan 7 (truck)
yang dipakai, lalu disaring: conf, zona (area/exclude), ukuran (CAM1_MIN_AREA_RATIO), dan 'itu forklift'.
Alat ini menampilkan SEMUA deteksi (conf >= 0.05, SEMUA kelas) dan alasan tiap deteksi lolos/ditolak.

Pakai (dari folder plate-det, sejajar dengan extract_crops.py, vehicle_activity.py, cam1_zones.json):
  python cek_deteksi.py --image tangkapan_layar.png
  python cek_deteksi.py --video klip/loading_01.mp4 --at 100
  python cek_deteksi.py --image frame.png --fk-model forklift_openvino_model     # sekalian cek filter forklift
Hasil: tabel di terminal + gambar cek_deteksi_hasil.jpg (hijau = lolos, kuning = lemah, merah = ditolak).
Untuk kondisi LIVE: simpan satu tangkapan layar dari stream saat truk ada, lalu pakai --image.
"""
import argparse
import sys


def decide(cls, conf, nx, ny, ratio, zones, a):
    """Mengembalikan (status, alasan). status: OK | LEMAH | TOLAK."""
    if cls not in (2, 7):
        return "TOLAK", "kelas bukan car(2)/truck(7): cam1 hanya memakai kelas 0,2,7 -> MENGABAIKAN"
    if conf < a.det_conf:
        return "TOLAK", f"conf {conf:.2f} < CAM1_DET_CONF {a.det_conf:.2f}: tidak dilacak"
    if not zones.in_area(nx, ny):
        return "TOLAK", "pusat kotak di luar poligon 'area' atau di dalam 'exclude'"
    if ratio < a.min_ratio:
        return "TOLAK", f"terlalu kecil: rasio {ratio:.4f} < CAM1_MIN_AREA_RATIO {a.min_ratio:.4f}"
    if conf < a.min_conf:
        return "LEMAH", f"conf {conf:.2f} < CAM1_MIN_CONF {a.min_conf:.2f}: bisa dilacak tapi TIDAK bisa membuka sesi baru"
    return "OK", "lolos semua saringan"


def main():
    ap = argparse.ArgumentParser(description="Cek tahap penolakan deteksi kendaraan (meniru cam1.py).")
    ap.add_argument("--image", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--at", type=float, default=0.0, help="detik di video")
    ap.add_argument("--zones", default="cam1_zones.json")
    ap.add_argument("--model", default="yolo11n_openvino_model")
    ap.add_argument("--imgsz", type=int, default=640, help="cam1.py memakai ukuran bawaan model (OpenVINO: 640)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--det-conf", type=float, default=0.15, help="= CAM1_DET_CONF (bawaan 0.15)")
    ap.add_argument("--min-conf", type=float, default=0.20, help="= CAM1_MIN_CONF (.env kamu 0.20)")
    ap.add_argument("--min-ratio", type=float, default=0.012, help="= CAM1_MIN_AREA_RATIO (.env kamu 0.012)")
    ap.add_argument("--fk-model", default="", help="aktifkan cek 'dikira forklift'")
    ap.add_argument("--fk-conf", type=float, default=0.35)
    ap.add_argument("--out", default="cek_deteksi_hasil.jpg")
    a = ap.parse_args()

    import cv2
    import numpy as np
    from extract_crops import YoloDet, Zones

    if a.image:
        frame = cv2.imread(a.image)
    elif a.video:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_MSEC, a.at * 1000)
        ok, frame = cap.read()
        cap.release()
        frame = frame if ok else None
    else:
        print("Beri --image atau --video.")
        return 1
    if frame is None:
        print("Gambar/frame tidak bisa dibaca.")
        return 1
    H, W = frame.shape[:2]
    zones = Zones(a.zones)
    if zones.area:
        xs = [p[0] for p in zones.area]
        ys = [p[1] for p in zones.area]
        rx1, ry1, rx2, ry2 = int(min(xs) * W), int(min(ys) * H), int(max(xs) * W), int(max(ys) * H)
    else:
        rx1, ry1, rx2, ry2 = 0, 0, W, H
    roi = frame[ry1:ry2, rx1:rx2]
    print(f"Frame {W}x{H} | ROI (potongan yang dilihat YOLO): x {rx1}-{rx2}, y {ry1}-{ry2}")

    det = YoloDet(a.model, 0.05, a.iou, None, a.imgsz)
    names = dict(getattr(det.m, "names", None) or {})
    raw = [(x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1, conf, cls) for (x1, y1, x2, y2, conf, cls) in det.detect(roi)]
    raw.sort(key=lambda d: -d[4])

    fk_all = []
    if a.fk_model:
        from vehicle_activity import ForkliftDet, expand_box, orange_pct, crop_box
        fkd = ForkliftDet(a.fk_model, a.fk_conf, a.iou, a.imgsz, zones)
        for d in fkd.detect(frame):
            if d[4] < a.fk_conf:
                continue
            ratio_f = (d[2] - d[0]) * (d[3] - d[1]) / (W * H) / zones.roi_area
            if ratio_f >= 0.06:
                continue
            if d[4] < 0.85 and orange_pct(crop_box(frame, d[:4])) < 2.5:
                continue
            fk_all.append(expand_box(d[:4], 0.05))
        print(f"Forklift lolos filter (gerbang oranye + ukuran): {len(fk_all)}")

    vis = frame.copy()
    for poly, col in [(zones.area, (0, 215, 255))] + [(p, (255, 0, 255)) for p in zones.exclude]:
        if len(poly) >= 3:
            cv2.polylines(vis, [np.array([(int(x * W), int(y * H)) for x, y in poly])], True, col, 2)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)

    colors = {"OK": (0, 200, 0), "LEMAH": (0, 215, 255), "TOLAK": (0, 0, 255)}
    veh_ok = 0
    n_veh = 0
    print(f"\n{'kelas':<12}{'conf':>6}  {'pusat(x,y)':<13}{'status':<7} alasan")
    for (x1, y1, x2, y2, conf, cls) in raw[:30]:
        nx, ny = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        ratio = (x2 - x1) * (y2 - y1) / (W * H) / zones.roi_area
        status, why = decide(cls, conf, nx, ny, ratio, zones, a)
        if cls in (2, 7):
            n_veh += 1
            if status != "TOLAK" and fk_all:
                from vehicle_activity import classify_vehicle
                if classify_vehicle(frame, (x1, y1, x2, y2), fk_all, ratio, 0.06, 2.5) == "forklift":
                    status, why = "TOLAK", "dikira FORKLIFT oleh filter forklift (tumpang tindih dengan kotak forklift / kecil + oranye)"
        if status == "OK":
            veh_ok += 1
        nm = f"{names.get(cls, cls)}({cls})"
        print(f"{nm:<12}{conf:>6.2f}  ({nx:.2f},{ny:.2f})   {status:<7} {why}")
        c = colors[status]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 3)
        cv2.putText(vis, f"{names.get(cls, cls)} {conf:.2f} {status}", (int(x1), max(24, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    cv2.imwrite(a.out, vis)

    print("\nKESIMPULAN:")
    if not raw:
        print("  YOLO tidak mendeteksi APA PUN (conf >= 0.05) di ROI. Model/ukuran input tidak melihat objeknya,"
              " atau objeknya terpotong keluar dari ROI.")
    elif n_veh == 0:
        print("  Ada deteksi, tetapi tidak satu pun kelas car/truck. Lihat tabel: kendaraan mungkin terbaca sebagai"
              " kelas lain (bus/train/dll) yang diabaikan cam1.py.")
    elif veh_ok == 0:
        print("  Ada deteksi car/truck, tetapi SEMUANYA ditolak/lemah. Alasan ada di kolom paling kanan.")
    else:
        print(f"  {veh_ok} kendaraan lolos semua saringan di frame ini. Kalau live tetap tidak terdeteksi,"
              " masalahnya di tahap setelah deteksi (pelacakan/sesi) atau di frame lain; kirim log live-nya.")
    print(f"Gambar anotasi: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())