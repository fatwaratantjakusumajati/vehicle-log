"""
Script untuk menentukan ROI (Region of Interest) dari stream RTSP.
Output: xmin, xmax, ymin, ymax

Cara pakai:
    python select_roi_rtsp.py rtsp://user:pass@ip:port/stream

Kontrol saat window terbuka:
    - Klik kiri + drag untuk menggambar kotak ROI
    - ENTER / SPACE untuk konfirmasi ROI
    - C untuk cancel/reset seleksi
    - ESC untuk keluar tanpa memilih
"""

import os
import cv2
import sys
import argparse
import numpy as np

# Paksa transport RTSP pakai TCP (lebih stabil daripada UDP, kurang packet loss)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def is_frame_blank(frame, std_threshold=5.0):
    """Cek apakah frame kosong/abu-abu polos (variasi pixel sangat rendah)."""
    return frame.std() < std_threshold


def get_frame_from_rtsp(rtsp_url, max_retries=30, warmup_skip=10):
    """
    Ambil satu frame valid dari stream RTSP.
    - warmup_skip: jumlah frame awal yang dibuang (decoder butuh waktu sinkron)
    - max_retries: total percobaan read sebelum menyerah
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise ConnectionError(f"Tidak bisa membuka stream RTSP: {rtsp_url}")

    frame = None
    valid_frame = None

    for i in range(max_retries):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Buang frame-frame awal (warm-up decoder)
        if i < warmup_skip:
            continue

        # Simpan frame pertama yang bukan abu-abu polos
        if not is_frame_blank(frame):
            valid_frame = frame
            break

    cap.release()

    if valid_frame is not None:
        return valid_frame

    if frame is not None:
        print("PERINGATAN: Semua frame terlihat kosong/abu-abu, memakai frame terakhir apa adanya.")
        return frame

    raise RuntimeError("Gagal mengambil frame dari RTSP setelah beberapa percobaan.")


def select_roi(frame, window_name="Pilih ROI - Enter/Space untuk konfirmasi, C untuk reset, ESC untuk batal"):
    """
    Pilih ROI pada frame yang sudah di-fit ke layar, lalu konversikan
    koordinat hasil seleksi kembali ke resolusi frame CCTV asli.
    """
    original_h, original_w = frame.shape[:2]

    # Ambil ukuran layar. Fallback jika Tkinter tidak tersedia.
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080

    # Sisakan ruang untuk taskbar/border window.
    max_w = int(screen_w * 0.90)
    max_h = int(screen_h * 0.85)

    scale = min(max_w / original_w, max_h / original_h, 1.0)
    display_w = int(original_w * scale)
    display_h = int(original_h * scale)

    if scale < 1.0:
        display_frame = cv2.resize(
            frame, (display_w, display_h), interpolation=cv2.INTER_AREA
        )
    else:
        display_frame = frame.copy()

    print(f"Resolusi CCTV asli : {original_w}x{original_h}")
    print(f"Resolusi tampilan   : {display_w}x{display_h}")
    print(f"Scale               : {scale:.4f}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, display_w, display_h)

    roi = cv2.selectROI(
        window_name, display_frame, showCrosshair=True, fromCenter=False
    )
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None

    # Kembalikan koordinat dari frame tampilan ke resolusi CCTV asli.
    xmin = int(round(x / scale))
    ymin = int(round(y / scale))
    xmax = int(round((x + w) / scale))
    ymax = int(round((y + h) / scale))

    # Pastikan koordinat tetap berada dalam frame asli.
    xmin = max(0, min(xmin, original_w - 1))
    xmax = max(0, min(xmax, original_w))
    ymin = max(0, min(ymin, original_h - 1))
    ymax = max(0, min(ymax, original_h))

    return xmin, xmax, ymin, ymax


def draw_and_save_preview(frame, roi, out_path="roi_preview.jpg"):
    """Gambar kotak ROI di frame lalu simpan sebagai gambar preview."""
    xmin, xmax, ymin, ymax = roi
    preview = frame.copy()
    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imwrite(out_path, preview)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Pilih ROI dari stream RTSP")
    parser.add_argument("rtsp_url", help="URL RTSP, contoh: rtsp://user:pass@192.168.1.xxx:554/stream")
    parser.add_argument("--save-preview", action="store_true", help="Simpan gambar preview dengan kotak ROI")
    args = parser.parse_args()

    print(f"Menghubungkan ke stream: {args.rtsp_url}")
    frame = get_frame_from_rtsp(args.rtsp_url)
    print(f"Frame berhasil diambil, ukuran asli: {frame.shape[1]}x{frame.shape[0]}")

    roi = select_roi(frame)

    if roi is None:
        print("ROI tidak dipilih (dibatalkan).")
        sys.exit(1)

    xmin, xmax, ymin, ymax = roi
    print("\n=== HASIL ROI ===")
    print(f"xmin = {xmin}")
    print(f"xmax = {xmax}")
    print(f"ymin = {ymin}")
    print(f"ymax = {ymax}")
    print("\nSiap copy ke program deteksi:")
    print(f"ROI = ({xmin}, {xmax}, {ymin}, {ymax})")

    if args.save_preview:
        path = draw_and_save_preview(frame, roi)
        print(f"\nPreview disimpan di: {path}")


if __name__ == "__main__":
    main()