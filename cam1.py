"""
cam1.py - Monitoring Loading Dock (CAM 1 "Loading Kiri"), rombak total.

IDE DASAR (lebih sederhana dari versi lama)
-------------------------------------------
Versi lama bergantung pada ID track YOLO yang terus pecah (192 track untuk 2 truk) lalu ditambal dengan
RE-LINK, filter oranye, filter forklift, dll. Di sini TIDAK ADA identitas track. Yang dilacak adalah
KEBERADAAN kendaraan di area dock (dermaga satu jalur):

    ada kendaraan valid terlihat  -> sesi dibuka   (waktu MASUK  = deteksi pertama)
    tidak terlihat lagi > toleransi -> sesi ditutup (waktu KELUAR = deteksi terakhir)

Toleransi keluar bergantung kondisi terakhir: kendaraan yang sedang PARKIR diberi waktu lama (tertutup
forklift/orang/deteksi berkedip tidak memecah sesi); kendaraan yang sedang BERGERAK diberi waktu singkat.

STATUS LOADING / UNLOADING
--------------------------
Selama kendaraan parkir, tiap ~1,5 dtk bak dipotret -> cargo_classifier (MobileNetV2 + LR milik Anda)
memberi P(bermuatan). Saat sesi ditutup: median sampel AWAL vs median sampel AKHIR.
    naik  (kosong -> penuh)  = LOADING
    turun (penuh  -> kosong) = UNLOADING
Perubahan lemah / tanpa dukungan keadaan akhir diberi flag perlu_review=true (sesuai prinsip di diagram:
"truk berhenti belum tentu bongkar-muat", butuh bukti visual) dan 3 foto bukti disimpan.

REVISI (multi-kendaraan + anti-forklift)
----------------------------------------
- Banyak sesi sekaligus (maks CAM1_MAX_SESSIONS): 2 kendaraan di dock = 2 sesi terpisah, masing-masing punya
  MASUK/KELUAR/status sendiri. Dulu hanya 1 sesi: kotak TERBESAR yang menang, sehingga kendaraan lain / forklift
  menimpa atau menyatu ke sesi yang sama (foto masuk/aktivitas/keluar bisa dari 3 kendaraan berbeda).
- Forklift dikenali dari BLOB ORANYE di frame (berapa pun ukurannya), bukan hanya dari "kotak kecil + oranye":
    kotak YOLO yang isinya forklift            -> dibuang (tidak pernah jadi sesi)
    kotak GABUNGAN forklift + kendaraan        -> dianggap "tertutup forklift": sesi tetap hidup, tetapi
                                                  bbox/sampel/foto tidak diperbarui dari kotak itu
    kotak kendaraan yang bersih                -> dipakai normal
- Deteksi lemah (CAM1_TRACK_CONF) hanya boleh MELANJUTKAN sesi yang sudah ada (kendaraan yang sebagian tertutup
  forklift sering conf-nya turun); membuka sesi baru tetap butuh CAM1_MIN_CONF.
- Setelah jeda > swap_gap, deteksi di tempat yang sama hanya dianggap kendaraan yang sama bila tampilannya mirip
  (histogram warna HSV); kalau beda -> sesi baru.
- Waktu dekat forklift dihitung per sesi. Bila skor muatan tidak bisa memutuskan (mis. mobil boks tertutup) tetapi
  ada aktivitas forklift >= CAM1_FK_NEAR_MIN detik, status diasumsikan CAM1_FORKLIFT_ASSUME. Default KOSONG
  (tidak menebak). Bila diisi LOADING/UNLOADING: sumber="asumsi_forklift" + perlu_review=true.

ARAH DARI GERAKAN FORKLIFT (untuk kendaraan tertutup)
-----------------------------------------------------
Isi bak boks/Granmax tidak terlihat, jadi arah dibaca dari forklift: tiap forklift di dekat kendaraan dilacak
dan crop-nya diberi skor P(bawa barang) oleh forklift_carry.py. Transisi bawa -> kosong dihitung sebagai TARUH
(barang masuk kendaraan = LOADING); kosong -> bawa dihitung sebagai AMBIL (barang keluar = UNLOADING).
Mayoritas kejadian menentukan arah, sumber="forklift_arah". Hanya aktif bila forklift_carry_model.pkl ada.
Skor muatan yang kuat dan konsisten tetap diutamakan; bila bertentangan -> perlu_review.

FUSI BUKTI (decide) & REVIEW
----------------------------
Keputusan Loading/Unloading = penjumlahan bukti dalam log-odds (positif = Loading):
    skor bak (naik/turun)  +  arah forklift (taruh/ambil)  +  logika "tiba kosong tidak mungkin Unloading"
confidence = tanh(|jumlah|/2). Skor bak yang None (ditolak classifier), berisik, atau sampelnya sedikit hanya
MELEMAHKAN buktinya, tidak pernah dihitung sebagai "Kosong". perlu_review hanya bila confidence < CAM1_REVIEW_MIN
(default 0.30); di bawah itu status = "Tidak Dapat Dipastikan".

PELABELAN OTOMATIS FORKLIFT (tanpa label manual)
------------------------------------------------
Sesi dengan hasil bak yang yakin (CAM1_FK_AUTOLABEL_CONF) menghasilkan label gratis untuk forklift: pada Loading,
forklift SEBELUM menyentuh kendaraan = bawa, SESUDAH = kosong (Unloading dibalik). Crop disimpan ke
CAM1_FK_AUTOLABEL_DIR/forklift_bawa|forklift_kosong -> python train_fk_carry.py --data <folder itu>.

Plat nomor dihapus total. cargo_classifier.py & feature_extractor.py dipakai apa adanya.

CARA PAKAI
----------
    python cam1.py                                  # live RTSP (dari .env)
    python cam1.py --video rekaman.mp4 --no-send    # uji offline (tidak kirim ke n8n)
    python cam1.py --draw --image frame.jpg         # editor zona (klik poligon)
    python cam1.py --preview --image frame.jpg      # tulis zones_preview.jpg
"""
import os

# Harus sebelum cv2 diimpor: RTSP lewat TCP supaya tidak ada artefak/frame rusak.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;10000000")

import argparse
import base64
import json
import logging
import math
import queue
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque, namedtuple
from datetime import datetime
from logging.handlers import RotatingFileHandler
from statistics import median

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

log = logging.getLogger("cam1")

# ----------------------------------------------------------------------------------------------
# Label status (ubah di sini bila n8n/dashboard Anda memfilter dengan teks lain)
# ----------------------------------------------------------------------------------------------
LOADING = "Loading"
UNLOADING = "Unloading"
UNSURE = "Tidak Dapat Dipastikan"
# [TINGKAT-A] truk boks tertutup yang dikerjakan manusia: ada aktivitas di stage, ARAH tidak ditebak.
# Teks di web dibuat sama dengan hasil "tidak pasti" yang biasa. Pembedanya ada di source="stage_orang" dan alasan di log/payload.
# Bila ingin teks sendiri: JANGAN mengandung kata "loading" / "bongkar" / "muatan" (classifyAktivitas akan salah menganggapnya Loading/Unloading).
STAGE_ACT = UNSURE

Det = namedtuple("Det", "x1 y1 x2 y2 conf cls")


# ----------------------------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------------------------
def _env(key, default, cast=float):
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        if cast is bool:
            return str(v).strip().lower() in ("1", "true", "yes", "on")
        return cast(str(v).strip().strip('"'))
    except (TypeError, ValueError):
        return default


def build_rtsp_url():
    override = os.environ.get("CAM1_RTSP_URL", "").strip()
    if override:
        return override
    user, pw = os.environ.get("CAMERA_USER", ""), os.environ.get("CAMERA_PASS", "")
    host, port = os.environ.get("CAMERA_HOST", ""), os.environ.get("CAMERA_PORT", "554")
    ch = os.environ.get("CAM1_CHANNEL", "101")
    return f"rtsp://{user}:{pw}@{host}:{port}/Streaming/Channels/{ch}"


class Cfg:
    """Semua parameter bisa di-override lewat .env (prefix CAM1_). Default = nilai awal yang masuk akal."""

    def __init__(self):
        e = _env
        self.camera_id = "cam1"
        self.camera_name = "Loading Kiri"
        self.model_path = os.environ.get("CAM1_MODEL_PATH") or os.environ.get("MODEL_PATH", "yolo11n_openvino_model")
        self.n8n_url = os.environ.get("N8N_WEBHOOK_URL", "")
        self.zones_path = os.environ.get("CAM1_ZONES", "cam1_zones.json")
        self.evidence_dir = os.environ.get("CAM1_EVIDENCE_DIR", "evidence_cam1")
        self.events_log = os.environ.get("CAM1_EVENTS_LOG", "events_cam1.jsonl")
        self.outbox_path = os.environ.get("CAM1_OUTBOX_PATH", "n8n_outbox_cam1.jsonl")

        # deteksi
        self.fps = e("CAM1_FPS", 2.0)                      # frame yang diproses per detik (2-3 cukup untuk CPU)
        self.det_log = e("CAM1_DET_LOG", 0, int)                     # 1 = tulis baris DET (tiap ~2 dtk: semua kendaraan yang dilihat YOLO + alasan diterima/ditolak). Untuk diagnosis.
        self.det_conf = e("CAM1_DET_CONF", 0.15)
        self.min_conf = e("CAM1_MIN_CONF", 0.50)           # kendaraan di bawah ini diabaikan
        self.iou = e("IOU_THRESHOLD", 0.45)
        self.min_area_ratio = e("CAM1_MIN_AREA_RATIO", 0.02)        # terhadap luas ROI (bbox zona 'area')
        # [FIX-JARAK-DERMAGA] kendaraan yang kotaknya lebih kecil dari ini (terhadap luas ROI) dianggap MASIH JAUH dari dermaga (antre/parkir di
        # jalan atas frame): waktu parkirnya TIDAK dihitung untuk gerbang MASUK. Di log: kotak jauh/benda diam 0.015-0.038, mobil dekat kamera >= 0.049,
        # truk dermaga >= 0.088. Celah sempit (0.038-0.049). BAWAAN MATI (0): truk boleh bongkar/muat di strip jauh. Isi mis. 0.045 bila ingin menyaring. 0 = mati.
        self.dock_min_ratio = e("CAM1_DOCK_MIN_RATIO", 0.0)
        # [PINDAH-POSISI] truk terkonfirmasi yang pindah tempat (mis. maju ke ujung halaman) dan sesaat hanya terbaca LEMAH (conf < min_conf) +
        # mengecil (kotak lebih kecil sampai relocate_size_ratio x) tidak bisa disambung _match_score maupun SERAH-TERIMA. Bila PRESENSI sudah
        # memastikan tempat parkir lama kosong, relocate_hits sighting lemah beruntun di tempat baru = kendaraan yang sama pindah. 0 = mati.
        self.roi_ext_top = e("CAM1_ROI_EXT_TOP", 0.05, float)      # [ROI-ATAS] ROI YOLO diperbesar ke atas sebesar fraksi tinggi frame (truk di ujung atas tidak terpotong); 0 = mati
        self.roi_ext_right = e("CAM1_ROI_EXT_RIGHT", 0.05, float)  # [ROI-KANAN] ROI YOLO diperlebar ke kanan (fraksi lebar frame); 0 = mati
        self.edge_margin = e("CAM1_EDGE_MARGIN", 0.06, float)      # [KENDARAAN-TEPI] kotak yang pusatnya <= fraksi lebar frame di luar poligon area masih boleh membuka sesi; 0 = mati
        self.edge_park = e("CAM1_EDGE_PARK", 10.0, float)          # sesi yang dibuka dari tepi wajib parkir diam berturut-turut >= ini (dtk); yang hanya lewat tidak pernah diumumkan
        self.zone_hyst = e("CAM1_ZONE_HYST", 1, int)       # [HISTERESIS-ZONA] 0 = mati (perilaku lama: pusat di luar area = ditolak)
        self.relocate_hits = e("CAM1_RELOCATE_HITS", 2, int)
        self.relocate_window = e("CAM1_RELOCATE_WINDOW", 60.0)
        self.relocate_size_ratio = e("CAM1_RELOCATE_SIZE_RATIO", 8.0)
        self.relocate_presence_misses = e("CAM1_RELOCATE_PRESENCE_MISSES", 2, int)  # spot lama harus terbukti kosong berulang sebelum sesi dipindah
        self.forklift_max_area = e("CAM1_FORKLIFT_MAX_AREA", 0.06)  # box lebih kecil + oranye = forklift
        self.orange_min_pct = e("CAM1_ORANGE_MIN_PCT", 2.5)
        # [FIX-FORKLIFT-HANTU-2] forklift oranye yang dekat kamera berkotak BESAR, jadi lolos aturan "kotak kecil + oranye" dan terbaca 'Truk'
        # (dashboard 4 Nov 07:44: foto Masuk & Keluar = forklift). Kotak dengan oranye >= nilai ini dianggap forklift berapa pun ukurannya,
        # KECUALI kotak itu menimpa sesi yang sudah ada (kendaraan + forklift menempel -> diperlakukan 'gabungan', sesi tetap hidup).
        self.orange_big_pct = e("CAM1_ORANGE_BIG_PCT", 6.0)
        # [SESI-GANDA] kotak KECIL (pecahan truk yang sebagian tertutup forklift/orang) di dalam/dekat truk yang sedang aktif jangan dibuka jadi sesi baru
        # dan jangan dipakai SERAH-TERIMA yang mengecilkan kotak sesi. Nilai = jendela (detik) sejak sesi itu terakhir terlihat. 0 = mati.
        # [FOTO-AWAL-AKHIR] cadangan: bila keputusan biasa gagal/dibuang, pakai label muatan awal (Kosong/Bermuatan) -> akhir. 0 = mati.
        self.foto_fb = e("CAM1_FOTO_FB", 0)
        self.foto_fb_empty = e("CAM1_FOTO_FB_EMPTY", 0.45)    # skor awal/akhir <= ini = Kosong
        self.foto_fb_loaded = e("CAM1_FOTO_FB_LOADED", 0.50)  # skor awal/akhir >= ini = Bermuatan
        self.foto_fb_delta = e("CAM1_FOTO_FB_DELTA", 0.10)    # selisih minimal skor akhir - awal
        self.open_guard = e("CAM1_OPEN_GUARD", 0.0)
        self.open_guard_ratio = e("CAM1_OPEN_GUARD_RATIO", 0.6)     # kandidat dianggap pecahan bila luasnya < ratio x luas kotak sesi
        self.open_guard_expand = e("CAM1_OPEN_GUARD_EXPAND", 1.5)   # pusat kandidat harus di dalam kotak sesi yang diperbesar sekian x
        self.green_big_pct = e("CAM1_GREEN_BIG_PCT", 0.0)       # [FK-HIJAU] 0 = mati. Kotak 'truk' yang >= persen ini hijau-tosca = forklift hijau (disarankan uji 3.0)
        self.sess_orange_frac = e("CAM1_SESS_ORANGE_FRAC", 0.5)   # sesi yang >= pecahan ini pembaruannya beroranye besar = forklift, MASUK ditahan. 0 = mati
        self.track_conf = e("CAM1_TRACK_CONF", 0.30)       # deteksi selemah ini boleh MELANJUTKAN sesi (bukan membuka)
        self.max_sessions = e("CAM1_MAX_SESSIONS", 3, int)
        self.appearance_max_dist = e("CAM1_APPEARANCE_MAX_DIST", 0.55)  # 0 = identik, 1 = beda total
        self.person_conf = e("CAM1_PERSON_CONF", 0.35)
        # forklift berbasis blob oranye
        self.fk_blob = e("CAM1_FK_BLOB", True, bool)
        self.fk_min_area = e("CAM1_FK_MIN_AREA", 0.001)    # luas minimum blob (rasio terhadap ROI)
        self.fk_max_area = e("CAM1_FK_MAX_AREA", 0.45)     # blob lebih besar = cahaya/warna seluruh scene, bukan forklift
        self.fk_expand = e("CAM1_FK_EXPAND", 0.12)         # perluas kotak forklift (mast, operator)
        self.fk_near_min = e("CAM1_FK_NEAR_MIN", 10.0)     # detik forklift di dekat kendaraan = bukti aktivitas
        self.bed_fk_overlap = e("CAM1_BED_FK_OVERLAP", 0.15)  # sampel muatan dilewati bila forklift menutup bak segini
        self.fk_assume = os.environ.get("CAM1_FORKLIFT_ASSUME", "").strip().capitalize()   # kosong = tidak menebak
        # arah dari gerakan forklift (bawa/kosong)
        self.fk_carry_interval = e("CAM1_FK_CARRY_INTERVAL", 0.8)   # jeda minimal antar skor per forklift (dtk)
        self.fk_link_gap = e("CAM1_FK_LINK_GAP", 12.0)               # forklift hilang <= ini (mis. masuk bak) masih track yang sama
        self.fk_link_dist = e("CAM1_FK_LINK_DIST", 1.5)              # jarak pusat maksimum (x ukuran kotak) untuk menyambung track
        self.fk_min_obs = e("CAM1_FK_MIN_OBS", 4, int)               # skor minimal per track supaya dihitung
        self.fk_hi = e("CAM1_FK_CARRY_HI", 0.60)                     # P >= ini = bawa
        self.fk_lo = e("CAM1_FK_CARRY_LO", 0.40)                     # P <= ini = kosong (di antaranya: pertahankan status)
        self.fk_min_run = e("CAM1_FK_MIN_RUN", 2, int)               # status harus bertahan segini skor berturut-turut
        self.fk_geo = e("CAM1_FK_GEO", 0, int)                       # 1 = arah forklift mempertimbangkan LOKASI transisi (dekat truk vs jauh). DEFAULT MATI: di record-5-10 terbukti membalik hasil
        self.fk_geo_margin = e("CAM1_FK_GEO_MARGIN", 0.25)           # beda jarak rata-rata (x ukuran truk) taruh vs ambil agar lokasi dianggap bicara
        self.fk_crop_pad = e("CAM1_FK_CROP_PAD", 0.0)               # lebarkan crop forklift (pecahan ukuran kotak per sisi) agar palet di garpu ikut terpotret; 0 = persis kotak detektor
        self.fk_stage = e("CAM1_FK_STAGE", 0, int)                   # 1 = pakai poligon "stage" (tumpukan palet) sebagai penanda lokasi. DEFAULT 0 sampai poligon diverifikasi di frame (di log, forklift tak pernah masuk poligon & kendaraan malah parkir di sana)
        self.fk_loc = e("CAM1_FK_LOC", 1, int)                       # 1 = lokasi ambil/taruh dibaca dari JARAK forklift ke kendaraan: >= fk_far = di tumpukan (jauh), <= fk_near = di kendaraan, di antaranya = tidak bersuara
        self.fk_far = e("CAM1_FK_FAR", 1.2, float)                   # jarak (x ukuran kendaraan) minimal agar transisi dianggap terjadi di tumpukan
        self.fk_near = e("CAM1_FK_NEAR", 1.1, float)                 # jarak maksimal agar transisi dianggap terjadi di kendaraan
        # [FIX-PARKIR-TOTAL] alternatif rentetan: total waktu parkir (akumulasi, boleh terputus orang/forklift lewat) >= CAM1_PARK_TOTAL dtk
        # DAN rentetan terpanjang >= CAM1_PARK_MIN_RUN dtk. 0 = mati (hanya aturan rentetan CAM1_PARK_CONFIRM).
        self.park_total = e("CAM1_PARK_TOTAL", 0.0, float)
        self.park_min_run = e("CAM1_PARK_MIN_RUN", 6.0, float)
        self.park_confirm = e("CAM1_PARK_CONFIRM", 6.0, float)       # kendaraan sejati harus pernah parkir diam berturut-turut >= ini (dtk) sebelum MASUK diumumkan; 0 = mati
        self.fk_ovl_max = e("CAM1_FK_OVL_MAX", 0.6, float)           # bila > pecahan ini dari pembaruan sesi kotaknya >= 60% tertutup kotak forklift, sesi dianggap forklift/bukan kendaraan
        self.announce_min_samples = e("CAM1_ANNOUNCE_MIN_SAMPLES", 4, int)   # MASUK baru diumumkan setelah >= sekian sampel bak valid (forklift+pengemudi parkir hampir tak punya sampel). 0 = mati
        self.announce_wait = e("CAM1_ANNOUNCE_WAIT", 90.0, float)           # ...kecuali kendaraan sudah terlihat selama ini (dtk): jangan sampai kendaraan sungguhan yang bak-nya lama tertutup tak pernah tercatat
        self.fk_ovl_min_n = e("CAM1_FK_OVL_MIN_N", 10, int)          # minimal pembaruan sebelum aturan di atas berlaku
        # [FK-OVL-VETO] hitung juga forklift yang dibuang gerbang FK-HANTU (conf >= fk_veto_conf, ingatan fk_veto_mem dtk) dan forklift yang baru lewat
        # (ingatan _fk_mem) pada hitungan 'kotak sesi tertutup kotak forklift'. 0 = mati. Dipakai bersama CAM1_FK_OVL_MAX (disarankan 0.3).
        self.fk_ovl_veto = e("CAM1_FK_OVL_VETO", 0, int)
        self.fk_min_crop = e("CAM1_FK_MIN_CROP", 40, int)            # sisi terpendek crop forklift (px)
        self.sample_dump_interval = e("CAM1_SAMPLE_DUMP_INTERVAL", 8.0)   # simpan 1 crop bak tiap sekian dtk per sesi (+ crop tambahan bila skor berubah >= 0.15). 0 = perilaku lama (tiap sampel, 150 pertama)
        self.sample_dump_max = e("CAM1_SAMPLE_DUMP_MAX", 250, int)         # batas crop tersimpan per sesi
        self.fk_sample_dir = os.environ.get("CAM1_FK_SAMPLE_DIR", "").strip().strip('"')   # simpan crop forklift (bahan latih)
        # detektor forklift berbasis MODEL (OpenVINO, hasil convert_forklift.py). Kosong = mati (perilaku lama).
        # Kotaknya DITAMBAHKAN ke kotak forklift hasil blob warna; tidak ada logika lain yang berubah.
        self.fk_model_path = os.environ.get("CAM1_FK_MODEL", "").strip().strip('"')
        self.fk_backend = os.environ.get("CAM1_FK_BACKEND", "direct").strip().lower()   # direct | ultralytics
        self.fk_det_conf = e("CAM1_FK_DET_CONF", 0.35)
        # [FIX-FK-HANTU] deteksi model forklift dengan conf < nilai ini HARUS mengandung piksel oranye (>= orange_min_pct),
        # kalau tidak dibuang (model sering salah mengenali truk boks putih sebagai forklift). 0 = gerbang mati.
        self.fk_orange_gate = e("CAM1_FK_ORANGE_GATE", 0.85)
        # [FK-VETO] forklift NON-oranye yang terdeteksi MODEL forklift (conf >= fk_veto_conf) tetapi dibuang gerbang FK-HANTU tidak ikut menolak kotak
        # "truk" di atasnya, padahal YOLO memberi label truck untuk forklift (tidak ada kelas forklift). Bila aktif (1): kandidat KENDARAAN KECIL
        # (rasio luas <= fk_veto_max_ratio) yang >= fk_veto_cover bagiannya tertutup kotak forklift itu (dalam fk_veto_mem dtk terakhir), conf YOLO-nya
        # < fk_veto_max_conf, dan TIDAK bertumpuk dengan kendaraan yang sudah diumumkan, dianggap FORKLIFT. Truk besar / yakin / terkonfirmasi tidak terpengaruh.
        # Default 0 (mati): perilaku lama.
        self.fk_veto = e("CAM1_FK_VETO", 0, int)
        self.fk_veto_conf = e("CAM1_FK_VETO_CONF", 0.45)
        self.fk_veto_cover = e("CAM1_FK_VETO_COVER", 0.60)
        self.fk_veto_max_ratio = e("CAM1_FK_VETO_MAX_RATIO", 0.07)
        self.fk_veto_max_conf = e("CAM1_FK_VETO_MAX_CONF", 0.70)
        self.fk_veto_mem = e("CAM1_FK_VETO_MEM", 3.0)
        # [FK-GARPU] forklift yang tampak hanya GARPU (tubuh oranye tertutup truk) tidak lolos gerbang oranye. Bila aktif (1), deteksi
        # tidak-oranye tetap DITERIMA hanya jika: ada sesi truk parkir yang sudah diumumkan, pusatnya di sekitar kotak truk itu
        # (diperluas fk_garpu_pad) atau di poligon stage, DAN kotaknya kecil (<= fk_garpu_area x luas truk; truk boks yang salah baca tidak lolos).
        # Default 0 (mati): perilaku lama. Di luar syarat itu, hantu tetap dibuang.
        self.fk_garpu = e("CAM1_FK_GARPU", 0, int)
        self.fk_garpu_area = e("CAM1_FK_GARPU_AREA", 0.35)
        self.fk_garpu_pad = e("CAM1_FK_GARPU_PAD", 0.15)
        # [TINGKAT-A] truk parkir + orang berada di poligon 'stage' >= stage_act_min detik -> status STAGE_ACT + perlu_review.
        # Hanya dipakai bila hasil lain "Tidak Dapat Dipastikan". 0 = mati. Butuh poligon 'stage' di cam1_zones.json.
        self.stage_act_min = e("CAM1_STAGE_ACT_MIN", 8.0)
        self.stage_act_conf = e("CAM1_STAGE_ACT_CONF", 0.30)     # conf minimal YOLO untuk orang yang dihitung
        self.stage_act_down = e("CAM1_STAGE_ACT_DOWNGRADE", 1)   # 1 = tebakan arah berkeyakinan rendah (PERLU REVIEW) tanpa forklift ikut diganti STAGE_ACT
        self.stage_act_near = e("CAM1_STAGE_ACT_NEAR", 0.35)     # orang harus di dalam kotak truk diperluas sekian (pecahan ukuran)
        self.fk_det_expand = e("CAM1_FK_DET_EXPAND", 0.05)   # perluas sedikit kotak model (garpu/operator di tepi)

        # sesi
        self.entry_hits = e("CAM1_ENTRY_HITS", 3, int)
        self.entry_seconds = e("CAM1_ENTRY_SECONDS", 5.0)  # sesi harus bertahan segini sebelum MASUK dikirim
        # konfirmasi kuat: MASUK baru dikirim bila sesi punya >= confirm_hits deteksi dengan conf >= confirm_conf.
        # Berguna saat CAM1_MIN_CONF diturunkan (agar kendaraan berbak terbuka/jauh terlacak sejak awal) tetapi
        # barang/objek lemah (conf rendah terus) tidak ikut terhitung. confirm_conf=0 = mati (perilaku lama).
        self.confirm_conf = e("CAM1_CONFIRM_CONF", 0.50)
        self.confirm_hits = e("CAM1_CONFIRM_HITS", 2, int)
        self.moving_grace = e("CAM1_MOVING_GRACE", 10.0)   # hilang saat bergerak -> keluar
        self.parked_grace = e("CAM1_PARKED_GRACE", 120.0)  # hilang saat parkir -> tunggu lebih lama
        self.swap_gap = e("CAM1_SWAP_GAP", 8.0)            # kendaraan baru di tempat lain setelah jeda ini = sesi baru
        self.hard_max = e("CAM1_SESSION_HARD_MAX", 43200.0)
        self.stat_window = e("CAM1_STATIONARY_WINDOW", 4.0)
        self.stat_move_ratio = e("CAM1_STATIONARY_MOVE_RATIO", 0.06)
        # [FIX-PARKIR-STABIL] kendaraan yang sudah parkir TETAP dianggap parkir selama pusat kotaknya masih dalam N x ukuran kendaraan dari titik parkir.
        # Orang/forklift yang lewat membuat kotak pickup bergoyang -> dulu terbaca "bergerak" -> sampel bak berhenti & sesi ditutup lewat
        # CAM1_MOVING_GRACE (10 dtk) padahal kendaraan diam. 0 = perilaku lama.
        self.park_hold_ratio = e("CAM1_PARK_HOLD_RATIO", 0.5)
        self.settle = e("CAM1_SETTLE_SECONDS", 3.0)
        # [PATCH-PREPARK] N detik pertama sesi: sampel bak BOLEH diambil walau truk belum dianggap parkir, supaya keadaan
        # saat TIBA (sebelum forklift/orang mulai bekerja) ikut tercatat sebagai skor awal. 0 = mati (perilaku lama).
        self.pre_park_window = e("CAM1_PRE_PARK_WINDOW", 0.0)

        # status muatan
        self.sample_interval = e("CAM1_SAMPLE_INTERVAL", 1.5)
        self.min_crop_side = e("CARGO_MIN_CROP_SIDE", 70, int)
        self.min_crop_pixels = e("CARGO_MIN_CROP_PIXELS", 8000, int)
        self.phase_k = e("CAM1_PHASE_K", 5, int)
        self.phase_frac = e("CAM1_PHASE_FRAC", 0.20)           # jendela fase = max(phase_k, frac x jumlah sampel)
        self.min_phase = e("CAM1_MIN_PHASE", 3, int)
        self.delta_strong = e("CAM1_DELTA_STRONG", 0.30)
        self.delta_weak = e("CAM1_DELTA_WEAK", 0.15)
        self.loaded_min = e("CAM1_LOADED_MIN", 0.45)
        self.max_samples = e("CAM1_MAX_SAMPLES", 300, int)
        # kosongkan = TIDAK pernah menebak. Isi LOADING/UNLOADING = dipakai HANYA bila skor tak bisa memutuskan
        # (ditandai sumber="asumsi" + perlu_review=true).
        self.default_direction = os.environ.get("CAM1_DEFAULT_DIRECTION", "").strip().capitalize()
        # fusi bukti (log-odds). confidence = tanh(|L|/2); review hanya bila confidence < review_min.
        self.fk_single_w = e("CAM1_FK_SINGLE_W", 0.0)        # pengali bobot arah forklift bila selisih taruh-ambil hanya 1 (0 = diabaikan; 1.0 = perilaku lama). Model forklift bawa/kosong masih berisik: pada log uji, 1 kejadian tunggal justru salah arah
        self.seq_enable = e("CAM1_SEQ_ENABLE", 1, int)       # 1 = kenali dua fase berurutan (Unloading+Loading) dalam satu sesi
        # [FIX-DUA-FASE] fase pertama (tiba bermuatan, sebelum orang/forklift menutupi bak) sering hanya 1-2 sampel; syarat 3 sampel/fase
        # membuat Unloading+Loading tidak pernah terbaca. Khusus pencarian 3 segmen dipakai batas ini (selisih & penurunan galat tetap ketat).
        self.seq_min_phase = e("CAM1_SEQ_MIN_PHASE", 2, int)
        self.seq_delta = e("CAM1_SEQ_DELTA", 0.25)           # selisih skor minimal tiap perubahan
        self.early_scale = e("CAM1_EARLY_SCALE", 1.5)        # early_drop hanya membandingkan sampel yang ukuran kotaknya berbeda <= sekian kali (cegah efek pindah posisi/skala)
        self.early_drop = e("CAM1_EARLY_DROP", 1, int)      # 1 = skor bak TURUN tajam sesudah sampel pertama (mis. 0.59 -> 0.28) lalu keputusan akhir Loading => Unloading+Loading (perlu_review)
        self.seq_gain = e("CAM1_SEQ_GAIN", 0.35)             # penurunan galat minimal 3 segmen vs 2 segmen
        # [PATCH-7] Loading SESUDAH Unloading yang skor baknya hampir tidak naik (bak pickup berisi sedikit / tertutup orang & forklift).
        self.refill_enable = e("CAM1_REFILL_ENABLE", 1, int)  # 1 = aktifkan deteksi "bak kosong -> forklift kembali -> skor bak ikut naik"
        self.refill_gap = e("CAM1_REFILL_GAP", 60.0)           # jeda minimal tanpa forklift setelah bak kosong sebelum forklift dianggap "kembali" (dtk)
        self.refill_fk_min = e("CAM1_REFILL_FK_MIN", 20.0)     # forklift harus ada di dekat kendaraan minimal sekian dtk setelah kembali
        self.refill_min_base = e("CAM1_REFILL_MIN_BASE", 8, int)   # sampel bak kosong (pembanding) minimal
        self.refill_min_late = e("CAM1_REFILL_MIN_LATE", 12, int)  # sampel sesudah forklift kembali minimal
        self.refill_frac = e("CAM1_REFILL_FRAC", 0.30)         # minimal porsi sampel akhir yang melewati p90 bak kosong (acak murni ~0.10)
        self.refill_rise = e("CAM1_REFILL_RISE", 0.05)         # minimal kenaikan p90 skor (akhir - kosong)
        self.fragment_contain = e("CAM1_FRAGMENT_CONTAIN", 0.60)   # kandidat baru yang berada >= segini di dalam sesi terbuka = fragmen, bukan kendaraan baru
        self.drop_noevidence = e("CAM1_DROP_NOEVIDENCE", 1, int)   # 1 = sesi tanpa satu pun sampel bak & tanpa skor forklift tidak dikirim SIKLUS_SELESAI
        self.force_decision = e("CAM1_FORCE_DECISION", 0, int)      # 1 = sesi yang sudah MASUK SELALU berhasil Loading/Unloading (tebakan terbaik + perlu_review), tidak pernah "Tidak Dapat Dipastikan"
        self.manual_min = e("CAM1_MANUAL_MIN", 20, int)
        self.manual_default = os.environ.get("CAM1_MANUAL_DEFAULT", "Unloading").strip().capitalize()
        self.force_default = os.environ.get("CAM1_FORCE_DEFAULT", "Loading").strip().capitalize()   # pilihan terakhir bila tak ada bukti apa pun (default: lokasi "Loading Kiri")
        self.guess_min = e("CAM1_GUESS_MIN", 0.20)             # keyakinan minimal agar arah terbaik dilaporkan (dengan perlu_review) alih-alih Tidak Dapat Dipastikan
        self.handover_gap = e("CAM1_HANDOVER_GAP", 30.0)        # kendaraan hilang lalu muncul di tempat LAIN dalam jeda ini + tampilan mirip = kendaraan yang sama pindah posisi (0 = mati)
        self.handover_max_speed = e("CAM1_HANDOVER_MAX_SPEED", 0.15)   # [FIX-SERAH] lompatan maks (lebar ROI per dtk); lebih cepat = bukan kendaraan yang sama
        # [FIX-UKURAN] sesi tidak boleh menyambung ke kotak yang luasnya >N kali / <1/N kali kotak acuannya (parkir terakhir). Mencegah sesi
        # "mengejar" benda besar/kecil lain (forklift berpalet dekat kamera, kabin saja) setelah kendaraan aslinya pergi/tertutup. 0 = mati.
        self.match_size_ratio = e("CAM1_MATCH_SIZE_RATIO", 3.0)
        # Perpindahan posisi bisa mengubah skala kotak secara drastis; handover tetap dibatasi
        # oleh jeda, kecepatan perpindahan, dan kemiripan tampilan di bawah.
        self.handover_size_ratio = e("CAM1_HANDOVER_SIZE_RATIO", 15.0)
        # [FIX-FRAGMEN] truk sudah parkir & terkonfirmasi: kotak KECIL (luas < frag_ratio x kotak truk) yang berada di DALAM kotak truknya
        # (>= fragment_contain, mis. kardus di bak) diabaikan, tidak disambung ke sesi. Tanpa ini kotak sesi menyusut ke kardus itu, truk aslinya
        # ditolak FIX-UKURAN dan terbuka sesi baru. 0 = mati.
        self.frag_ratio = e("CAM1_FRAG_RATIO", 0.5)
        # [FIX-PERGI] kendaraan terkonfirmasi yang terlihat menjauh dari titik parkir dan mengecil sampai <= depart_ratio x luas parkirnya, lalu
        # tidak terdeteksi > depart_grace dtk = sudah pergi: sesi langsung ditutup (KELUAR), tidak menunggu parked_grace (600 dtk) di mana
        # truk lain bisa tersambung ke sesi lama. depart_ratio 0 = mati.
        self.depart_ratio = e("CAM1_DEPART_RATIO", 0.25)
        self.depart_grace = e("CAM1_DEPART_GRACE", 12.0)
        # [PRESENSI] kendaraan parkir yang tidak lagi terdeteksi YOLO (conf rendah, pintu dibuka, orang/forklift menutupi) TIDAK langsung dianggap keluar:
        # bila area parkirnya masih tampak sama (korelasi tekstur >= presence_ncc DAN jarak warna <= presence_dist) sesi ditahan. 0 = mati (perilaku lama).
        self.presence = e("CAM1_PRESENCE", 1, int)
        self.presence_interval = e("CAM1_PRESENCE_INTERVAL", 4.0)      # dtk antar pemeriksaan visual
        self.presence_ncc = e("CAM1_PRESENCE_NCC", 0.60)               # korelasi (0..1) antara area parkir sekarang dan acuan terakhir
        self.presence_dist = e("CAM1_PRESENCE_DIST", 1.0)              # jarak histogram warna maks (0 identik, 1 beda total); 1.0 = tidak dipakai (cahaya berubah menggeser warna), tetap dicatat di log
        self.presence_chain = e("CAM1_PRESENCE_CHAIN", 0.80)           # [PRESENSI-BERANTAI] korelasi antar-pemeriksaan (4 dtk); perubahan PELAN (pintu/cahaya) = masih ada, 0 = mati
        self.presence_max_hold = e("CAM1_PRESENCE_MAX_HOLD", 3600.0)   # batas tahan (dtk) sejak deteksi terakhir
        # [PRESENSI-TUTUP] area parkir yang TERTUTUP forklift (korelasi turun karena ada forklift di depan kendaraan) bukan bukti kendaraan pergi:
        # pemeriksaan dianggap tak meyakinkan dan sesi ditahan paling lama presence_occl dtk berturut-turut. 0 = mati (perilaku lama).
        self.presence_occl = e("CAM1_PRESENCE_OCCL", 0.0)
        self.presence_occl_cover = e("CAM1_PRESENCE_OCCL_COVER", 0.15)  # bagian area parkir yang tertutup kotak forklift (0..1) agar dianggap tertutup
        self.presence_occl_cover_max = e("CAM1_PRESENCE_OCCL_COVER_MAX", 0.60)  # tertutup >= ini = "kotak itu forklift-nya sendiri", bukan kendaraan yang tertutup -> tidak ditahan
        # [FIX-PEMILIK-FK] 1 = forklift dimiliki sesi yang kotaknya PALING DEKAT dengan pusat forklift (bukan yang tumpang-tindihnya terbesar)
        self.fk_owner_nearest = e("CAM1_FK_OWNER_NEAREST", 1, int)
        # [FIX-UMUMKAN] kendaraan yang bak-nya terus tertutup orang/forklift (0 sampel) tetap boleh diumumkan bila deteksinya kuat berulang
        # (>= N deteksi conf >= confirm_conf) DAN sudah parkir diam >= park_confirm. Sebelumnya wajib hidup >= announce_wait (90 dtk). 0 = mati.
        self.announce_strong = e("CAM1_ANNOUNCE_STRONG", 10, int)
        self.handover_dist = e("CAM1_HANDOVER_DIST", 0.60)      # jarak tampilan (histogram HSV) maksimum untuk serah-terima sesi
        self.review_min = e("CAM1_REVIEW_MIN", 0.30)
        # [PATCH-1] ingatan forklift antar frame: kotak kendaraan yang tak bertumpuk dengan forklift yang terlihat <= detik ini lalu
        # TIDAK boleh membuka sesi baru (forklift berpalet sering dibaca 'truck' pada frame yang model forkliftnya kebetulan meleset). 0 = mati.
        self.fk_mem_sec = e("CAM1_FK_MEM_SEC", 12.0)
        # [PATCH-2] hasil dipaksa perlu_review bila: sampel bak terakhir lebih tua dari end_gap sebelum sesi ditutup, ada lubang sampel
        # lebih panjang dari hole_gap, atau (bila poligon dock ada) kendaraan tak pernah parkir di dock. 0 = mati.
        self.end_gap = e("CAM1_END_GAP", 60.0)
        self.hole_gap = e("CAM1_HOLE_GAP", 180.0)
        # [PATCH-3] dua fase (Unloading+Loading) dari suara lokasi tumpukan forklift berurutan waktu. Butuh poligon 'stage'.
        self.fk_seq_enable = e("CAM1_FK_SEQ_ENABLE", 1, int)
        self.fk_seq_min = e("CAM1_FK_SEQ_MIN", 2, int)       # suara minimal per fase
        # [PATCH-4] garis waktu per sesi di log (TIMELINE) untuk diagnosis. 0 = mati.
        self.tl_log = e("CAM1_TIMELINE", 1, int)
        self.tl_interval = e("CAM1_TIMELINE_INTERVAL", 3.0)
        self.llr_bed = e("CAM1_LLR_BED", 6.0)                # bobot selisih skor bak (selisih 0.30 ~ confidence 0.72)
        self.fk_llr = e("CAM1_FK_LLR", 1.1)                  # bobot arah forklift (1 kejadian ~ confidence 0.50)
        self.arrival_prior = e("CAM1_ARRIVAL_PRIOR", 0.0)    # tiba kosong + forklift aktif -> condong Loading
        # 1 = bila arah HANYA didukung asumsi tiba-kosong/penuh + forklift aktif (tanpa selisih skor bak yang berarti dan tanpa
        # arah dari gerakan forklift), hasil tetap dilaporkan tetapi ditandai perlu_review=true. 0 = perilaku lama.
        self.prior_review = e("CAM1_PRIOR_REVIEW", 1, int)
        # 1 = orang hanya mengganggu bila berdiri di area BAK (CAM1_BED_BOX diperluas 10%), bukan di mana saja di kotak kendaraan
        # (penjaga yang berdiri di samping kabin tidak lagi membuang sampel). 0 = perilaku lama.
        self.person_bed_only = e("CAM1_PERSON_BED_ONLY", 0, int)
        # [FIX-LOADING-ORANG] margin zona bak untuk cek "orang di bak" (pecahan lebar/tinggi bak). Bawaan 0.10 = perilaku lama. Saat loading
        # pickup, orang berdiri di tepi bak terus-menerus -> 0 sampel. Nilai 0 atau negatif (mis. -0.15) mengecilkan zona ke bagian dalam bak.
        self.person_bed_pad = e("CAM1_PERSON_BED_PAD", 0.10)
        # [DATA-ORANG] Pengumpul data PASIF: simpan crop orang yang berada di dekat kendaraan (bahan melatih model "orang membawa barang").
        # TIDAK memengaruhi keputusan Loading/Unloading. Kosong (CAM1_PERSON_DIR=) = mati.
        self.person_dir = os.environ.get("CAM1_PERSON_DIR", "person_crops").strip().strip('"')
        self.person_interval = e("CAM1_PERSON_INTERVAL", 1.5)         # jeda minimal antar penyimpanan per sesi (dtk)
        self.person_max = e("CAM1_PERSON_MAX", 250, int)              # batas jumlah crop per sesi
        self.person_near = e("CAM1_PERSON_NEAR", 0.8)                 # orang dihitung "dekat" bila pusatnya di dalam kotak kendaraan diperluas sekian x ukurannya
        self.person_min_side = e("CAM1_PERSON_MIN_SIDE", 40, int)     # sisi terpendek crop (px); lebih kecil dilewati
        # [ARAH-ORANG] arah Loading/Unloading dari ORANG yang membawa barang (kendaraan tertutup/van tanpa forklift).
        # Hanya MENGISI kekosongan: dipakai bila keputusan lain = "Tidak Dapat Dipastikan". 0 = mati.
        self.pc_enable = e("CAM1_PC_ENABLE", 1, int)
        # [PC-SETELAH-FK] arah-orang juga dicoba bila forklift SUNGGUHAN ada tetapi jalur forklift tidak memberi arah (tanpa transisi bawa/kosong). 0 = mati.
        self.pc_after_fk = e("CAM1_PC_AFTER_FK", 0, int)
        # [FIX-FK-HANTU-ORANG] forklift hantu sesaat (skor sedikit, < fk_near_min dtk) tidak boleh memblokir arah-dari-orang. 0 = perilaku lama.
        self.pc_fk_obs_max = e("CAM1_PC_FK_OBS_MAX", 3, int)
        self.pc_conf = e("CAM1_PC_CONF", 0.30)                       # conf YOLO minimal untuk orang yang dinilai
        self.pc_box = e("CAM1_PC_BOX", 1.0)                          # orang dinilai bila pusatnya di kotak kendaraan diperluas sekian x
        self.pc_interval = e("CAM1_PC_INTERVAL", 0.8)                # jeda minimal antar skor per orang (dtk)
        self.pc_link_gap = e("CAM1_PC_LINK_GAP", 9.0)                # orang hilang <= ini masih track yang sama
        self.pc_link_dist = e("CAM1_PC_LINK_DIST", 1.2)              # jarak pusat maks (x ukuran kotak orang) untuk menyambung track
        self.pc_hi = e("CAM1_PC_HI", 0.60)                           # P(bawa) >= ini = bawa
        self.pc_lo = e("CAM1_PC_LO", 0.40)                           # P(bawa) <= ini = kosong
        self.pc_min_run = e("CAM1_PC_MIN_RUN", 2, int)               # status harus bertahan segini skor berturut-turut
        self.pc_move = e("CAM1_PC_MOVE", 0.12)                       # selisih jarak ke kendaraan (x ukuran kendaraan) agar langkah membawa dianggap 'mendekat/menjauh'
        self.pc_camera_move = e("CAM1_PC_CAMERA_MOVE", 0.035)        # perpindahan minimal (koordinat frame) sepanjang sumbu kendaraan<->stage
        self.pc_near = e("CAM1_PC_NEAR", 0.45)                       # jarak <= ini = di kendaraan
        self.pc_far = e("CAM1_PC_FAR", 0.60)                         # jarak >= ini = di tumpukan/dock
        self.pc_win = e("CAM1_PC_WIN", 6.0)                          # jendela sebelum/sesudah membawa untuk bukti lintasan stage saja
        self.pc_dist = e("CAM1_PC_DIST", 1, int)                     # [POLA-JARAK] arah dari pola jarak orang ke kendaraan bila skor bawa tak memberi arah; 0 = mati
        self.pcd_min_obs = e("CAM1_PCD_MIN_OBS", 100, int)           # pengamatan orang minimal agar pola dipercaya
        self.pcd_near = e("CAM1_PCD_NEAR", 0.35)                     # jarak (x ukuran kendaraan) <= ini = orang sampai di kendaraan/bak
        self.pcd_load_frac = e("CAM1_PCD_LOAD_FRAC", 0.15)           # Loading bila >= sekian dari pengamatan orang berada dekat kendaraan ...
        self.pcd_load_people = e("CAM1_PCD_LOAD_PEOPLE", 3, int)     # ... dan >= sekian orang berbeda pernah sampai di kendaraan
        self.pcd_load_stage = e("CAM1_PCD_LOAD_STAGE", 0.10)          # Loading juga wajib ada orang di stage (kerja selalu diawali dari stage) >= sekian pengamatan
        self.pcd_unl_frac = e("CAM1_PCD_UNL_FRAC", 0.05)             # Unloading bila <= sekian pengamatan dekat kendaraan ...
        self.pcd_unl_people = e("CAM1_PCD_UNL_PEOPLE", 1, int)       # ... <= sekian orang pernah sampai di kendaraan ...
        self.pcd_unl_stage = e("CAM1_PCD_UNL_STAGE", 0.40)           # ... dan >= sekian pengamatan orang berada di stage
        self.pc_min_votes = e("CAM1_PC_MIN_VOTES", 2, int)
        # [FIX-MARGIN] arah dari orang hanya diputuskan bila selisih suara cukup tegas: |L-U|/(L+U) >= nilai ini. 2 lawan 1 (0.33) = terlalu tipis
        # (kasus 13 Nov 12:12: salah Loading) -> dibiarkan 'Tidak Dapat Dipastikan'. 0 = mati (perilaku lama).
        self.pc_min_agree = e("CAM1_PC_MIN_AGREE", 0.40)           # suara minimal agar arah orang dipakai
        self.pc_stage_min_sec = e("CAM1_PC_STAGE_MIN_SEC", 30.0)  # fallback arah stage perlu satu aksi Unloading + aktivitas stage berkelanjutan
        # pelabelan otomatis forklift dari sesi bak-terbuka yang yakin (kosong = mati)
        self.fk_autolabel_dir = os.environ.get("CAM1_FK_AUTOLABEL_DIR", "").strip().strip('"')
        self.fk_autolabel_conf = e("CAM1_FK_AUTOLABEL_CONF", 0.80)   # confidence DARI SKOR BAK saja
        self.fk_touch_margin = e("CAM1_FK_TOUCH_MARGIN", 3.0)        # detik di sekitar momen menyentuh = dibuang (ambigu)
        self.fk_autolabel_min_side = e("CAM1_FK_AUTOLABEL_MIN_SIDE", 3, int)
        self.fk_autolabel_max = e("CAM1_FK_AUTOLABEL_MAX", 40, int)  # per kelas per sesi
        self.photo_interval = e("CAM1_PHOTO_INTERVAL", 1.0)
        self.fk_trans_dir = os.environ.get("CAM1_FK_TRANS_DIR", "").strip().strip('"')   # simpan foto PENUH saat forklift menaruh/mengambil (diagnosis); kosong = mati
        self.sample_dir = os.environ.get("CAM1_SAMPLE_DIR", "").strip().strip('"')   # simpan potongan bak tiap sampel (bahan latih ulang); kosong = mati

        # n8n
        self.n8n_timeout = e("N8N_TIMEOUT_SECONDS", 10.0)
        self.n8n_retries = e("N8N_RETRY_ATTEMPTS", 3, int)
        self.n8n_retry_delay = e("N8N_RETRY_DELAY_SECONDS", 3.0)
        self.outbox_interval = e("N8N_OUTBOX_INTERVAL_SECONDS", 30.0)
        self.outbox_max_mb = e("N8N_OUTBOX_MAX_MB", 500.0)


# ----------------------------------------------------------------------------------------------
# Geometri & zona
# ----------------------------------------------------------------------------------------------
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def point_box_dist_norm(point, box):
    """Jarak titik ke tepi kotak, dinormalisasi dengan sisi terpanjang kotak."""
    px, py = point
    dx = max(box[0] - px, 0.0, px - box[2])
    dy = max(box[1] - py, 0.0, py - box[3])
    scale = max(1.0, box[2] - box[0], box[3] - box[1])
    return math.hypot(dx, dy) / scale


class Zones:
    """Poligon dalam koordinat normal; di-bind ke ukuran frame saat frame pertama datang."""

    def __init__(self, data):
        self.data = {"area": data.get("area") or [], "exclude": data.get("exclude") or [], "dock": data.get("dock") or [],
                     "stage": data.get("stage") or [], "edge": data.get("edge") or [],
                     # Titik tetap di area pintu tujuan Unloading (koordinat normal 0..1).
                     "unload_target": data.get("unload_target")}
        self.w = self.h = 0
        self._area = self._excl = self._dock = self._stage = self._edge = None
        self.roi = (0, 0, 1, 1)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def bind(self, w, h):
        if (w, h) == (self.w, self.h):
            return
        self.w, self.h = w, h

        def px(poly):
            return np.array([[x * w, y * h] for x, y in poly], dtype=np.float32).reshape(-1, 1, 2)

        self._area = px(self.data["area"]) if len(self.data["area"]) >= 3 else None
        self._excl = [px(p) for p in self.data["exclude"] if len(p) >= 3]
        self._dock = px(self.data["dock"]) if len(self.data["dock"]) >= 3 else None
        self._stage = px(self.data["stage"]) if len(self.data["stage"]) >= 3 else None
        self._edge = px(self.data["edge"]) if len(self.data["edge"]) >= 3 else None
        if self._area is not None:
            x, y, bw, bh = cv2.boundingRect(self._area.astype(np.int32))
            self.roi = (max(0, x), max(0, y), min(w, x + bw), min(h, y + bh))
        else:
            self.roi = (0, 0, w, h)

    @staticmethod
    def _in(poly, x, y):
        return poly is not None and cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def in_area(self, x, y):
        if self._area is not None and not self._in(self._area, x, y):
            return False
        return not any(self._in(p, x, y) for p in self._excl)

    def near_area(self, x, y, margin_px):
        """[KENDARAAN-TEPI] True bila titik di luar poligon area tapi dalam jarak margin_px darinya DAN bukan di zona exclude."""
        if self._area is None:
            return False
        if any(self._in(p, x, y) for p in self._excl):
            return False
        if self._in(self._edge, x, y):                  # poligon "edge" di cam1_zones.json (pinggir yang digambar sendiri)
            return True
        return margin_px > 0 and cv2.pointPolygonTest(self._area, (float(x), float(y)), True) >= -float(margin_px)

    def in_dock(self, x, y):
        return self._in(self._dock, x, y)

    def has_stage(self):
        return len(self.data["stage"]) >= 3

    def in_stage(self, x, y):
        """Area tumpukan palet (sumber/tujuan barang di sisi gudang). Titik = pusat kotak forklift."""
        return self._in(self._stage, x, y)

    @property
    def roi_area(self):
        return max(1, (self.roi[2] - self.roi[0]) * (self.roi[3] - self.roi[1]))

    def draw(self, img):
        def pts(poly):
            return np.array([[x * self.w, y * self.h] for x, y in poly], dtype=np.int32)

        ov = img.copy()
        if len(self.data["area"]) >= 3:
            cv2.polylines(img, [pts(self.data["area"])], True, (0, 220, 0), 3)
        if len(self.data["dock"]) >= 3:
            cv2.fillPoly(ov, [pts(self.data["dock"])], (0, 140, 255))
            cv2.polylines(img, [pts(self.data["dock"])], True, (0, 140, 255), 3)
        if len(self.data["edge"]) >= 3:
            cv2.polylines(img, [pts(self.data["edge"])], True, (0, 255, 255), 2)
        if len(self.data["stage"]) >= 3:
            cv2.fillPoly(ov, [pts(self.data["stage"])], (255, 120, 0))
            cv2.polylines(img, [pts(self.data["stage"])], True, (255, 120, 0), 3)
        for p in self.data["exclude"]:
            if len(p) >= 3:
                cv2.fillPoly(ov, [pts(p)], (0, 0, 255))
                cv2.polylines(img, [pts(p)], True, (0, 0, 255), 2)
        return cv2.addWeighted(ov, 0.25, img, 0.75, 0)


# ----------------------------------------------------------------------------------------------
# Util gambar
# ----------------------------------------------------------------------------------------------
def orange_pct(crop):
    """Persentase piksel oranye (warna khas forklift). Sama dengan yang dipakai sistem lama."""
    if crop is None or crop.size == 0:
        return 0.0
    small = cv2.resize(crop, (160, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, (3, 140, 120), (20, 255, 255)).mean()) / 255.0 * 100.0


def green_pct(crop):
    """[FK-HIJAU] Persentase piksel hijau-tosca (forklift sewaan non-oranye). Hue 74-94 (biru kerucut ~105 dan kontainer 0% tidak ikut)."""
    if crop is None or crop.size == 0:
        return 0.0
    small = cv2.resize(crop, (160, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, (74, 90, 90), (94, 255, 255)).mean()) / 255.0 * 100.0


def clip_box(b, w, h, pad=0.0):
    bw, bh = b[2] - b[0], b[3] - b[1]
    x1, y1 = int(max(0, b[0] - bw * pad)), int(max(0, b[1] - bh * pad))
    x2, y2 = int(min(w, b[2] + bw * pad)), int(min(h, b[3] + bh * pad))
    return x1, y1, x2, y2


def crop_vehicle(frame, bbox, pad=0.0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(bbox, w, h, pad)
    return frame[y1:y2, x1:x2]


def _bed_box_from_env():
    """Pecahan (x0, y0, x1, y1) dari kotak kendaraan yang dipotong sebagai BAK. Bawaan = geometri lama.
    Ubah lewat CAM1_BED_BOX="x0,y0,x1,y1" (mis. "0.0,0.0,0.52,0.60" untuk truk bak terbuka berkabin di kanan).
    PENTING: cargo_model.pkl harus dilatih dengan potongan yang SAMA (kumpulkan ulang lewat CAM1_SAMPLE_DIR, label, latih)."""
    d = (0.2, 0.3, 0.95, 0.85)
    raw = os.environ.get("CAM1_BED_BOX", "").strip().strip('"')
    if not raw:
        return d
    try:
        v = tuple(float(x) for x in raw.split(","))
        if len(v) == 4 and 0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1 and (v[2] - v[0]) >= 0.15 and (v[3] - v[1]) >= 0.15:
            return v
    except ValueError:
        pass
    return d


BED_BOX = _bed_box_from_env()


def cargo_bed(truck_crop):
    """Potongan bak (geometri: BED_BOX). HARUS sama dengan yang dipakai saat training cargo_model.pkl (distribusi data harus cocok)."""
    h, w = truck_crop.shape[:2]
    if h < 20 or w < 20:
        return truck_crop
    x0, y0, x1, y1 = BED_BOX
    bed = truck_crop[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    return bed if bed.size >= 1000 else truck_crop


def to_jpg(img, max_w=800, quality=82):
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def b64(jpg):
    return base64.b64encode(jpg).decode("ascii") if jpg else ""


def fmt_ts(t):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------------------------
# Penentu Loading / Unloading (fungsi murni -> mudah diuji)
# ----------------------------------------------------------------------------------------------
def foto_arah(res, cfg):
    """[FOTO-AWAL-AKHIR] Loading bila tiba KOSONG lalu pergi BERMUATAN, Unloading bila sebaliknya; selain itu None.
    Memakai skor awal/akhir yang sama dengan label status_muatan_awal/akhir (p_to_status), plus selisih minimal."""
    if cfg.foto_fb <= 0:
        return None
    e, l = res.get("early"), res.get("late")
    if e is None or l is None:
        return None
    if e <= cfg.foto_fb_empty and l >= cfg.foto_fb_loaded and (l - e) >= cfg.foto_fb_delta:
        return LOADING
    if e >= cfg.foto_fb_loaded and l <= cfg.foto_fb_empty and (e - l) >= cfg.foto_fb_delta:
        return UNLOADING
    return None


def pc_allowed(res, fk_seconds, fk_obs_n, cfg):
    """Apakah jalur arah-orang boleh dicoba saat sesi ditutup. Jalur lama: keputusan belum ada, forklift tidak aktif lama dan hanya sedikit skor forklift.
    [PC-SETELAH-FK] (opsional): forklift sungguhan ada, tetapi tidak memberi arah (tidak ada bukti 'forklift_arah') -> arah-orang tetap dicoba."""
    if not cfg.pc_enable or res.get("status") not in (None, UNSURE):
        return False
    if fk_seconds < cfg.fk_near_min and fk_obs_n <= cfg.pc_fk_obs_max:
        return True
    return bool(cfg.pc_after_fk and "forklift_arah" not in (res.get("terms") or {}))


def p_to_status(p):
    return None if p is None else ("Bermuatan" if p >= 0.5 else "Kosong")


def _fk_runs(obs, cfg, relaxed=False):
    """Seperti _fk_states, tapi tiap status membawa rentang indeks obs: [(status, i_awal, i_akhir)]."""
    ps = [p for _, p in obs]
    if len(ps) < cfg.fk_min_obs:
        return []
    sm = [float(median(ps[max(0, i - 1):i + 2])) for i in range(len(ps))]   # median bergeser 3 titik
    hi = max(0.65, cfg.fk_hi - 0.1) if relaxed else cfg.fk_hi
    lo = min(0.35, cfg.fk_lo + 0.1) if relaxed else cfg.fk_lo
    min_run = max(2, cfg.fk_min_run - 1) if relaxed else cfg.fk_min_run
    state, seq = None, []
    for i, v in enumerate(sm):                                               # histeresis hi/lo
        if v >= hi:
            state = True
        elif v <= lo:
            state = False
        if state is not None:
            seq.append((state, i))
    runs = []                                                                # [status, panjang, i_awal, i_akhir]
    for st, i in seq:
        if runs and runs[-1][0] == st:
            runs[-1][1] += 1
            runs[-1][3] = i
        else:
            runs.append([st, 1, i, i])
    runs = [r for r in runs if r[1] >= min_run]                              # buang kedipan pendek
    out = []
    for st, _, i0, i1 in runs:
        if out and out[-1][0] == st:
            out[-1] = (st, out[-1][1], i1)
        else:
            out.append((st, i0, i1))
    return out


def _fk_states(obs, cfg):
    """obs: [(t, p_bawa)] satu track forklift -> daftar status berurutan (True=bawa, False=kosong) setelah dihaluskan."""
    return [r[0] for r in _fk_runs(obs, cfg)]


def _resolve_stage(at, loc, cfg):
    """Lokasi satu transisi forklift -> True (di tumpukan) / False (di kendaraan) / None (tidak bisa dipastikan).
    Poligon 'stage' (bila dipakai & berisi titik) menang untuk True; selain itu JARAK ke kendaraan menentukan:
    >= fk_far = tumpukan, <= fk_near = kendaraan, di antaranya = None (tidak bersuara)."""
    if at:
        return True
    if getattr(cfg, "fk_loc", 1) and loc is not None:
        if loc >= cfg.fk_far:
            return True
        if loc <= cfg.fk_near:
            return False
        return None
    return at                                   # False (poligon) atau None


def fk_direction(tracks, cfg, dists=None, stages=None):
    """
    tracks: list track forklift, tiap track = list [(t, p_bawa)].
    dists (opsional): list sejajar dengan tracks; tiap elemen = jarak forklift ke pusat kendaraan (x ukuran kendaraan)
                      per obs. Bila ada, LOKASI transisi ikut dipakai: menaruh barang di truk terjadi DEKAT truk,
                      mengambil dari area penumpukan terjadi JAUH dari truk (dan sebaliknya saat Unloading).
    TARUH (bawa -> kosong) = barang masuk ke kendaraan = LOADING. AMBIL (kosong -> bawa) = UNLOADING.
    Return dict(status|None, drops, picks, review, confidence, reason, eff_drops, eff_picks, geo, geo_w).
    drops/picks = hitungan mentah; eff_* = hitungan setelah mempertimbangkan lokasi (dipakai decide()).
    """
    drops = picks = 0
    d_drop, d_pick = [], []
    stg_flags = []                               # (jenis, di_tumpukan True/False/None) per transisi
    trans = []                                   # (jenis, indeks track, jarak ke truk | None) per transisi, untuk log
    events = []                                  # [PATCH-3] (waktu, jenis, di_tumpukan, jarak, indeks track) per transisi
    relaxed_tracks = set()
    for ti, obs in enumerate(tracks or []):
        runs = _fk_runs(obs, cfg)
        if len(runs) < 2:
            relaxed_runs = _fk_runs(obs, cfg, relaxed=True)
            if len(relaxed_runs) > len(runs):
                runs = relaxed_runs
                relaxed_tracks.add(ti)
        dd = dists[ti] if dists is not None and ti < len(dists) else None
        sg = stages[ti] if stages is not None and ti < len(stages) else None
        for r0, r1 in zip(runs, runs[1:]):
            a, b = r0[0], r1[0]
            if not (a != b):
                continue
            loc = None
            if dd is not None and len(dd) == len(obs):
                seg = dd[r0[2]:r1[1] + 1]            # dari akhir status lama sampai awal status baru
                if seg:
                    loc = sum(seg) / len(seg)
            at = None
            if getattr(cfg, "fk_stage", 0) and sg is not None and len(sg) == len(obs) and all(v is not None for v in sg):
                seg_s = sg[r0[2]:r1[1] + 1]
                if seg_s:
                    at = sum(1 for v in seg_s if v) * 2 >= len(seg_s)    # mayoritas titik di tumpukan
            at = _resolve_stage(at, loc, cfg)
            stg_flags.append(("taruh" if (a and not b) else "ambil", at))
            try:
                events.append((0.5 * (obs[r0[2]][0] + obs[r1[1]][0]), "taruh" if (a and not b) else "ambil", at, loc, ti))
            except Exception:
                pass
            if a and not b:
                drops += 1
                trans.append(("taruh", ti, None if loc is None else round(loc, 2)))
                if loc is not None:
                    d_drop.append(loc)
            elif b and not a:
                picks += 1
                trans.append(("ambil", ti, None if loc is None else round(loc, 2)))
                if loc is not None:
                    d_pick.append(loc)
    total = drops + picks
    res = dict(status=None, drops=drops, picks=picks, review=True, confidence=0.0, reason="",
               eff_drops=drops, eff_picks=picks, geo=None, geo_w=1.0, trans=trans, stage=stg_flags, stage_votes=None,
               events=events, relaxed=bool(relaxed_tracks))
    if total == 0:
        res["reason"] = "tidak ada transisi bawa/kosong pada forklift"
        return res
    # LOKASI TUMPUKAN (poligon "stage"): barang mengalir tumpukan -> kendaraan = Loading; kendaraan -> tumpukan = Unloading.
    #   ambil DI tumpukan / taruh DI LUAR tumpukan  = suara Loading;  taruh DI tumpukan / ambil DI LUAR tumpukan = suara Unloading.
    dec = [(k, at) for k, at in stg_flags if at is not None]      # hanya transisi yang lokasinya jelas yang bersuara
    if dec:
        L = sum(1 for k, at in dec if (k == "ambil") == at)
        U = len(dec) - L
        res["stage_votes"] = (L, U)
        res.update(eff_drops=L, eff_picks=U)
        if L == U:
            res["reason"] = f"forklift: lokasi tumpukan seimbang ({L} suara Loading vs {U} Unloading)"
            return res
        n_ = L + U
        agree = abs(L - U) / n_
        res.update(status=LOADING if L > U else UNLOADING,
                   confidence=round(min(1.0, agree * min(1.0, n_ / 3.0)), 2),
                   review=not (n_ >= 2 and agree >= 0.75),
                   reason=(f"forklift: dari lokasi (jauh=tumpukan, dekat=kendaraan) {L} suara Loading vs {U} Unloading "
                           f"({len(stg_flags) - len(dec)} transisi lokasinya ambigu diabaikan); {drops} taruh/{picks} ambil"))
        return res
    # lokasi: taruh paling dekat truk & ambil paling jauh = Loading (geo > 0); kebalikannya = Unloading
    if getattr(cfg, "fk_geo", 1) and drops > 0 and picks > 0 and len(d_drop) == drops and len(d_pick) == picks:
        geo = sum(d_pick) / picks - sum(d_drop) / drops
        res["geo"] = round(geo, 2)
        m = max(1e-6, cfg.fk_geo_margin)
        if abs(geo) >= m:
            loading = geo > 0
            n_ = drops if loading else picks
            res.update(eff_drops=drops if loading else 0, eff_picks=0 if loading else picks,
                       geo_w=min(1.0, abs(geo) / (2.0 * m)))
            res.update(status=LOADING if loading else UNLOADING,
                       confidence=round(min(1.0, res["geo_w"] * min(1.0, n_ / 3.0)), 2),
                       review=not (n_ >= 2 and res["geo_w"] >= 0.75),
                       reason=(f"forklift: {drops} taruh vs {picks} ambil, tapi lokasi menentukan: "
                               f"{'menaruh' if loading else 'mengambil'} terjadi dekat truk (selisih jarak {geo:+.2f})"))
            return res
    if drops == picks:
        res["reason"] = f"forklift: {drops} taruh vs {picks} ambil (seimbang)" + (
            f", lokasi tidak membedakan (selisih jarak {res['geo']:+.2f})" if res["geo"] is not None else "")
        if relaxed_tracks:
            res["reason"] += "; terdeteksi dengan ambang adaptif (bukti lemah)"
        return res
    agree = abs(drops - picks) / total
    res.update(status=LOADING if drops > picks else UNLOADING,
               confidence=round(min(1.0, agree * min(1.0, total / 3.0)), 2),
               review=bool(relaxed_tracks) or not (total >= 2 and agree >= 0.75),
               reason=f"forklift: {drops} kali menaruh barang, {picks} kali mengambil barang" +
                      ("; ambang adaptif, perlu review" if relaxed_tracks else ""))
    return res


def _pc_runs(obs, hi, lo, min_run):
    """obs [(t, p_bawa)] -> [(True=bawa/False=kosong, i_awal, i_akhir)] setelah median 3 titik + histeresis + buang kedipan."""
    ps = [p for _, p in obs]
    if len(ps) < 3:
        return []
    sm = [float(median(ps[max(0, i - 1):i + 2])) for i in range(len(ps))]
    state, runs = None, []
    for i, v in enumerate(sm):
        if v >= hi:
            state = True
        elif v <= lo:
            state = False
        if state is None:
            continue
        if runs and runs[-1][0] == state:
            runs[-1][1] += 1
            runs[-1][3] = i
        else:
            runs.append([state, 1, i, i])
    out = []
    for st, n_, i0, i1 in runs:
        if n_ < min_run:
            continue
        if out and out[-1][0] == st:
            out[-1] = (st, out[-1][1], i1)
        else:
            out.append((st, i0, i1))
    return out


def person_distance_pattern(tracks, cfg):
    """[POLA-JARAK] Arah dari POLA orang, tanpa skor bawa (bak tertutup / barang kecil / van yang diturunkan sopir):
    Loading  = banyak orang berkali-kali SAMPAI di kendaraan (menaruh barang ke bak).
    Unloading = orang gudang hanya menunggu di stage, hampir tidak ada yang sampai di kendaraan (barang diturunkan sopir ke stage).
    Pola ini LEMAH (dikalibrasi dari 2 sesi) -> hasilnya selalu perlu review. Mengembalikan (status|None, alasan)."""
    n = nn = stg = 0
    near_people = 0
    for tr in tracks:
        d, sg = tr.get("dist") or [], tr.get("stg") or []
        k = sum(1 for x in d if x <= cfg.pcd_near)
        n += len(d)
        nn += k
        stg += sum(1 for x in sg if x)
        near_people += 1 if k >= 2 else 0
    if n < cfg.pcd_min_obs:
        return None, f"pola jarak: pengamatan orang terlalu sedikit ({n} < {cfg.pcd_min_obs})"
    fn, fs = nn / n, stg / n
    info = (f"{nn}/{n} pengamatan orang dekat kendaraan ({100 * fn:.0f}%), {near_people} orang sampai di kendaraan, "
            f"{100 * fs:.0f}% pengamatan di stage")
    if fn >= cfg.pcd_load_frac and near_people >= cfg.pcd_load_people and fs >= cfg.pcd_load_stage:
        return LOADING, "pola jarak: banyak orang sampai di kendaraan -> perkiraan Loading (" + info + ")"
    if fn <= cfg.pcd_unl_frac and near_people <= cfg.pcd_unl_people and fs >= cfg.pcd_unl_stage:
        return UNLOADING, "pola jarak: orang menunggu di stage, hampir tidak ada yang sampai di kendaraan -> perkiraan Unloading (" + info + ")"
    return None, "pola jarak: tidak jelas (" + info + ")"


def person_direction(tracks, cfg, camera_axis=None):
    """[ARAH-ORANG] tracks: list dict(obs=[(t, p_bawa)], dist=[jarak orang ke tepi kendaraan, x ukuran kendaraan], stg=[...]).
    Model 'bawa' baru yakin SETELAH barang terlihat di tangan (sering sudah dekat tujuan), jadi lokasi AMBIL tidak dipercaya
    (selalu bias ke arah salah). Yang dipakai:
      A) PERJALANAN: bila sumbu kamera tersedia, vote arah dari proyeksi gerak kaki sepanjang
           kendaraan<->stage. Jika tidak, gunakan perubahan jarak ke tepi kendaraan. Perlintasan
           poligon stage menjadi bukti pendukung, bukan vote arah tersendiri.
      B) Arah diukur di dalam rangkaian deteksi membawa, lalu diringkas dengan median menjadi maksimal
           satu vote per track. Lokasi saat barang ditaruh hanya bukti pendukung."""
    L = U = 0
    stage_L = stage_U = 0              # dukungan arah dari perpindahan zona; bukan vote mandiri
    stage_drop_u = 0
    detail = []
    for ti, tr in enumerate(tracks or []):
        obs, dist, stg = tr.get("obs") or [], tr.get("dist") or [], tr.get("stg") or []
        pos = tr.get("pos") or []
        if len(dist) != len(obs):
            continue
        # Banyak rangkaian "bawa" dari orang yang sama bukan bukti independen.
        # Median perpindahan membuat setiap track memberi paling banyak satu vote.
        track_deltas = []

        def stage_state(start, end, include_end=True):
            if len(stg) != len(obs):
                return None
            values = [stg[k] for k, (t, _) in enumerate(obs)
                      if start <= t and (t <= end if include_end else t < end)
                      and stg[k] is not None]
            if len(values) < 2:          # [FIX-STAGE-1SAMPEL] satu sampel tidak cukup untuk menyatakan orang pindah zona
                return None
            return sum(bool(v) for v in values) * 2 >= len(values)

        runs = _pc_runs(obs, cfg.pc_hi, cfg.pc_lo, cfg.pc_min_run)
        for st, i0, i1 in runs:                                      # (A) perjalanan
            if not st:
                continue
            t0, t1 = obs[i0][0], obs[i1][0]
            # Arah diukur hanya selama model menyatakan orang sedang membawa barang.
            # Jendela setelah run bisa menangkap orang yang sudah menaruh barang lalu berjalan pergi.
            run_dist = dist[i0:i1 + 1]
            half = max(1, len(run_dist) // 2)
            d0 = float(median(run_dist[:half]))
            d1 = float(median(run_dist[-half:]))
            dd = d1 - d0
            direction_threshold = cfg.pc_move
            if camera_axis is not None and len(pos) == len(obs):
                ax, ay = camera_axis
                run_proj = [p[0] * ax + p[1] * ay for p in pos[i0:i1 + 1]]
                if run_proj:
                    dd = float(median(run_proj[-half:]) - median(run_proj[:half]))
                    direction_threshold = cfg.pc_camera_move
            pre = [dist[k] for k in range(len(obs)) if t0 - cfg.pc_win <= obs[k][0] < t0]
            post = [dist[k] for k in range(len(obs)) if t1 < obs[k][0] <= t1 + cfg.pc_win]
            pre_stage = stage_state(t0 - cfg.pc_win, t0, include_end=False)
            post_stage = stage_state(t1, t1 + cfg.pc_win)
            direction_basis = "sumbu kamera" if camera_axis is not None and len(pos) == len(obs) else "jarak kendaraan"
            if dd <= -direction_threshold:
                track_deltas.append(dd / direction_threshold)
                detail.append(f"orang{ti}: bawa t={t0:.0f}-{t1:.0f}, {direction_basis} {dd:+.3f} (kandidat Loading)")
            elif dd >= direction_threshold:
                track_deltas.append(dd / direction_threshold)
                detail.append(f"orang{ti}: bawa t={t0:.0f}-{t1:.0f}, {direction_basis} {dd:+.3f} (kandidat Unloading)")
            elif pre_stage is not None and post_stage is not None and pre_stage != post_stage:
                if pre_stage:
                    stage_L += 1
                    detail.append(f"orang{ti}: bawa menyeberang stage->kendaraan (pendukung Loading; jarak belum cukup berubah untuk vote)")
                else:
                    stage_U += 1
                    detail.append(f"orang{ti}: bawa menyeberang kendaraan->stage (pendukung Unloading; jarak belum cukup berubah untuk vote)")
            else:
                detail.append(f"orang{ti}: bawa t={t0:.0f}-{t1:.0f}, arah {direction_basis} {dd:+.3f} (gerak belum cukup) 0")
        for r0, r1 in zip(runs, runs[1:]):                           # (B) catat tempat taruh; jangan gandakan vote perjalanan
            if not (r0[0] and not r1[0]):
                continue
            seg = dist[r0[2]:r1[1] + 1]
            if not seg:
                continue
            loc = sum(seg) / len(seg)
            seg_stage = stg[r0[2]:r1[1] + 1] if len(stg) == len(obs) else []
            known_stage = [v for v in seg_stage if v is not None]
            at_stage = (sum(bool(v) for v in known_stage) * 2 >= len(known_stage)
                        if known_stage else None)
            if at_stage is True:
                stage_drop_u += 1
                detail.append("orang{}: menaruh di zona stage (bukti pendukung Unloading; tidak dihitung sebagai vote kedua)".format(ti))
            elif loc <= cfg.pc_near:
                detail.append(f"orang{ti}: menaruh dekat kendaraan (jarak {loc:.2f}; pendukung Loading, tanpa vote tambahan)")
            elif loc >= cfg.pc_far:
                detail.append(f"orang{ti}: menaruh jauh dari kendaraan (jarak {loc:.2f}; pendukung Unloading, tanpa vote tambahan)")
            else:
                detail.append(f"orang{ti}: lokasi menaruh di jarak {loc:.2f} (ambigu; tanpa vote tambahan)")
        if track_deltas:
            track_delta = float(median(track_deltas))
            if track_delta <= -1.0:
                L += 1
                detail.append(f"orang{ti}: vote track Loading (median skor {track_delta:+.2f}; ambang 1; {len(track_deltas)} rangkaian membawa)")
            elif track_delta >= 1.0:
                U += 1
                detail.append(f"orang{ti}: vote track Unloading (median skor {track_delta:+.2f}; ambang 1; {len(track_deltas)} rangkaian membawa)")
            else:
                detail.append(f"orang{ti}: rangkaian membawa saling berlawanan (median skor {track_delta:+.2f}); tanpa vote")
    res = dict(status=None, L=L, U=U, confidence=0.0, review=True, detail=detail,
               stage_drop_u=stage_drop_u,
               reason=f"orang membawa barang: {L} suara Loading vs {U} Unloading")
    n = L + U
    # Satu lintasan membawa yang jelas + lintasan stage searah boleh memberi label awal,
    # tetapi tetap wajib review. Lintasan stage sendiri tidak pernah cukup untuk menentukan arah.
    if L == 1 and U == 0 and stage_L >= 1 and stage_U == 0:
        res.update(status=LOADING, confidence=0.35, review=True)
        res["reason"] += f" + {stage_L} lintasan stage->kendaraan searah (perlu review)"
        return res
    if U == 1 and L == 0 and stage_U >= 1 and stage_L == 0:
        res.update(status=UNLOADING, confidence=0.35, review=True)
        res["reason"] += f" + {stage_U} lintasan kendaraan->stage searah (perlu review)"
        return res
    if L == U or n < cfg.pc_min_votes:      # bukti stage saja tidak melewati batas suara minimal
        return res
    agree = abs(L - U) / n
    if agree < getattr(cfg, "pc_min_agree", 0.0):       # [FIX-MARGIN] selisih suara terlalu tipis -> jangan menebak
        res["reason"] += f" (ditolak: selisih suara tipis {agree:.2f} < {cfg.pc_min_agree:.2f})"
        return res
    res.update(status=LOADING if L > U else UNLOADING,
               confidence=round(min(1.0, agree * min(1.0, n / 3.0)), 2),
               review=not (n >= 3 and agree >= 0.75))
    return res


def make_pc_scorer():
    """Skor 'orang membawa barang' (person_carry_model.pkl). None bila model belum ada -> arah dari orang MATI."""
    try:
        import person_carry
        if not person_carry.is_available():
            log.warning("person_carry_model.pkl tidak ditemukan -> arah dari orang MATI")
            return None
        return person_carry.carry_probability
    except Exception as e:
        import sys as _sys
        _pm = _sys.modules.get("person_carry")
        log.warning("person_carry tidak tersedia (%s) | berkas yang terimpor: %s | isi modul: %s", e,
                    getattr(_pm, "__file__", None) or getattr(_pm, "__path__", None) or "tidak ada",
                    [n for n in dir(_pm) if not n.startswith("__")][:12] if _pm is not None else "-")
        return None


def fk_trip_split(crops, margin=3.0, min_side=3):
    """
    crops: [(t, jarak_ke_pusat_kendaraan, jpg)] urut waktu, satu track forklift.
    Satu perjalanan bersih = jauh -> dekat (menyentuh/menaruh/mengambil) -> jauh. Return (sebelum, sesudah) atau None.
    'sebelum' = crop sebelum menyentuh kendaraan, 'sesudah' = sesudah; crop di sekitar sentuhan (+-margin) dibuang.
    Track yang bolak-balik (beberapa perjalanan tergabung) ditolak supaya label tidak tercampur.
    """
    if len(crops) < 2 * min_side + 1:
        return None
    ds = [d for _, d, _ in crops]
    dmin, dmax = min(ds), max(ds)
    if dmax - dmin < 0.15:
        return None
    thr = dmin + 0.35 * (dmax - dmin)
    near = [d <= thr for d in ds]
    segs = sum(1 for i, v in enumerate(near) if v and (i == 0 or not near[i - 1]))
    if segs != 1:
        return None
    idx = [i for i, v in enumerate(near) if v]
    t0, t1 = crops[idx[0]][0], crops[idx[-1]][0]
    before = [x for x in crops if x[0] < t0 - margin]
    after = [x for x in crops if x[0] > t1 + margin]
    if len(before) < min_side or len(after) < min_side:
        return None
    return before, after


def detect_sequence(vals, cfg):
    """
    Deteksi DUA perubahan berurutan pada deret skor bak P(bermuatan), mis. penuh -> kosong -> penuh
    (= Unloading lalu Loading). Cocokkan deret dengan 3 segmen konstan (SSE minimum, tiap segmen >= min_phase
    sampel) lalu terima bila: dua selisih berlawanan tanda, keduanya >= CAM1_SEQ_DELTA, dan 3 segmen menurunkan
    galat >= CAM1_SEQ_GAIN dibanding 1 perubahan terbaik. Return dict(phases, idx, review, confidence, txt) | None.
    """
    n, m = len(vals), max(2, int(getattr(cfg, "seq_min_phase", cfg.min_phase)))
    if n < 3 * m or not getattr(cfg, "seq_enable", 1):
        return None
    v = np.asarray(vals, dtype=float)
    cs, cs2 = np.concatenate([[0.0], np.cumsum(v)]), np.concatenate([[0.0], np.cumsum(v * v)])

    def sse(a, b):
        k = b - a
        return (cs2[b] - cs2[a]) - (cs[b] - cs[a]) ** 2 / k

    best3, best2 = None, None
    for i in range(m, n - 2 * m + 1):
        for j in range(i + m, n - m + 1):
            e = sse(0, i) + sse(i, j) + sse(j, n)
            if best3 is None or e < best3[0]:
                best3 = (e, i, j)
    for i in range(m, n - m + 1):
        e = sse(0, i) + sse(i, n)
        if best2 is None or e < best2[0]:
            best2 = (e, i)
    if best3 is None or best2 is None:
        return None
    _, i, j = best3
    m1, m2, m3 = (float(np.mean(v[:i])), float(np.mean(v[i:j])), float(np.mean(v[j:])))
    d1, d2 = m2 - m1, m3 - m2
    if d1 * d2 >= 0 or min(abs(d1), abs(d2)) < cfg.seq_delta:
        return None
    gain = (best2[0] - best3[0]) / max(best2[0], 1e-9)
    if gain < cfg.seq_gain:
        return None
    phases = [LOADING if d1 > 0 else UNLOADING, LOADING if d2 > 0 else UNLOADING]
    strength = min(abs(d1), abs(d2))
    return dict(phases=phases, idx=(i, j), review=bool(strength < cfg.delta_strong or n < 4 * m),
                confidence=round(min(0.95, math.tanh(cfg.llr_bed * strength / 2.0)), 2),
                txt=f"{m1:.2f} -> {m2:.2f} -> {m3:.2f} (fase {i}/{j - i}/{n - j} sampel, penurunan galat {gain:.0%})")


def detect_cargo_change(samples, cfg):
    """Cari satu perubahan keadaan bak yang stabil, tahan pencilan, dan tetap urutkan waktu."""
    indexed = list(enumerate(samples))
    if any("parked" in sample for sample in samples):
        indexed = [(i, sample) for i, sample in indexed if sample.get("parked")]
    min_phase = max(2, cfg.min_phase)
    if len(indexed) < 2 * min_phase:
        return None

    vals = [float(sample["p"]) for _, sample in indexed]
    n = len(vals)
    baseline = float(median(vals))
    baseline_error = sum(abs(v - baseline) for v in vals)
    if baseline_error <= 1e-9:
        return None

    best = None
    for split in range(min_phase, n - min_phase + 1):
        before, after = vals[:split], vals[split:]
        before_med, after_med = float(median(before)), float(median(after))
        delta = after_med - before_med
        if abs(delta) < cfg.seq_delta:
            continue
        before_error = sum(abs(v - before_med) for v in before)
        after_error = sum(abs(v - after_med) for v in after)
        gain = (baseline_error - before_error - after_error) / baseline_error
        # [FIX-GAIN-LOKAL] Gain global dibandingkan dengan galat SELURUH sesi: fase pendek (mis. 13 sampel bak masih tertutup terpal
        # sebelum dibongkar) kalah oleh ratusan sampel bak kosong/terisi lain, sehingga Unloading di awal ditolak (log 4 Nov:
        # gain terbaik 0.28 < 0.35 padahal skor 0.35 -> 0.05). Dinilai juga secara LOKAL: hanya jendela selebar fase pendek
        # di tiap sisi titik perubahan. Diterima bila salah satu (global / lokal) lolos ambang.
        w = max(2 * min_phase, min(len(before), len(after)))
        wb, wa = before[-w:], after[:w]
        wm = float(median(wb + wa))
        w_base = sum(abs(v - wm) for v in wb + wa)
        if w_base > 1e-9:
            wb_med, wa_med = float(median(wb)), float(median(wa))
            local_gain = (w_base - sum(abs(v - wb_med) for v in wb) - sum(abs(v - wa_med) for v in wa)) / w_base
            gain = max(gain, local_gain)
        if gain < cfg.seq_gain:
            continue

        supported = (
            (delta < 0 and max(before) >= cfg.loaded_min and after_med < cfg.loaded_min)
            or (delta > 0 and min(before) < cfg.loaded_min and after_med >= cfg.loaded_min)
        )
        if not supported:
            continue

        before_mad = float(median([abs(v - before_med) for v in before]))
        after_mad = float(median([abs(v - after_med) for v in after]))
        noise = 1.4826 * max(before_mad, after_mad)
        q_noise = 1.0 / (1.0 + (noise / 0.15) ** 2)
        q_n = min(1.0, n / float(2 * cfg.phase_k))
        llr = cfg.llr_bed * q_n * q_noise * delta
        score = abs(llr) * gain
        if best is None or score > best["score"]:
            original_split = indexed[split][0]
            left_t = indexed[split - 1][1].get("t")
            right_t = indexed[split][1].get("t")
            change_t = (left_t + right_t) / 2.0 if left_t is not None and right_t is not None else None
            best = dict(
                score=score,
                split=original_split,
                t=change_t,
                early=before_med,
                late=after_med,
                delta=delta,
                confidence=math.tanh(abs(llr) / 2.0),
                llr=llr,
                gain=gain,
                noise=noise,
                n_before=len(before),
                n_after=len(after),
            )
    return best


def fk_sequence(events, cfg):
    """
    [PATCH-3] Dua fase berurutan dari gerakan forklift. events = [(waktu, 'ambil'|'taruh', di_tumpukan, jarak, track)].
    Tiap transisi memberi satu suara: ambil DI tumpukan / taruh DI LUAR tumpukan = Loading; kebalikannya = Unloading.
    Diterima bila SEMUA transisi punya lokasi tumpukan (poligon 'stage' ada), suara terurut waktu membentuk TEPAT dua blok murni
    (mis. U U L L), dan tiap blok >= cfg.fk_seq_min suara. Return dict(phases, counts, t_switch) | None.
    """
    if not events or not getattr(cfg, "fk_seq_enable", 1):
        return None
    events = [e for e in events if e[2] is not None]          # transisi yang lokasinya ambigu tidak bersuara
    if len(events) < 2 * cfg.fk_seq_min:
        return None
    ev = sorted(events, key=lambda e: e[0])
    votes = [(e[0], LOADING if (e[1] == "ambil") == bool(e[2]) else UNLOADING) for e in ev]
    blocks = []                                   # [jenis, jumlah, waktu_awal, waktu_akhir]
    for t, v in votes:
        if blocks and blocks[-1][0] == v:
            blocks[-1][1] += 1
            blocks[-1][3] = t
        else:
            blocks.append([v, 1, t, t])
    if len(blocks) != 2 or min(b[1] for b in blocks) < cfg.fk_seq_min:
        return None
    return dict(phases=[blocks[0][0], blocks[1][0]], counts=[blocks[0][1], blocks[1][1]],
                t_switch=0.5 * (blocks[0][3] + blocks[1][2]))


def detect_refill_after_unload(samples, cargo_change, fk_tracks, cfg):
    """
    [PATCH-7] Loading yang menyusul Unloading ketika skor bak nyaris tidak naik.
    Kasus log 4 Nov (2-act_potong.mp4, pickup): bak penuh 0.44 -> kosong 0.05 (Unloading, jelas), lalu ~220 dtk bak kosong,
    forklift kembali dan memuat. Skor bak saat memuat hanya 0.05 -> ~0.16 (bak tertutup orang/forklift, sampel 'x'/'O' dibuang,
    isi pickup tidak dikenali penuh oleh model), jadi selisih median < CAM1_SEQ_DELTA dan fase Loading tidak pernah terbaca.
    Di sini fase kedua dinilai dari tiga bukti sekaligus:
      (a) ada Unloading yang jelas (bak turun dan akhirnya < loaded_min),
      (b) forklift KEMBALI sesudah jeda >= refill_gap tanpa forklift dan bertahan >= refill_fk_min dtk,
      (c) skor bak sesudah forklift kembali bergeser naik dari bak kosong: >= refill_frac sampel melewati p90 bak kosong
          (tanpa perubahan nyata hanya ~10%) DAN p90 skor naik >= refill_rise.
    Hasil selalu perlu_review + bukti tipis. Return dict(phases, t_start, frac, ...) | None.
    """
    if not getattr(cfg, "refill_enable", 1) or not cargo_change or cargo_change.get("t") is None:
        return None
    if cargo_change["llr"] >= 0 or cargo_change["late"] >= cfg.loaded_min:
        return None
    t_ch = float(cargo_change["t"])
    fk_t = sorted(float(t) for obs in (fk_tracks or []) for t, _ in obs if t > t_ch)
    if not fk_t:
        return None
    start, prev = None, t_ch
    for t in fk_t:
        if t - prev >= cfg.refill_gap:
            start = t
            break
        prev = t
    if start is None:
        return None
    back = [t for t in fk_t if t >= start]
    near = sum(min(b - a, 6.0) for a, b in zip(back, back[1:]))
    if near < cfg.refill_fk_min:
        return None
    pool = [x for x in samples if x.get("parked", True)]
    base = [float(x["p"]) for x in pool if t_ch < x["t"] < start]
    late = [float(x["p"]) for x in pool if x["t"] >= start]
    if len(base) < cfg.refill_min_base or len(late) < cfg.refill_min_late:
        return None
    base_p90 = float(np.percentile(base, 90))
    late_p90 = float(np.percentile(late, 90))
    frac = float(np.mean(np.asarray(late) > base_p90))
    rise = late_p90 - base_p90
    if frac < cfg.refill_frac or rise < cfg.refill_rise:
        return None
    return dict(phases=[UNLOADING, LOADING], t_start=start, frac=frac, rise=rise, fk_near=near,
                n_base=len(base), n_late=len(late),
                txt=(f"bak kosong {float(np.median(base)):.2f} (p90 {base_p90:.2f}, {len(base)} sampel); forklift kembali "
                     f"{start - t_ch:.0f} dtk sesudah bak kosong, selama ~{near:.0f} dtk; sesudahnya {frac:.0%} dari {len(late)} sampel bak di atas p90 bak kosong "
                     f"(acak ~10%), p90 naik {rise:+.2f}"))


def decide(samples, cfg, fk_seconds=0.0, fk_tracks=None, fk_dists=None, fk_stages=None, t_end=None, docked=None, manual=0, final=False):
    """[PATCH-2] Pembungkus: hasil _decide_core + pemeriksaan kelengkapan bukti (hanya menambah perlu_review, tidak mengubah status).
    t_end = waktu sesi ditutup; docked = True/False bila poligon dock ada (None = tidak diketahui).
    final=True menjalankan pencarian perubahan bak yang tahan pencilan hanya saat sesi ditutup."""
    out = _decide_core(samples, cfg, fk_seconds, fk_tracks, fk_dists, fk_stages, manual, final)
    st = out.get("status")
    if st == UNSURE or not (st in (LOADING, UNLOADING) or (st and "+" in st)):
        return out
    why = []
    if samples and t_end is not None and cfg.end_gap > 0 and t_end - samples[-1]["t"] > cfg.end_gap:
        why.append(f"keadaan AKHIR bak tidak teramati (sampel terakhir {t_end - samples[-1]['t']:.0f} dtk sebelum sesi ditutup)")
    if len(samples) >= 2 and cfg.hole_gap > 0:
        hole = max(b["t"] - a["t"] for a, b in zip(samples, samples[1:]))
        if hole > cfg.hole_gap:
            why.append(f"ada lubang sampel bak {hole:.0f} dtk (kemungkinan forklift/orang menutupi bak saat aktivitas)")
    if docked is False:
        why.append("kendaraan tidak pernah parkir di zona dock (mungkin forklift/objek lewat)")
    if why:
        out["review"] = True
        out["reason"] = (out.get("reason", "") + "; " + "; ".join(why) + " -> perlu review").lstrip("; ")
    return out


def _decide_core(samples, cfg, fk_seconds=0.0, fk_tracks=None, fk_dists=None, fk_stages=None, manual=0, final=False):
    """
    Penentu Loading/Unloading berbasis FUSI BUKTI. Tiap bukti = log-odds (positif = Loading, negatif = Unloading):
      skor        selisih P(bermuatan) awal -> akhir; dilemahkan bila sampel sedikit / berisik / keadaan akhir tak mendukung
      forklift_arah   (taruh - ambil)/sqrt(total) dari model forklift bawa/kosong (hanya bila modelnya ada)
      logika_forklift forklift aktif lama di dekat kendaraan yang TIBA kosong -> tak mungkin Unloading (condong Loading);
                      tiba penuh -> condong Unloading (lemah). Hanya dipakai bila model forklift diam (tidak ada arah).
    confidence = tanh(|jumlah|/2). < cfg.review_min -> "Tidak Dapat Dipastikan" + review (atau asumsi konfigurasi lama).
    Return dict: status, source, confidence, review, early, late, delta, n, n_phase, change_idx, reason,
                 bed_conf/bed_dir (keyakinan dari skor bak SAJA; dipakai pelabelan forklift), terms.
    """
    sample_indexes = list(range(len(samples)))
    if any("parked" in sample for sample in samples):
        sample_indexes = [i for i, sample in enumerate(samples) if sample.get("parked")]
    score_samples = [samples[i] for i in sample_indexes]
    vals = [s["p"] for s in score_samples]
    n = len(vals)
    out = dict(status=UNSURE, source="tidak_pasti", confidence=0.0, review=True,
               early=None, late=None, delta=None, n=n, n_phase=0, change_idx=None, reason="",
               bed_conf=0.0, bed_dir=None, terms={})
    terms, notes = {}, []

    # (1) skor bak ----------------------------------------------------------------------------------
    if n >= 2:   # perkiraan skor awal walau sampel sedikit (dipakai logika tiba-kosong & payload status_muatan_awal)
        out["early"] = round(float(median(vals[:min(cfg.phase_k, n)])), 3)
    if n < 2 * cfg.min_phase:
        notes.append(f"sampel muatan kurang ({n} < {2 * cfg.min_phase}); bak tidak terbaca / kendaraan jarang parkir")
    else:
        # jendela fase ikut membesar bersama jumlah sampel (n/5): median atas banyak sampel lebih kebal terhadap pencilan dan
        # menangkap tren pelan (mis. bak terisi bertahap) yang tak terlihat bila hanya 5 sampel pertama vs 5 terakhir dibandingkan
        k = min(max(cfg.phase_k, int(n * cfg.phase_frac)), n // 2)
        early, late = float(median(vals[:k])), float(median(vals[-k:]))
        delta = late - early
        out.update(early=round(early, 3), late=round(late, 3), delta=round(delta, 3), n_phase=k)
        best, best_i = 0.0, None
        for i in range(cfg.min_phase, n - cfg.min_phase + 1):
            d = abs(float(np.mean(vals[i:])) - float(np.mean(vals[:i])))
            if d > best:
                best, best_i = d, i
        out["change_idx"] = sample_indexes[best_i] if best_i is not None else None
        noise = max(float(np.std(vals[:k])), float(np.std(vals[-k:])))
        q_noise = 1.0 / (1.0 + (noise / 0.15) ** 2)               # skor berisik -> bukti melemah
        q_n = min(1.0, n / float(2 * cfg.phase_k))                # fase belum penuh -> bukti melemah
        support = (late >= cfg.loaded_min) if delta > 0 else (early >= cfg.loaded_min)
        l_bed = cfg.llr_bed * q_n * q_noise * (1.0 if support else 0.5) * delta
        terms["skor"] = l_bed
        out["bed_conf"] = round(math.tanh(abs(l_bed) / 2.0), 3)
        out["bed_dir"] = None if l_bed == 0 else (LOADING if l_bed > 0 else UNLOADING)
        notes.append(f"skor muatan {early:.2f} -> {late:.2f} (selisih {delta:+.2f}, {k} sampel/fase dari {n}"
                     + ("" if q_n >= 1.0 else ", sampel sedikit") + ("" if q_noise >= 0.8 else ", berisik")
                     + ("" if support else ", keadaan akhir tidak mendukung") + ")")

    cargo_change = detect_cargo_change(samples, cfg) if final else None
    if cargo_change and abs(cargo_change["llr"]) > abs(terms.get("skor", 0.0)):
        terms["skor"] = cargo_change["llr"]
        out.update(
            early=round(cargo_change["early"], 3),
            late=round(cargo_change["late"], 3),
            delta=round(cargo_change["delta"], 3),
            n_phase=min(cargo_change["n_before"], cargo_change["n_after"]),
            change_idx=cargo_change["split"],
            bed_conf=round(cargo_change["confidence"], 3),
            bed_dir=LOADING if cargo_change["llr"] > 0 else UNLOADING,
        )
        notes.append(
            f"perubahan bak stabil {cargo_change['early']:.2f} -> {cargo_change['late']:.2f} "
            f"(gain {cargo_change['gain']:.0%}, MAD {cargo_change['noise']:.2f})"
        )

    # (2) arah dari gerakan forklift (taruh = Loading, ambil = Unloading) ---------------------------
    fkd = fk_direction(fk_tracks, cfg, fk_dists, fk_stages) if fk_tracks else None
    if fkd and fkd["status"]:
        e_d, e_p = fkd["eff_drops"], fkd["eff_picks"]
        total = max(1, e_d + e_p)
        # satu kejadian bersih saja (selisih taruh-ambil = 1) mudah berasal dari skor forklift yang berisik -> bobotnya dikecilkan
        w_single = cfg.fk_single_w if abs(e_d - e_p) <= 1 else 1.0
        if fkd.get("relaxed") and abs(e_d - e_p) <= 1 and w_single == 0:
            w_single = 0.5
        terms["forklift_arah"] = max(-3.5, min(3.5, w_single * fkd["geo_w"] * cfg.fk_llr * (e_d - e_p) / math.sqrt(total)))
        notes.append(fkd["reason"])
        if fkd.get("relaxed"):
            notes.append("arah forklift memakai ambang adaptif; perlu review")
    elif fkd and fkd["reason"]:
        notes.append(fkd["reason"])

    # (3) logika keadaan tiba + aktivitas forklift (hanya bila model forklift tidak bicara) ----------
    active = fk_seconds >= cfg.fk_near_min
    if active:
        notes.append(f"forklift aktif di dekat kendaraan ~{fk_seconds:.0f} dtk")
    if active and out["early"] is not None:
        g = min(1.0, fk_seconds / (3.0 * cfg.fk_near_min))
        if out["early"] < cfg.loaded_min:
            v, why = cfg.arrival_prior * g, f"tiba KOSONG (skor awal {out['early']:.2f}) -> tidak mungkin Unloading, condong Loading"
        elif out["early"] >= 0.65:
            v, why = -0.4 * cfg.arrival_prior * g, f"tiba BERMUATAN (skor awal {out['early']:.2f}) -> condong Unloading (lemah)"
        else:
            v, why = 0.0, ""
        # bila model forklift sudah bicara dan BERLAWANAN, forklift didahulukan (skor tiba bisa keliru, mis. boks tertutup
        # dibaca 'Kosong'); bila searah, logika ini ikut menguatkan.
        # asumsi lemah ini juga kalah dari pengukuran langsung skor bak yang berlawanan (mis. tiba penuh tapi isi bak naik = Loading tambahan)
        bed_opposes = "skor" in terms and abs(terms["skor"]) >= 0.3 and v * terms["skor"] < 0
        if v != 0.0 and not ("forklift_arah" in terms and v * terms["forklift_arah"] < 0) and not bed_opposes:
            terms["logika_forklift"] = v
            notes.append(why)

    # (3b) dua fase berurutan dalam satu sesi (mis. Unloading lalu Loading) -------------------------
    seq = detect_sequence(vals, cfg)
    if seq:
        out.update(status="+".join(seq["phases"]), source="skor_dua_fase", review=seq["review"],
                   confidence=seq["confidence"], sequence=seq["phases"],
                   change_idx=sample_indexes[seq["idx"][0]],
                   reason=f"{'; '.join(notes)}; DUA FASE: {seq['txt']}")
        return out

    if cargo_change and cargo_change["t"] is not None and fkd:
        later = [
            (event[0], LOADING if (event[1] == "ambil") == bool(event[2]) else UNLOADING)
            for event in fkd.get("events", [])
            if event[2] is not None and event[0] > cargo_change["t"]
        ]
        if later:
            counts = {LOADING: 0, UNLOADING: 0}
            for _, direction in later:
                counts[direction] += 1
            later_direction = max(counts, key=counts.get)
            n_later = counts[later_direction]
            agreement = n_later / len(later)
            first_direction = LOADING if cargo_change["llr"] > 0 else UNLOADING
            if (later_direction != first_direction and n_later >= max(2, cfg.fk_seq_min)
                    and agreement >= 0.75):
                phases = [first_direction, later_direction]
                out.update(
                    status="+".join(phases),
                    source="skor_bak+forklift_arah",
                    confidence=round(min(cargo_change["confidence"], agreement), 2),
                    review=True,
                    sequence=phases,
                    change_idx=cargo_change["split"],
                    terms={k_: round(v, 2) for k_, v in terms.items()},
                    reason=(f"{'; '.join(notes)}; urutan terkonfirmasi: fase bak {first_direction}, "
                            f"lalu {n_later}/{len(later)} transisi forklift mendukung {later_direction}"),
                )
                return out

    # (3b2) [PATCH-7] Unloading jelas, lalu forklift kembali dan skor bak ikut naik (walau di bawah CAM1_SEQ_DELTA) -> Unloading+Loading
    refill = detect_refill_after_unload(samples, cargo_change, fk_tracks, cfg) if final else None
    if refill:
        notes.append("Loading menyusul Unloading: " + refill["txt"])
        out.update(status="+".join(refill["phases"]), source="bak_kosong+forklift_kembali", review=True, thin=True,
                   confidence=round(min(0.6, 0.3 + 0.5 * (refill["frac"] - cfg.refill_frac) + 2.0 * (refill["rise"] - cfg.refill_rise)), 2),
                   sequence=list(refill["phases"]), terms={k_: round(v, 2) for k_, v in terms.items()},
                   reason="; ".join(notes) + "; Loading hanya dari kembalinya forklift + kenaikan kecil skor bak -> PERLU REVIEW")
        return out

    # (3c) [PATCH-3] dua fase dari suara lokasi tumpukan forklift (butuh poligon 'stage') --------------
    fseq = fk_sequence(fkd["events"], cfg) if fkd else None
    if fseq:
        n1, n2 = fseq["counts"]
        out.update(status="+".join(fseq["phases"]), source="forklift_dua_fase", review=bool(min(n1, n2) < 3),
                   confidence=round(min(0.9, 0.4 + 0.15 * min(n1, n2)), 2), sequence=fseq["phases"],
                   reason=f"{'; '.join(notes)}; DUA FASE dari forklift+tumpukan: {fseq['phases'][0]} x{n1} lalu {fseq['phases'][1]} x{n2}")
        return out

    # (3d) [PATCH-5] penurunan skor bak DI AWAL sesi = kendaraan tiba bermuatan lalu dibongkar (mis. orang membongkar manual di bak,
    # sehingga forklift tak terlihat dan sampel awal banyak terbuang karena orang di kotak). Satu sampel tinggi di awal tidak
    # menggeser median fase awal, jadi tidak terbaca oleh skor biasa. Hanya dipakai bila keputusan akhir tetap Loading, dan
    # SELALU perlu_review (bukti berasal dari sedikit sampel).
    total_pre = sum(terms.values())
    if getattr(cfg, "early_drop", 1) and total_pre > 0 and math.tanh(abs(total_pre) / 2.0) >= cfg.review_min and n >= 6:
        moving_prefix = []
        if any("parked" in sample for sample in samples):
            for sample in samples:
                if sample.get("parked"):
                    break
                moving_prefix.append(sample)
        use_moving_arrival = (
            len(moving_prefix) >= 3
            and sum(float(sample["p"]) >= cfg.loaded_min for sample in moving_prefix[:3]) >= 2
        )
        head_samples = moving_prefix[:3] if use_moving_arrival else score_samples[:3]
        rest_samples = score_samples[:8] if use_moving_arrival else score_samples[3:11]
        peak = max(float(sample["p"]) for sample in head_samples)
        # [FIX-SKALA] Penurunan skor hanya dipercaya bila dibandingkan pada UKURAN KOTAK yang sebanding. Kendaraan yang berpindah
        # posisi (mis. dari dekat kamera ke dermaga jauh) mengubah skala crop bak sehingga skor bisa turun tanpa ada barang keluar.
        # Sampel lama tanpa field "sz" tetap diperlakukan seperti sebelumnya.
        scale_note = ""
        head_szs = [sample.get("sz") for sample in head_samples if sample.get("sz")]
        if len(head_szs) == len(head_samples):
            head_sz = float(median(head_szs))
            lim = max(1.05, float(getattr(cfg, "early_scale", 1.5)))
            kept = [
                sample for sample in rest_samples
                if sample.get("sz") and 1.0 / lim <= sample["sz"] / head_sz <= lim
            ]
            if len(kept) < len(rest_samples):
                scale_note = f" [{len(rest_samples) - len(kept)} sampel pembanding dibuang: ukuran kotak beda > {lim:.1f}x dari sampel awal]"
            rest_samples = kept
        rest = [float(sample["p"]) for sample in rest_samples]
        later_loading = []
        if use_moving_arrival and fkd:
            later_loading = [
                event for event in fkd["events"]
                if event[2] is not None and event[0] > head_samples[-1]["t"]
                and (LOADING if (event[1] == "ambil") == bool(event[2]) else UNLOADING) == LOADING
            ]
        has_later_forklift_loading = (
            not use_moving_arrival
            or len(later_loading) >= max(2, cfg.fk_seq_min)
        )
        if peak >= 0.5 and len(rest) >= 3 and has_later_forklift_loading:
            rest_med = float(median(rest))
            tail_med = float(median(vals[-min(5, n):]))
            # [FIX-DUA-FASE] dulu WAJIB tail <= rest + 0.15 ("bertahan rendah sampai akhir"), padahal pada Unloading lalu Loading bak
            # terisi lagi di akhir -> pola yang justru dicari malah ditolak. Bila bak terisi lagi, bukti awal harus lebih kuat:
            # minimal 2 dari 3 sampel awal >= 0.5 (bukan satu sampel pencilan).
            refilled = tail_med > rest_med + 0.15
            head_high = sum(float(sample["p"]) >= 0.5 for sample in head_samples)
            if peak - rest_med >= cfg.seq_delta and (not refilled or head_high >= 2):
                origin = "sebelum kendaraan parkir" if use_moving_arrival else "di awal sesi"
                notes.append(f"skor bak turun {origin} ({peak:.2f} -> {rest_med:.2f}{', lalu terisi lagi' if refilled else ', bertahan sampai akhir'}): tiba bermuatan lalu dibongkar{scale_note}")
                out.update(status=f"{UNLOADING}+{LOADING}", source="skor_awal_turun+forklift", review=True, thin=True,
                           confidence=round(min(0.7, 0.2 + (peak - rest_med)), 2), sequence=[UNLOADING, LOADING],
                           terms={k_: round(v, 2) for k_, v in terms.items()},
                           reason="; ".join(notes) + "; Unloading di awal hanya dari penurunan skor (sedikit sampel) -> PERLU REVIEW")
                return out

    # (4) gabungkan ----------------------------------------------------------------------------------
    # Median fase yang berpindah status muatan adalah bukti arah meski delta kecil;
    # status itu baru boleh dipakai jika jumlah sampel memenuhi kedua fase.
    state_transition = None
    if n >= 2 * cfg.min_phase and out.get("early") is not None and out.get("late") is not None:
        initial_state = p_to_status(out["early"])
        final_state = p_to_status(out["late"])
        if initial_state != final_state:
            state_transition = LOADING if final_state == "Bermuatan" else UNLOADING

    # [FIX-BAK-TAK-BERUBAH] perubahan kecil di dalam status yang sama tidak boleh ikut "menyetir" hasil lemah.
    if "skor" in terms and abs(out.get("delta") or 0.0) < cfg.delta_weak:
        if state_transition:
            notes.append(f"median muatan berubah {initial_state} -> {final_state}; arah {state_transition} dipertahankan meski selisih skor di bawah {cfg.delta_weak:.2f}")
        else:
            terms["skor"] = 0.0
            notes.append(f"selisih skor bak {out.get('delta')} < {cfg.delta_weak:.2f}: dianggap TIDAK berubah")
    total_l = sum(terms.values())
    conf = math.tanh(abs(total_l) / 2.0)
    out["terms"] = {k_: round(v, 2) for k_, v in terms.items()}
    out["confidence"] = round(conf, 2)
    note_txt = "; ".join(notes)
    if conf >= cfg.review_min:
        direction = LOADING if total_l > 0 else UNLOADING
        sign = 1.0 if total_l > 0 else -1.0
        agree = [k_ for k_, v in sorted(terms.items(), key=lambda kv: -abs(kv[1])) if v * sign >= 0.25]
        oppose = [k_ for k_, v in terms.items() if v * sign <= -0.8]
        prior_only = bool(agree) and all(k_ == "logika_forklift" for k_ in agree)
        if prior_only:
            # [FIX-ASUMSI] "tiba kosong + forklift aktif" BUKAN bukti. Dulu ini menghasilkan Loading (perlu_review) yang
            # tampil di dashboard seperti hasil pasti; kasus CAM1_20261023_110409_0001 (pickup berpalet, Unloading) salah di sini.
            out.update(status=UNSURE, source="tidak_pasti", review=True, confidence=0.0,
                       reason=f"{note_txt}; hanya asumsi keadaan tiba + aktivitas forklift, tanpa bukti langsung -> TIDAK diputuskan")
            return out
        out.update(status=direction, source="+".join(agree) or "fusi",
                   review=bool(state_transition or oppose or (fkd and fkd.get("relaxed"))),
                   reason=f"{note_txt}; keyakinan {conf:.2f}"
                          + (f"; BERTENTANGAN dengan {', '.join(oppose)}" if oppose else ""))
        return out

    if state_transition:
        out.update(status=state_transition, source="perubahan_status_muatan", review=True,
                   reason=f"{note_txt}; perubahan median status muatan -> {state_transition} (arah lemah, PERLU REVIEW)")
        return out

    # bukti lemah tapi searah dan berasal dari pengukuran langsung (skor bak / arah forklift, BUKAN sekadar asumsi tiba):
    # laporkan arah terbaik dengan perlu_review=true, bukan 'Tidak Dapat Dipastikan'. Batas bawah: CAM1_GUESS_MIN (0 = matikan).
    direct = {k_: v for k_, v in terms.items() if k_ != "logika_forklift"}
    if cfg.guess_min > 0 and conf >= cfg.guess_min and direct and total_l * sum(direct.values()) > 0:
        direction = LOADING if total_l > 0 else UNLOADING
        out.update(status=direction, source="+".join(sorted(direct, key=lambda k_: -abs(direct[k_]))) + "_lemah", review=True,
                   reason=f"{note_txt}; keyakinan {conf:.2f} < {cfg.review_min:.2f} -> arah terbaik dilaporkan, PERLU REVIEW")
        return out

    # keyakinan terlalu rendah: asumsi konfigurasi lama (bila diisi) atau Tidak Dapat Dipastikan
    out["reason"] = f"{note_txt}; keyakinan {conf:.2f} < {cfg.review_min:.2f}"
    # [PATCH-6] KEPUTUSAN PAKSA: tebakan terbaik dari bukti yang ada, urut dari yang paling langsung. Selalu perlu_review.
    if getattr(cfg, "force_decision", 1):
        pick = None
        if abs(total_l) >= 0.10:                      # jumlah bukti yang nyaris nol hanya derau, bukan arah
            pick = (LOADING if total_l > 0 else UNLOADING, "jumlah_bukti")
        elif fkd and fkd.get("drops", 0) != fkd.get("picks", 0):
            pick = (LOADING if fkd["drops"] > fkd["picks"] else UNLOADING, "forklift_tunggal")
        elif (manual >= cfg.manual_min and fk_seconds < cfg.fk_near_min and cfg.manual_default in (LOADING, UNLOADING)
            and not (fkd and fkd.get("drops", 0) + fkd.get("picks", 0) > 0)):
            pick = (cfg.manual_default, "prior_manual")
        elif out.get("delta") is not None and abs(out["delta"]) >= 0.03:
            pick = (LOADING if out["delta"] > 0 else UNLOADING, "skor_bak_lemah")
        elif out.get("early") is not None and abs(out["early"] - 0.5) >= 0.1:
            pick = (UNLOADING if out["early"] >= 0.5 else LOADING, "keadaan_tiba")      # tiba penuh -> condong Unloading; tiba kosong -> Loading
        if pick:
            out.update(status=pick[0], source="paksa_" + pick[1], confidence=0.15, review=True,
                       reason=f"{out['reason']}; bukti lemah -> tebakan terbaik {pick[0]} dari {pick[1]} (PAKSA, PERLU REVIEW)")
            return out
    d = cfg.default_direction
    if d in (LOADING, UNLOADING):
        out.update(status=d, source="asumsi", confidence=0.2, review=True,
                   reason=f"{out['reason']}; memakai asumsi konfigurasi CAM1_DEFAULT_DIRECTION={d}")
    elif fk_seconds >= cfg.fk_near_min and cfg.fk_assume in (LOADING, UNLOADING):
        e0 = out["early"]
        contradicts = e0 is not None and ((cfg.fk_assume == LOADING and e0 >= 0.60) or
                                          (cfg.fk_assume == UNLOADING and e0 < cfg.loaded_min))
        if not contradicts:
            out.update(status=cfg.fk_assume, source="asumsi_forklift", confidence=0.3, review=True,
                       reason=f"{out['reason']}; forklift aktif di dekat kendaraan {fk_seconds:.0f} dtk -> "
                              f"diasumsikan {cfg.fk_assume} (CAM1_FORKLIFT_ASSUME)")
    if out["status"] == UNSURE and getattr(cfg, "force_decision", 1):
        fd = cfg.force_default if cfg.force_default in (LOADING, UNLOADING) else LOADING
        out.update(status=fd, source="paksa_prior_lokasi", confidence=0.1, review=True,
                   reason=f"{out['reason']}; TANPA bukti yang bisa dipakai -> prior lokasi {fd} (CAM1_FORCE_DEFAULT, PERLU REVIEW)")
    return out


# ----------------------------------------------------------------------------------------------
# Pengirim n8n (antrian + retry + outbox disk)
# ----------------------------------------------------------------------------------------------
class N8NSender:
    def __init__(self, cfg, enabled=True):
        self.cfg, self.enabled = cfg, enabled and bool(cfg.n8n_url)
        self.q = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        if self.enabled:
            self._t.start()

    def send(self, payload):
        # catatan lokal selalu ditulis (tanpa gambar) sebagai cadangan/audit
        try:
            slim = {k: v for k, v in payload.items() if not k.endswith("base64")}
            with open(self.cfg.events_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(slim, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            log.warning("gagal tulis events log: %s", e)
        if self.enabled:
            self.q.put(payload)

    def _post(self, payload):
        req = urllib.request.Request(self.cfg.n8n_url, data=json.dumps(payload, default=str).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.n8n_timeout) as r:
            if getattr(r, "status", 200) >= 300:
                raise RuntimeError(f"HTTP {r.status}")
            r.read()

    def _outbox_add(self, payload):
        with self._lock:
            p = self.cfg.outbox_path
            if os.path.isfile(p) and os.path.getsize(p) > self.cfg.outbox_max_mb * 1024 * 1024:
                log.error("[N8N] outbox penuh, event dibuang: %s", payload.get("event_id"))
                return
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        log.warning("[N8N] disimpan ke outbox: %s", payload.get("event_id"))

    def _flush_outbox(self):
        with self._lock:
            p = self.cfg.outbox_path
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                return
            with open(p, "r", encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            done = 0
            for line in lines:
                try:
                    self._post(json.loads(line))
                except json.JSONDecodeError:
                    pass  # baris rusak dibuang
                except Exception as e:
                    code = getattr(e, "code", None)
                    if isinstance(code, int) and 400 <= code < 500:
                        pass  # ditolak permanen oleh server -> buang agar tidak macet
                    else:
                        break
                done += 1
            rest = lines[done:]
            if rest:
                with open(p + ".tmp", "w", encoding="utf-8") as f:
                    f.write("\n".join(rest) + "\n")
                os.replace(p + ".tmp", p)
            else:
                os.remove(p)
            if done:
                log.info("[N8N] outbox: %d event terkirim ulang", done)

    def _run(self):
        last_flush = 0.0
        while not self._stop.is_set() or not self.q.empty():
            try:
                payload = self.q.get(timeout=1.0)
            except queue.Empty:
                payload = None
            if payload is not None:
                for attempt in range(1, self.cfg.n8n_retries + 1):
                    try:
                        self._post(payload)
                        log.info("[N8N] terkirim: %s %s", payload.get("jenis_kejadian"), payload.get("vehicle_id"))
                        break
                    except Exception as e:
                        if attempt == self.cfg.n8n_retries:
                            log.warning("[N8N] gagal (%s)", e)
                            self._outbox_add(payload)
                        else:
                            time.sleep(self.cfg.n8n_retry_delay * attempt)
            if time.time() - last_flush > self.cfg.outbox_interval:
                last_flush = time.time()
                try:
                    self._flush_outbox()
                except Exception as e:
                    log.warning("[N8N] flush outbox error: %s", e)

    def close(self, timeout=20):
        self._stop.set()
        if self.enabled:
            self._t.join(timeout=timeout)


# ----------------------------------------------------------------------------------------------
# Sesi kendaraan & monitor (inti sistem)
# ----------------------------------------------------------------------------------------------
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def expand_box(b, r):
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - w * r, b[1] - h * r, b[2] + w * r, b[3] + h * r)


def appearance_sig(frame, bbox):
    """Histogram HSV 4x4x8 (L1-normalized) dari kotak kendaraan: dipakai membedakan kendaraan setelah jeda."""
    crop = crop_vehicle(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    small = cv2.resize(crop, (64, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 8], [0, 180, 0, 256, 0, 256]).flatten().astype(np.float64)
    return hist / max(hist.sum(), 1e-9)


def thumb_norm(frame, bbox):
    """Cuplikan abu-abu 32x24 dari kotak, dikurangi rata-rata dan dinormalkan: hasil kali titik dua cuplikan = korelasi (tahan perubahan cahaya)."""
    crop = crop_vehicle(frame, bbox)
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.resize(g, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= float(g.mean())
    n = float(np.linalg.norm(g))
    if n < 1e-3:
        return None                      # frame rata (hitam/beku polos): tidak bisa dipakai sebagai bukti
    return (g / n).flatten()


def appearance_dist(a, b):
    if a is None or b is None:
        return 0.0
    bc = float(np.sum(np.sqrt(a * b)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


class Session:
    def __init__(self, vid, t_first, bbox, photo, sig=None):
        self.vid = vid
        self.t_in = t_first
        self.t_last = t_first
        self.bbox = bbox
        self.conf = 0.0
        self.cls = 7
        self.moving_hist = deque()          # (t, cx, cy, size)
        self.parked = False
        self.parked_since = None
        self.park_best = 0.0                # lama terpanjang parkir diam berturut-turut (dtk)
        self.park_sum = 0.0                 # total akumulasi waktu parkir diam (dtk)
        self.park_tick = None
        self.big_since = None               # [FIX-JARAK-DERMAGA] mulai parkir dengan kotak cukup besar (di dermaga) tanpa putus
        self.far_n = 0                      # jumlah pembaruan parkir tapi kotak terlalu kecil (jauh dari dermaga)
        self.far_logged = False
        self.edge = False                   # [KENDARAAN-TEPI] sesi dibuka dari pinggir area
        self.reloc_hist = []                # [PINDAH-POSISI] sighting lemah beruntun [(waktu, pusat)]
        self.fk_ovl_n = 0                   # pembaruan di mana kotak sesi >= 60% tertutup kotak forklift
        self.n_upd = 0                      # jumlah pembaruan (kotak bersih)
        self.gate_logged = False
        self.ref_thumb, self.ref_t = None, -1e9
        self.pres_prev, self.pres_chain = None, False      # [PRESENSI] acuan cuplikan area kendaraan saat masih terdeteksi
        self.t_present, self.pres_t, self.pres_log_t, self.pres_n, self.pres_ok = None, -1e9, -1e9, 0, None
        self.pres_miss_n = 0               # cek kosong berturut-turut; satu cek gagal bisa disebabkan occlusion
        self.occl_t0 = None                 # [PRESENSI-TUTUP] awal penutupan area parkir oleh forklift
        self.last_parked_bbox = None
        self.departing = False              # [FIX-PERGI] kotak terakhir sudah kecil & menjauh dari titik parkir
        self.frag_n = 0                     # [FIX-FRAGMEN] berapa kali kotak kecil di dalam truk diabaikan
        self.t_dock_in = None
        self.t_dock_out = None
        self.samples = []                   # dict(t, p, jpg)
        self.last_sample_t = 0.0
        self.photo_in = photo               # foto saat pertama terlihat
        self.photo_arrival = None           # foto saat pertama parkir (menimpa photo_in)
        self.photo_last = photo
        self.orange_n = 0                   # jumlah crop yang diperiksa warna oranyenya
        self.orange_hi = 0                  # ...berapa yang oranyenya besar (forklift mendominasi)
        self.last_photo_t = 0.0
        self.announced = False              # MASUK sudah dikirim?
        self.stg_sec = 0.0                  # [TINGKAT-A] detik ada orang di stage dekat truk saat truk parkir
        self.stg_max = 0                    # jumlah orang bersamaan terbanyak di stage
        self.stg_first = None
        self.stg_prev = t_first
        self.photo_stg = None
        self.stg_photo_t = 0.0
        self.max_conf = 0.0
        self.strong_hits = 0                # jumlah deteksi dengan conf >= confirm_conf (syarat MASUK)
        self.sig = sig                      # tampilan (histogram) kendaraan ini
        self.sig_t = t_first
        self.fk_seconds = 0.0               # total detik forklift berada di dekat kendaraan ini
        self.prev_t = t_first
        self.photo_fk = None                # foto saat forklift di dekat kendaraan (bukti aktivitas)
        self.last_fk_photo_t = 0.0
        self.merged_n = 0                   # berapa kali hanya terlihat sebagai kotak gabungan dengan forklift
        self.info = None                    # hasil decide() terakhir (untuk overlay)
        self.fk_tracks = []                 # track forklift di dekat kendaraan: dict(box, t_last, last_obs, obs=[(t, p_bawa)])
        self.fk_dump_n = 0
        self.pc_tracks = []                 # [ARAH-ORANG] track orang di dekat kendaraan: dict(box, t_last, last_obs, obs, dist, stg)
        self.skip = {}                      # diagnostik: berapa frame sampel muatan dilewati, per alasan
        self.tl = []                        # [PATCH-4] garis waktu ringkas untuk log TIMELINE
        self.tl_t = -1e9

    def bump(self, why):
        self.skip[why] = self.skip.get(why, 0) + 1


class Cam1Monitor:
    """Banyak sesi sekaligus. Tidak ada ID track YOLO: sesi dicocokkan lewat posisi (IoU/jarak) + tampilan."""

    def __init__(self, cfg, zones, scorer, sender, fk_scorer=None, pc_scorer=None):
        """fk_scorer(crop_forklift) -> P(forklift membawa barang) atau None. None = arah dari forklift mati."""
        self.cfg, self.zones, self.scorer, self.sender = cfg, zones, scorer, sender
        self.fk_scorer = fk_scorer
        self.pc_scorer = pc_scorer
        log.info("parameter efektif | forklift bawa/kosong: hi=%.2f lo=%.2f min_run=%d min_obs=%d geo=%d | "
                 "fusi: review_min=%.2f arrival_prior=%.2f prior_review=%d | sesi: entry=%.0fs settle=%.1fs | "
                 "bak: BED_BOX=%s person_bed_only=%d fk_crop_pad=%.2f fk_llr=%.2f | tumpukan: fk_stage=%d poligon_stage=%s",
                 cfg.fk_hi, cfg.fk_lo, cfg.fk_min_run, cfg.fk_min_obs, cfg.fk_geo,
                 cfg.review_min, cfg.arrival_prior, cfg.prior_review, cfg.entry_seconds, cfg.settle,
                 ",".join(f"{v:g}" for v in BED_BOX), cfg.person_bed_only, cfg.fk_crop_pad, cfg.fk_llr,
                 cfg.fk_stage, "ADA" if zones.has_stage() else "BELUM digambar")
        log.info("build cam1: FIX-UKURAN-2(ref=kotak terakhir) PRESENSI FIX-FRAGMEN FIX-PERGI | ratio=%.1f | FIX-PEMILIK-FK nearest=%d | FK-memory=%.0fs | FIX-UMUMKAN strong=%d | moving_grace=%.0f parked_grace=%.0f | frag_ratio=%.2f depart_ratio=%.2f depart_grace=%.0f | dock_min_ratio=%.3f | FIX-PINDAH hits=%d window=%.0f miss_presence=%d | roi_ext_top=%.2f edge_margin=%.2f edge_park=%.0f | PRESENSI-TUTUP occl=%.0f cover=%.2f | FK-VETO=%d conf>=%.2f | FK-HIJAU>=%.1f%% | SESI-GANDA=%.0f | FOTO-FB=%d | PC-SETELAH-FK=%d | FK-OVL-VETO=%d max=%.2f",
                 cfg.match_size_ratio, cfg.fk_owner_nearest, cfg.fk_mem_sec, cfg.announce_strong, cfg.moving_grace, cfg.parked_grace,
                 cfg.frag_ratio, cfg.depart_ratio, cfg.depart_grace, cfg.dock_min_ratio, cfg.relocate_hits, cfg.relocate_window, cfg.relocate_presence_misses, cfg.roi_ext_top, cfg.edge_margin, cfg.edge_park, cfg.presence_occl, cfg.presence_occl_cover, cfg.fk_veto, cfg.fk_veto_conf, cfg.green_big_pct, cfg.open_guard, cfg.foto_fb, cfg.pc_after_fk, cfg.fk_ovl_veto, cfg.fk_ovl_max)
        if cfg.fk_llr <= 0:
            log.warning("CAM1_FK_LLR=%.2f -> bobot arah forklift NOL: gerakan forklift (termasuk lokasi tumpukan) TIDAK ikut menentukan status", cfg.fk_llr)
        if BED_BOX != (0.2, 0.3, 0.95, 0.85):
            log.warning("CAM1_BED_BOX=%s BUKAN geometri bawaan: cargo_model.pkl harus dilatih ulang dengan potongan yang sama "
                        "(kumpulan sampel di CAM1_SAMPLE_DIR), kalau tidak skor bak tidak dapat dipercaya", ",".join(f"{v:g}" for v in BED_BOX))
        self.sessions = []
        self.pending = []                   # klaster kandidat sebelum sesi dibuka
        self.seq = 0
        self.stats = dict(masuk=0, selesai=0, dibuang=0)
        self.last_fk = []                   # kotak forklift frame terakhir (overlay)
        self.rej = {}                       # diagnostik: alasan kendaraan (car/truck) tidak jadi kandidat/sesi -> [jumlah, contoh]
        self._fk_mem = []                   # [PATCH-1] [(waktu, kotak forklift)] beberapa detik terakhir
        self._fk_veto_mem = []              # [FK-VETO] [(waktu, kotak, conf)] forklift non-oranye yang dibuang gerbang FK-HANTU
        self.fk_veto_n = 0

    @property
    def s(self):
        """Kompatibilitas: sesi pertama (atau None bila tidak ada sesi)."""
        return self.sessions[0] if self.sessions else None

    # ---- forklift -------------------------------------------------------------------------
    def _forklift_boxes(self, frame):
        """Blob oranye besar di dalam ROI = forklift (berapa pun ukurannya). Mengembalikan list kotak (frame penuh)."""
        c, z = self.cfg, self.zones
        if not c.fk_blob:
            return []
        x1, y1, x2, y2 = z.roi
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        rh, rw = roi.shape[:2]
        sc = 480.0 / rw if rw > 480 else 1.0
        small = cv2.resize(roi, (max(1, int(rw * sc)), max(1, int(rh * sc))), interpolation=cv2.INTER_AREA) if sc < 1.0 else roi
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (3, 140, 120), (20, 255, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        sh, sw = m.shape[:2]
        total = float(sw * sh)
        boxes = []
        for i in range(1, n):
            bx, by, bw, bh, a = [int(v) for v in stats[i]]
            if a < c.fk_min_area * total or bw * bh > c.fk_max_area * total:
                continue
            boxes.append([bx, by, bx + bw, by + bh])
        # gabungkan potongan satu forklift (kanopi + badan terpisah oleh operator): tumpang-tindih horizontal & jarak vertikal wajar
        merged = True
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    ox = min(a[2], b[2]) - max(a[0], b[0])
                    gap_y = max(a[1], b[1]) - min(a[3], b[3])
                    wmax = max(a[2] - a[0], b[2] - b[0])
                    wmin = min(a[2] - a[0], b[2] - b[0])
                    if ox >= 0.5 * wmin and gap_y <= 0.8 * wmax:
                        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        out = []
        for b in boxes:
            fb = (x1 + b[0] / sc, y1 + b[1] / sc, x1 + b[2] / sc, y1 + b[3] / sc)
            fb = expand_box(fb, c.fk_expand)
            cx, cy = center(fb)
            if z.in_area(cx, cy):
                out.append(fb)
        return out

    def _classify(self, frame, d, fks, ratio):
        """'forklift' = buang | 'merged' = kotak gabungan forklift+kendaraan | 'ok' = kendaraan bersih."""
        c = self.cfg
        a_d = area(d)
        for fk in fks:
            inter = inter_area(d, fk)
            if inter <= 0:
                continue
            a_f = area(fk)
            if inter / a_d >= 0.6 or iou(d, fk) >= 0.4:
                return "forklift"
            if inter / a_f >= 0.6:
                return "merged" if a_d > 1.3 * a_f else "forklift"
        # [FIX-TRUK-DIBUANG] kotak yang menumpuk kuat (IoU>=0.5) dengan kendaraan terkonfirmasi = kendaraan itu sendiri. Oranye dari
        # forklift di sekitarnya tidak boleh membuat truk dibuang sebagai forklift (penyebab sesi hilang lalu "diambil alih" benda lain).
        if any(s_.announced and iou(d, r) >= 0.5
               for s_ in self.sessions for r in ([s_.bbox] + ([s_.last_parked_bbox] if s_.last_parked_bbox else []))):
            return "ok"
        # cadangan (forklift terlalu kecil/pudar untuk blob): kotak kecil + oranye
        o = orange_pct(crop_vehicle(frame, d))
        if ratio < c.forklift_max_area and o >= c.orange_min_pct:
            return "forklift"
        # forklift besar (dekat kamera): oranye dominan di kotak, ukuran tidak dibatasi
        if c.orange_big_pct > 0 and o >= c.orange_big_pct:
            on_session = any(
                max(iou(d, r), inter_area(d, r) / max(1.0, min(area(d), area(r)))) >= 0.5
                for s_ in self.sessions for r in ([s_.bbox] + ([s_.last_parked_bbox] if s_.last_parked_bbox else [])))
            return "merged" if on_session else "forklift"
        # [FK-HIJAU] forklift hijau-tosca (sewaan): sama seperti aturan oranye besar, tapi untuk warna hijau
        if c.green_big_pct > 0:
            _g = green_pct(crop_vehicle(frame, d))
            if _g >= c.green_big_pct:
                self.green_n = getattr(self, "green_n", 0) + 1
                if self.green_n in (1, 5, 20) or self.green_n % 100 == 0:
                    log.info("FK-HIJAU kotak '%s' dianggap forklift hijau (hijau=%.1f%% >= %.1f%%, conf YOLO=%.2f, rasio=%.3f) pusat=(%d,%d) total=%d",
                             getattr(d, "cls", "?"), _g, c.green_big_pct, float(d.conf), ratio, *center(d), self.green_n)
                on_session = any(
                    max(iou(d, r), inter_area(d, r) / max(1.0, min(area(d), area(r)))) >= 0.5
                    for s_ in self.sessions for r in ([s_.bbox] + ([s_.last_parked_bbox] if s_.last_parked_bbox else [])))
                return "merged" if on_session else "forklift"
        # [FK-VETO] forklift non-oranye yang dibuang gerbang FK-HANTU: kandidat kecil yang tertutup kotaknya = forklift itu sendiri
        if c.fk_veto > 0 and self._fk_veto_mem and ratio <= c.fk_veto_max_ratio and float(d.conf) < c.fk_veto_max_conf:
            for _t, _fb, _cf in self._fk_veto_mem:
                if inter_area(d, _fb) / max(1.0, a_d) >= c.fk_veto_cover:
                    self.fk_veto_n += 1
                    if self.fk_veto_n in (1, 5, 20) or self.fk_veto_n % 100 == 0:
                        log.info("FK-VETO kotak 'truk' dianggap forklift non-oranye (conf YOLO=%.2f, conf forklift=%.2f, rasio=%.3f) pusat=(%d,%d) total=%d",
                                 float(d.conf), _cf, ratio, *center(d), self.fk_veto_n)
                    return "forklift"
        return "ok"

    # ---- filter kandidat ------------------------------------------------------------------
    def _det_log(self, now, fks):
        """Baris DET tiap >= 2 dtk: setiap car/truck yang dilihat YOLO (kelas:conf@(x,y normal) keputusan) + pusat forklift."""
        if now - getattr(self, "_det_t", -1e9) < 2.0:
            return
        self._det_t = now
        z = self.zones
        items = " ; ".join(f"{'car' if k == 2 else 'truck'}:{cf:.2f}@({x:.2f},{y:.2f}) {v}" + ("" if r is None else f" r={r:.3f}")
                           for k, cf, x, y, v, r in (getattr(self, "_dbg_last", None) or [])) or "-"
        fk = " ".join(f"({center(b)[0] / max(1, z.w):.2f},{center(b)[1] / max(1, z.h):.2f})" for b in fks) or "-"
        # conf detektor forklift per kotak (untuk menentukan CAM1_FK_DET_CONF dari data nyata, mis. memisahkan forklift hantu)
        fkc = " ".join(f"({x:.2f},{y:.2f})={cf:.2f}" for x, y, cf in (getattr(self, "_fk_conf", None) or [])) or "-"
        log.info("DET t=%.0f | sesi=%d | kendaraan: %s | forklift: %s | fk_conf: %s", now, len(self.sessions), items, fk, fkc)

    def _rej(self, why, d, ratio=None):
        """Catat satu kendaraan (car/truck) yang ditolak + contoh terakhir (conf, rasio luas, pusat x,y piksel)."""
        r = self.rej.setdefault(why, [0, None])
        r[0] += 1
        cx, cy = center(d)
        r[1] = (round(float(d.conf), 2), None if ratio is None else round(float(ratio), 3), int(cx), int(cy))

    def rej_summary(self):
        return " | ".join(f"{k}={v[0]} (contoh conf={v[1][0]} rasio={v[1][1]} pusat=({v[1][2]},{v[1][3]}))"
                          for k, v in sorted(self.rej.items(), key=lambda kv: -kv[1][0])) or "-"

    def _candidates(self, frame, dets, fks):
        z, c = self.zones, self.cfg
        raw = []
        dbg = [] if c.det_log else None

        def note(d, verdict, ratio=None):
            if dbg is not None:
                cx_, cy_ = center(d)
                dbg.append((int(d.cls), float(d.conf), cx_ / max(1, z.w), cy_ / max(1, z.h), verdict, ratio))

        for d in dets:
            if d.cls not in (2, 7):
                continue
            if d.conf < c.track_conf:
                self._rej("conf<track_conf", d)
                note(d, "conf_rendah")
                continue
            cx, cy = center(d)
            outside = False
            edge = False
            if (not z.in_area(cx, cy) and c.edge_margin >= 0 and z.near_area(cx, cy, c.edge_margin * max(1, z.w))
                    and not (c.zone_hyst > 0 and self._overlaps_confirmed_session(d))):
                edge = True                   # [KENDARAAN-TEPI] di pinggir area: boleh membuka sesi, tetapi wajib berhenti >= edge_park dtk
            elif not z.in_area(cx, cy):
                # [HISTERESIS-ZONA] kotak yang pusatnya di luar area TAPI bertumpuk kuat dengan kotak sesi yang sudah MASUK = kendaraan itu sendiri
                # yang bergeser ke tepi area (mis. maju ke ujung halaman). Boleh MELANJUTKAN sesi itu, tidak boleh MEMBUKA sesi baru, sehingga
                # kendaraan yang sekadar lewat di jalur pagar (luar area) tetap diabaikan.
                if c.zone_hyst > 0 and self._overlaps_confirmed_session(d):
                    outside = True
                else:
                    self._rej("luar_zona_atau_exclude", d)
                    note(d, "luar_zona")
                    continue
            ratio = area(d) / z.roi_area
            if ratio < c.min_area_ratio:
                self._rej("terlalu_kecil", d, ratio)
                note(d, "kecil", ratio)
                continue
            kind = self._classify(frame, d, fks, ratio)
            if kind == "forklift":
                self._rej("dianggap_forklift", d, ratio)
                note(d, "FORKLIFT", ratio)
                continue
            note(d, "gabungan" if kind == "merged" else ("ok_tepi" if outside else ("ok_pinggir" if edge else "ok")), ratio)
            raw.append(dict(det=d, merged=(kind == "merged"), outside=outside, edge=edge))
        self._dbg_last = dbg
        # kotak gabungan yang SUDAH punya kotak bersih di dalamnya tidak diperlukan
        clean = [r for r in raw if not r["merged"]]
        raw = [r for r in raw if not r["merged"] or
               not any(inter_area(cl["det"], r["det"]) / max(1.0, area(cl["det"])) > 0.7 for cl in clean)]
        # NMS lintas kelas (car & truck untuk kendaraan yang sama)
        raw.sort(key=lambda r: (r["merged"], -r["det"].conf))
        keep = []
        for r in raw:
            dup = False
            for k in keep:
                it = inter_area(r["det"], k["det"])
                if iou(r["det"], k["det"]) > 0.5 or it / max(1.0, min(area(r["det"]), area(k["det"]))) > 0.8:
                    dup = True
                    break
            if not dup:
                keep.append(r)
        return keep

    def _overlaps_confirmed_session(self, d):
        bb = tuple(d[:4])
        for s_ in self.sessions:
            if not s_.announced:
                continue
            for r in (s_.bbox, s_.last_parked_bbox):
                if r is not None and (iou(bb, r) >= 0.4 or inter_area(bb, r) / max(1.0, min(area(bb), area(r))) >= 0.6):
                    return True
        return False

    # ---- gerak / parkir -------------------------------------------------------------------
    def _update_motion(self, s, now, bbox):
        c = self.cfg
        cx, cy = center(bbox)
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        s.moving_hist.append((now, cx, cy, size))
        while s.moving_hist and now - s.moving_hist[0][0] > c.stat_window:
            s.moving_hist.popleft()
        span = s.moving_hist[-1][0] - s.moving_hist[0][0]
        if span < c.stat_window * 0.7:
            return  # riwayat belum cukup, pertahankan status sebelumnya
        xs, ys = [h[1] for h in s.moving_hist], [h[2] for h in s.moving_hist]
        move = max(max(xs) - min(xs), max(ys) - min(ys))
        parked = move <= c.stat_move_ratio * size
        if not parked and s.parked and c.park_hold_ratio > 0 and s.last_parked_bbox is not None:
            pcx, pcy = center(s.last_parked_bbox)
            if max(abs(cx - pcx), abs(cy - pcy)) <= c.park_hold_ratio * size:
                return   # masih di titik parkir yang sama: goyangan kotak, bukan bergerak; pertahankan status parkir
        if parked and not s.parked:
            s.parked_since = now
        if not parked:
            s.parked_since = None
        s.parked = parked
        if parked:
            s.last_parked_bbox = bbox

    # ---- pencocokan sesi <-> kandidat ----------------------------------------------------
    def _fragment_of(self, s, cd):
        """[FIX-FRAGMEN] (rasio_luas, porsi_di_dalam) bila kandidat adalah kotak kecil di dalam truk yang sedang parkir & terkonfirmasi, selain itu None."""
        c = self.cfg
        if c.frag_ratio <= 0 or cd["merged"] or not s.parked or not self._confirmed(s):
            return None
        d = cd["det"]
        a = max(1.0, area(d))
        need = c.fragment_contain if c.fragment_contain > 0 else 0.60
        for ref in (s.bbox, s.last_parked_bbox):
            if ref is None:
                continue
            a_ref = max(1.0, area(ref))
            inside = inter_area(d, ref) / a
            if a < c.frag_ratio * a_ref and inside >= need:
                return a / a_ref, inside
        return None

    def _match_score(self, s, cd, now, frame):
        c, d = self.cfg, cd["det"]
        _fr = self._fragment_of(s, cd)
        if _fr is not None:
            s.frag_n += 1
            if s.frag_n in (1, 50) or s.frag_n % 500 == 0:
                log.info("FRAG-DIABAIKAN %s | kotak kecil (luas %.0f%% kotak truk, %.0f%% di dalamnya, conf=%.2f) tidak disambung ke sesi | total=%d",
                         s.vid, _fr[0] * 100, _fr[1] * 100, d.conf, s.frag_n)
            return None
        refs = [s.bbox] + ([s.last_parked_bbox] if s.last_parked_bbox else [])
        # [FIX-UKURAN] kotak yang ukurannya jauh beda dari acuan = benda lain (mis. forklift berpalet r=0.31 vs truk r=0.04), bukan kendaraan ini
        if c.match_size_ratio > 0:
            # [FIX-UKURAN-2] selama deteksi masih menyambung (jeda <= swap_gap) bandingkan dengan kotak TERAKHIR, bukan titik parkir lama:
            # truk yang melaju mendekati kamera membesar bertahap (tiap langkah <3x) tapi bisa 20x dari titik parkirnya yang jauh.
            if now - s.t_last <= c.swap_gap:
                a_ref = max(1.0, area(s.bbox))
            else:
                a_ref = max(1.0, area(s.last_parked_bbox or s.bbox))
            q = max(1.0, area(d)) / a_ref
            if q > c.match_size_ratio or q < 1.0 / c.match_size_ratio:
                return None
        best_iou = max(iou(d, r) for r in refs)
        gap = now - s.t_last
        if gap <= c.swap_gap:
            if best_iou >= 0.10:
                return 1.0 + best_iou
            size = max(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1])
            (cx, cy), (sx, sy) = center(d), center(s.bbox)
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 / max(size, 1.0)
            return (1.0 - dist) if dist <= 0.8 else None
        # setelah jeda: harus di tempat yang sama DAN tampilannya mirip, kalau tidak = kendaraan lain
        if best_iou < 0.30:
            return None
        if not cd["merged"] and s.sig is not None:
            if appearance_dist(s.sig, appearance_sig(frame, tuple(d[:4]))) > c.appearance_max_dist:
                return None
        return best_iou

    # ---- sampel muatan --------------------------------------------------------------------
    def _maybe_sample(self, s, frame, now, bbox, dets, fks):
        c = self.cfg
        pre = c.pre_park_window > 0 and now - s.t_in <= c.pre_park_window   # [PATCH-PREPARK]
        if not pre and (not s.parked or s.parked_since is None or now - s.parked_since < c.settle):
            s.bump("belum_parkir")
            return
        if now - s.last_sample_t < c.sample_interval:
            return
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bed_box = (bbox[0] + BED_BOX[0] * w, bbox[1] + BED_BOX[1] * h, bbox[0] + BED_BOX[2] * w, bbox[1] + BED_BOX[3] * h)
        # orang di dalam kotak kendaraan (atau hanya di area bak bila CAM1_PERSON_BED_ONLY=1) -> skor bak tercemar, lewati sampel ini
        if c.person_bed_only:
            pad_x, pad_y = c.person_bed_pad * w, c.person_bed_pad * h
            zone = (bed_box[0] - pad_x, bed_box[1] - pad_y, bed_box[2] + pad_x, bed_box[3] + pad_y)
        else:
            zone = bbox
        for d in dets:
            if d.cls == 0:
                px, py = center(d)
                if zone[0] <= px <= zone[2] and zone[1] <= py <= zone[3]:
                    s.bump("orang_di_kotak")
                    return
        # forklift menutupi bak -> skor tercemar (garpu/oranye), lewati
        ba = max(1.0, area(bed_box))
        if any(inter_area(bed_box, fk) / ba > c.bed_fk_overlap for fk in fks):
            s.bump("forklift_menutup_bak")
            return
        truck = crop_vehicle(frame, bbox, pad=0.03)
        if truck.size == 0:
            return
        bed = cargo_bed(truck)
        bh, bw = bed.shape[:2]
        if min(bh, bw) < c.min_crop_side or bh * bw < c.min_crop_pixels:
            s.bump("crop_kecil")
            return
        s.last_sample_t = now
        try:
            p = self.scorer(bed)
        except Exception as e:
            log.warning("scorer error: %s", e)
            p = None
        if c.sample_dir:
            try:
                n_all = getattr(s, "dump_all", 0) + 1                      # semua sampel (untuk mode lama)
                s.dump_all = n_all
                n = getattr(s, "dump_n", 0)                                # jumlah yang tersimpan
                if c.sample_dump_interval <= 0:
                    do = n_all <= 150
                else:
                    gap = now - getattr(s, "dump_t", -1e9)
                    lp = getattr(s, "dump_p", None)
                    changed = p is not None and lp is not None and abs(p - lp) >= 0.15
                    do = n < c.sample_dump_max and (gap >= c.sample_dump_interval or (changed and gap >= 2.0))
                if do:
                    n += 1
                    s.dump_n, s.dump_t = n, now
                    if p is not None:
                        s.dump_p = p
                    dd = os.path.join(c.sample_dir, s.vid)
                    os.makedirs(dd, exist_ok=True)
                    cv2.imwrite(os.path.join(dd, f"{s.vid}_{n:03d}_t{now - s.t_in:.0f}_p{'NA' if p is None else format(p, '.2f')}.jpg"), bed)
            except Exception:
                pass
        if p is None:
            s.bump("ditolak_classifier")
            return  # ditolak classifier (forklift / boks tertutup / ragu) -> TIDAK ditebak
        s.samples.append(dict(t=now, p=float(p), sz=float(max(w, h)), parked=bool(s.parked),
                             jpg=to_jpg(crop_vehicle(frame, bbox, 0.08), 640, 78)))
        if s.photo_arrival is None:
            s.photo_arrival = s.samples[-1]["jpg"]
        if len(s.samples) > c.max_samples:  # kompaksi: buang tiap sampel ganjil di bagian tengah
            k = c.phase_k
            s.samples = s.samples[:k] + s.samples[k:-k][::2] + s.samples[-k:]
        s.info = decide(s.samples, c, s.fk_seconds, self._fk_obs(s), self._fk_dist(s), self._fk_stage(s))
        log.debug("[%s] sampel P=%.2f (n=%d)", s.vid, p, len(s.samples))

    # ---- siklus hidup sesi ----------------------------------------------------------------
    def _new_vid(self, t):
        self.seq += 1
        return f"CAM1_{datetime.fromtimestamp(t).strftime('%Y%m%d_%H%M%S')}_{self.seq:04d}"

    def _announce_entry(self, s):
        s.announced = True
        early = [x["p"] for x in s.samples[:self.cfg.phase_k]]
        muatan = p_to_status(float(median(early))) if len(early) >= 2 else "-"
        photo = s.photo_in or s.photo_arrival or s.photo_last   # [PATCH-FOTO] foto saat pertama terlihat, bukan sampel bak pertama
        self._save_photo(s.vid, "masuk", photo)
        payload = self._base_payload(s, "MASUK", s.t_in, photo)
        payload.update(status_muatan=muatan, timestamp_masuk=fmt_ts(s.t_in))
        self.sender.send(payload)
        self.stats["masuk"] += 1
        log.info("MASUK  %s | %s | muatan awal=%s", s.vid, fmt_ts(s.t_in), muatan)

    def _base_payload(self, s, kind, t, photo_jpg):
        b = [int(v) for v in s.bbox]
        return {
            "timestamp": fmt_ts(t),
            "kamera_id": self.cfg.camera_id,
            "kamera_nama": self.cfg.camera_name,
            "jenis_kendaraan": "Truk" if s.cls == 7 else "Mobil/Pickup",
            "confidence": round(float(s.max_conf), 4),
            "bbox_x1": b[0], "bbox_y1": b[1], "bbox_x2": b[2], "bbox_y2": b[3],
            "gambar_base64": b64(photo_jpg),
            "jenis_kejadian": kind,
            "vehicle_id": s.vid,
            "track_id": s.vid,
            "event_id": f"{s.vid}_{kind}",
            # kolom lama dibiarkan kosong supaya workflow n8n/DB yang sudah ada tidak patah (plat dihapus)
            "warna": "-", "rgb_r": 0, "rgb_g": 0, "rgb_b": 0,
            "arah_kendaraan": "-", "arah_confidence": 0.0, "plat_nomor": None, "plat_confidence": 0.0,
        }

    def _save_photo(self, vid, name, jpg):
        if not jpg or not self.cfg.evidence_dir:
            return None
        try:
            d = os.path.join(self.cfg.evidence_dir, vid)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{name}.jpg")
            with open(path, "wb") as f:
                f.write(jpg)
            return path
        except Exception as e:
            log.warning("gagal simpan foto: %s", e)
            return None

    def _close(self, s, reason, t_out=None):
        if s in self.sessions:
            self.sessions.remove(s)
        t_out = t_out if t_out is not None else s.t_last
        if not s.announced:
            self.stats["dibuang"] += 1
            _ph = self._save_photo("_dibuang", s.vid, s.photo_last or s.photo_in)
            log.info("sesi %s dibuang (hanya %.1f dtk, belum dikonfirmasi MASUK) | alasan=%s kelas=%s conf_maks=%.2f deteksi_kuat=%d "
                     "pusat=(%d,%d) parkir=%s foto=%s", s.vid, t_out - s.t_in, reason, s.cls, s.max_conf, s.strong_hits,
                     *center(s.bbox), s.parked, _ph or "-")
            return

        docked = (s.t_dock_in is not None) if len(self.zones.data.get("dock") or []) >= 3 else None
        res = decide(s.samples, self.cfg, s.fk_seconds, self._fk_obs(s), self._fk_dist(s),
                     self._fk_stage(s), t_end=t_out, docked=docked,
                     manual=s.skip.get("orang_di_kotak", 0), final=True)
        # [ARAH-ORANG] keputusan lain "Tidak Dapat Dipastikan" -> coba arah dari orang yang membawa barang (van/boks tanpa forklift)
        if pc_allowed(res, s.fk_seconds, sum(len(tr["obs"]) for tr in s.fk_tracks), self.cfg):   # forklift sungguhan (skor banyak / dekat lama) -> jalur lama, JANGAN disentuh; hantu sesaat boleh
            try:
                for _ti, _tr in enumerate(s.pc_tracks):      # seri mentah per orang: detik:P(bawa):jarak_ke_kendaraan:S=di stage
                    log.info("ARAH-ORANG-SERI %s orang%d | %s", s.vid, _ti, " ".join(
                        f"{o[0] - s.t_in:.0f}:{o[1]:.2f}:{dd:.2f}:{'S' if sg else '-'}@{ps[0]:.2f},{ps[1]:.2f}"
                        for o, dd, sg, ps in zip(_tr["obs"], _tr["dist"], _tr["stg"], _tr["pos"])) or "-")
                for _ti, _tr in enumerate(s.pc_tracks):      # [UKURAN-ORANG] data uji: detik:lebar:tinggi (normal frame) per orang
                    if _tr.get("wh"):
                        log.info("ARAH-ORANG-UKURAN %s orang%d | %s", s.vid, _ti, " ".join(
                            f"{o[0] - s.t_in:.0f}:{w_:.3f}:{h_:.3f}" for o, (w_, h_) in zip(_tr["obs"], _tr["wh"])))
                log.info("ARAH-ORANG-KENDARAAN %s | kotak_norm=(%.3f,%.3f,%.3f,%.3f)", s.vid, s.bbox[0] / max(1, self.zones.w), s.bbox[1] / max(1, self.zones.h),
                         s.bbox[2] / max(1, self.zones.w), s.bbox[3] / max(1, self.zones.h))
                # Titik pintu/drop-off lebih spesifik daripada centroid stage: pada kamera ini
                # arah kendaraan->pintu adalah Unloading, arah sebaliknya adalah Loading.
                camera_axis = None
                unload_target = self.zones.data.get("unload_target")
                stage_poly = self.zones.data.get("stage") or []
                if len(unload_target or []) == 2:
                    sx, sy = float(unload_target[0]), float(unload_target[1])
                    target_name = "pintu-unloading"
                elif len(stage_poly) >= 3:
                    sx = sum(p[0] for p in stage_poly) / len(stage_poly)
                    sy = sum(p[1] for p in stage_poly) / len(stage_poly)
                    target_name = "centroid-stage (fallback)"
                else:
                    sx = sy = None
                    target_name = "tidak tersedia"
                if sx is not None and self.zones.w > 0 and self.zones.h > 0:
                    vx = ((s.bbox[0] + s.bbox[2]) / 2.0) / self.zones.w
                    vy = ((s.bbox[1] + s.bbox[3]) / 2.0) / self.zones.h
                    ax, ay = sx - vx, sy - vy
                    alen = math.hypot(ax, ay)
                    if alen > 1e-6:
                        camera_axis = (ax / alen, ay / alen)
                        log.info("ARAH-ORANG-SUMBU %s | target=%s sumbu=(%.3f,%.3f); positif=Unloading",
                                 s.vid, target_name, *camera_axis)
                _pd = person_direction(s.pc_tracks, self.cfg, camera_axis=camera_axis)
                log.info("ARAH-ORANG %s | track=%d skor=%d | %s | status=%s conf=%.2f | %s", s.vid, len(s.pc_tracks),
                         sum(len(tr["obs"]) for tr in s.pc_tracks), _pd["reason"], _pd["status"], _pd["confidence"],
                         "; ".join(_pd["detail"][:12]) or "-")
                if not _pd["status"] and self.cfg.pc_dist and not s.samples and not any(tr["obs"] for tr in s.fk_tracks):   # diagnosis saja; jarak/keberadaan orang tanpa gerakan membawa tidak menentukan arah
                    _, _dr = person_distance_pattern(s.pc_tracks, self.cfg)
                    log.info("POLA-JARAK %s | %s (diagnostik saja; bukan vote arah)", s.vid, _dr)
                if _pd["status"]:
                    res["status"] = _pd["status"]
                    res["source"] = "orang_arah"
                    res["confidence"] = _pd["confidence"]
                    res["review"] = True                      # arah dari orang selalu minta review sampai terbukti akurat
                    res["reason"] = (res.get("reason", "") + "; " if res.get("reason") else "") + _pd["reason"]
                elif (res.get("status") in (None, UNSURE) and _pd["L"] == 0 and _pd["U"] <= 1
                      and _pd["stage_drop_u"] >= 1 and s.stg_sec >= self.cfg.pc_stage_min_sec
                      and s.fk_seconds < self.cfg.fk_near_min):
                    # Satu track Unloading boleh diperkuat aksi menaruh di stage + aktivitas lama.
                    # Tanpa vote arah pun fallback tetap memerlukan bukti stage; Loading tidak ditimpa.
                    res["status"] = UNLOADING
                    res["source"] = "orang_arah_stage"
                    res["confidence"] = 0.35
                    res["review"] = True
                    _arah_note = (f"{_pd['U']} track memberi arah Unloading" if _pd["U"]
                                  else "tanpa vote arah perjalanan membawa barang")
                    res["reason"] = (res.get("reason", "") + "; " if res.get("reason") else "") + (
                        f"{_arah_note}, {_pd['stage_drop_u']} aksi menaruh di stage, "
                        f"aktivitas stage {s.stg_sec:.0f} dtk (>= {self.cfg.pc_stage_min_sec:.0f} dtk) "
                        "-> fallback Unloading; perlu review")
                    log.info("ARAH-ORANG-STAGE %s | %s", s.vid, res["reason"])
            except Exception as _ex:
                log.warning("ARAH-ORANG gagal saat tutup sesi: %s", _ex)
        log.info("STAGE-DIAG %s | orang_di_stage=%.0f dtk (maks %d) | status_awal=%s review=%s fk=%.0f dtk", s.vid, s.stg_sec, s.stg_max, res.get("status"), res.get("review"), s.fk_seconds)
        if (s.stg_sec >= self.cfg.stage_act_min and res.get("source") == "skor"
                and res.get("status") in (LOADING, UNLOADING) and s.fk_seconds < self.cfg.fk_near_min):
            res["review"] = True
            res["reason"] = (res.get("reason", "") + "; aktivitas di stage tanpa arah forklift: "
                             "arah skor bak tidak cukup untuk mengesahkan status -> perlu review")
        # [FOTO-AWAL-AKHIR] arah dari label muatan awal -> akhir (hanya bila saklar CAM1_FOTO_FB menyala). Tidak mengubah sesi yang sudah
        # diputuskan jalur lain; hanya (a) mempertahankan tebakan lemah yang SEARAH, atau (b) mengisi 'Tidak Dapat Dipastikan'. Selalu PERLU REVIEW.
        _fa = foto_arah(res, self.cfg)
        _fa_keep = False
        if _fa is not None:
            if res.get("status") == _fa and res.get("source") not in ("orang_arah", "orang_jarak"):
                _fa_keep = True
                res["review"] = True
                res["reason"] = (res.get("reason", "") + "; " if res.get("reason") else "") + \
                    f"dipertahankan: muatan awal {p_to_status(res['early'])} -> akhir {p_to_status(res['late'])} searah {_fa}"
                log.info("FOTO-AWAL-AKHIR %s | tebakan %s dipertahankan (muatan awal %.2f -> akhir %.2f)", s.vid, _fa, res["early"], res["late"])
            elif res.get("status") in (None, UNSURE):
                res["status"], res["source"] = _fa, "foto_awal_akhir"
                res["confidence"] = min(float(res.get("confidence") or 0.0) or 0.4, 0.45)
                res["review"] = True
                res["reason"] = (res.get("reason", "") + "; " if res.get("reason") else "") + \
                    f"arah dari muatan awal {p_to_status(res['early'])} ({res['early']:.2f}) -> akhir {p_to_status(res['late'])} ({res['late']:.2f}) -> {_fa}; perlu review"
                _fa_keep = True
                log.info("FOTO-AWAL-AKHIR %s | status diisi %s (muatan awal %.2f -> akhir %.2f)", s.vid, _fa, res["early"], res["late"])
        # [TINGKAT-A] bukti arah tidak ada, tapi truk parkir dan ada orang bekerja di stage -> catat sebagai AKTIVITAS (arah tidak ditebak)
        _lemah = (self.cfg.stage_act_down and not _fa_keep and res.get("review") and res.get("status") in (LOADING, UNLOADING) and res.get("source") not in ("orang_arah", "orang_arah_stage", "orang_jarak")
                  and s.fk_seconds < self.cfg.fk_near_min)          # tebakan lemah tanpa forklift -> jangan dipertahankan
        if ((res.get("status") in (None, UNSURE) or _lemah) and self.cfg.stage_act_min > 0 and s.stg_sec >= self.cfg.stage_act_min):
            _tebakan = f" (tebakan sistem sebelumnya: {res['status']}, keyakinan {res['confidence']:.2f}, tidak dipakai)" if _lemah else ""
            res["status"] = STAGE_ACT
            res["source"] = "stage_orang"
            res["confidence"] = 0.0
            res["review"] = True
            res["reason"] = (f"truk parkir; ada {s.stg_max} orang (maks bersamaan) di zona stage selama {s.stg_sec:.0f} dtk; "
                             f"arah muat/bongkar tidak bisa dipastikan (boks tertutup / tanpa forklift) -> perlu review{_tebakan}")
            log.info("STAGE-AKTIVITAS %s | %s", s.vid, res["reason"])
        if (self.cfg.drop_noevidence and not self.cfg.force_decision and res["status"] == UNSURE and res.get("source") != "stage_orang" and not s.samples
                and not any(tr["obs"] for tr in s.fk_tracks)):
            # [FIX-MASUK-YATIM] sesi ini SUDAH mengirim MASUK. Dulu di sini langsung return -> MASUK menggantung tanpa Keluar.
            # Sekarang SIKLUS_SELESAI tetap dikirim, ditandai perlu review + kemungkinan salah deteksi.
            log.info("sesi %s TANPA bukti sama sekali (0 sampel bak, 0 skor forklift) | alasan=%s durasi=%.0f dtk pusat=(%d,%d) -- "
                     "kemungkinan objek diam/fragmen; MASUK sudah terkirim, jadi SIKLUS_SELESAI tetap dikirim (perlu review)",
                     s.vid, reason, t_out - s.t_in, *center(s.bbox))
            res["review"] = True
            res["reason"] = ((res.get("reason", "") + "; ") if res.get("reason") else "") + \
                "tanpa bukti sama sekali (0 sampel bak, 0 skor forklift); kemungkinan objek diam/fragmen, bukan kendaraan yang dimuat -> perlu review"
        if s.tl:
            log.info("TIMELINE %s | t_sejak_masuk:[P=parkir m=bergerak][F=forklift dekat][O=orang di kotak][x=tak terlihat]skor_bak | %s",
                     s.vid, " ".join(s.tl))
        log.info("SKOR-BAK %s | n=%d | detik_sejak_masuk:P = %s", s.vid, len(s.samples),
                 " ".join(f"{x['t'] - s.t_in:.0f}:{x['p']:.2f}" for x in s.samples))
        for i, tr in enumerate(s.fk_tracks):
            ps = [round(p, 2) for _, p in tr["obs"]]
            log.info("FK-TRACK %s #%d | %d skor | %s | states=%s", s.vid, i, len(ps), ps,
                _fk_states(tr["obs"], self.cfg))
        if not s.fk_tracks:
            log.info("FK-TRACK %s: tidak ada track forklift", s.vid)
        try:
            self._autolabel_forklift(s, res)
        except Exception as e:
            log.warning("autolabel forklift gagal: %s", e)
        photo_out = s.photo_last
        photo_act = None
        if s.samples:
            idx = res["change_idx"] if res["change_idx"] is not None else len(s.samples) // 2
            photo_act = s.samples[min(idx, len(s.samples) - 1)]["jpg"]
        if photo_act is None:
            photo_act = s.photo_fk  # tanpa sampel muatan: pakai foto saat forklift di dekat kendaraan
        if photo_act is None:
            photo_act = s.photo_stg  # [TINGKAT-A] foto saat ada orang di stage
        photo_in = s.photo_in or s.photo_arrival or s.photo_last   # [PATCH-FOTO]
        paths = {n: self._save_photo(s.vid, n, j) for n, j in
                 (("masuk", photo_in), ("aktivitas", photo_act), ("keluar", photo_out))}

        fkd = fk_direction(self._fk_obs(s), self.cfg, self._fk_dist(s), self._fk_stage(s))
        log.info("FK-GEO %s | taruh/ambil=%d/%d efektif=%d/%d | selisih_jarak(ambil-taruh)=%s | transisi(jenis#track@jarak)=%s | tumpukan(jenis:ya/tidak)=%s | suara(L/U)=%s | %s", s.vid,
                 fkd["drops"], fkd["picks"], fkd["eff_drops"], fkd["eff_picks"], fkd["geo"],
                 ",".join(f"{k}#{t}@{d}" for k, t, d in fkd["trans"]) or "-",
                 ",".join(f"{k}:{'ya' if at else ('tidak' if at is not None else 'NA')}" for k, at in fkd["stage"]) or "-",
                 fkd["stage_votes"], fkd["reason"])
        try:
            evs = sorted(fkd.get("events") or [], key=lambda e: e[0])
            log.info("FK-EVENTS %s | detik_sejak_masuk:jenis@jarak_truk@di_tumpukan#track = %s", s.vid,
                     " ".join(f"{e[0] - s.t_in:.0f}:{e[1]}@{'-' if e[3] is None else format(e[3], '.2f')}@"
                              f"{'NA' if e[2] is None else ('ya' if e[2] else 'tidak')}#{e[4]}" for e in evs) or "-")
        except Exception as ex:
            log.warning("FK-EVENTS gagal: %s", ex)
        try:
            self._dump_transitions(s)
        except Exception as e:
            log.warning("simpan foto transisi gagal: %s", e)
        diag = self._diag(s)
        log.info("DIAG %s | sampel_ok=%d dilewati[%s] | forklift: track=%d skor n=%s min/med/max=%s/%s/%s | model_forklift=%s",
                 s.vid, len(s.samples), diag["skip"] or "-", len(s.fk_tracks), diag["n"], diag["min"], diag["med"],
                 diag["max"], "ADA" if self.fk_scorer is not None else "TIDAK ADA (arah forklift mati)")
        dur = max(0.0, t_out - s.t_in)
        dock_dur = (s.t_dock_out - s.t_dock_in) if (s.t_dock_in and s.t_dock_out) else 0.0
        payload = self._base_payload(s, "SIKLUS_SELESAI", t_out, photo_out)
        payload.pop("timestamp", None)
        payload.pop("arah_kendaraan", None)
        payload.update(
            timestamp_masuk=fmt_ts(s.t_in), timestamp_keluar=fmt_ts(t_out),
            status_muatan_awal=p_to_status(res["early"]), status_muatan_akhir=p_to_status(res["late"]),
            durasi_detik=round(dur, 2), durasi_menit=round(dur / 60.0, 2),
            waktu_dock_mulai=fmt_ts(s.t_dock_in) if s.t_dock_in else None,
            waktu_dock_selesai=fmt_ts(s.t_dock_out) if s.t_dock_out else None,
            durasi_dock_detik=round(dock_dur, 2),
            status_aktivitas=res["status"], status_loading=res["status"],
            aktivitas_alasan=res["reason"], aktivitas_sumber=res["source"],
            aktivitas_confidence=res["confidence"], perlu_review=bool(res["review"]), aktivitas_bukti_tipis=bool(res.get("thin", False)),
            skor_muatan_awal=res["early"], skor_muatan_akhir=res["late"],
            aktivitas_sampel_awal=res["n_phase"], aktivitas_sampel_akhir=res["n_phase"], jumlah_sampel=res["n"],
            forklift_dekat_detik=round(s.fk_seconds, 1),
            forklift_taruh=fkd["drops"], forklift_ambil=fkd["picks"],
            # diagnostik: kenapa bukti kosong/lemah (dipakai untuk menentukan perbaikan berikutnya)
            forklift_track=len(s.fk_tracks), forklift_skor_n=diag["n"], forklift_skor_min=diag["min"],
            forklift_skor_median=diag["med"], forklift_skor_max=diag["max"],
            diag_sampel=diag["skip"],
            alasan_tutup=reason,
            foto_masuk_base64=b64(photo_in), foto_aktivitas_base64=b64(photo_act), foto_keluar_base64=b64(photo_out),
            foto_masuk_path=paths["masuk"], foto_aktivitas_path=paths["aktivitas"], foto_keluar_path=paths["keluar"],
        )
        self.sender.send(payload)
        self.stats["selesai"] += 1
        log.info("KELUAR %s | masuk %s keluar %s | %.0f dtk | STATUS=%s%s | forklift dekat %.0f dtk | %s", s.vid,
                 fmt_ts(s.t_in), fmt_ts(t_out), dur, res["status"], " (PERLU REVIEW)" if res["review"] else "",
                 s.fk_seconds, res["reason"])

    # ---- pembaruan sesi -------------------------------------------------------------------
    @staticmethod
    def _fk_obs(s):
        return [tr["obs"] for tr in s.fk_tracks]

    @staticmethod
    def _fk_dist(s):
        return [tr.get("dist") for tr in s.fk_tracks]

    @staticmethod
    def _fk_stage(s):
        return [tr.get("stg") for tr in s.fk_tracks]

    def _diag(self, s):
        """Ringkasan diagnostik sesi (alasan sampel dilewati + sebaran skor 'bawa' forklift). Nilai datar, aman untuk n8n/DB."""
        ps = [p for tr in s.fk_tracks for _, p in tr["obs"]]
        skip = dict(s.skip)
        if s.merged_n:
            skip["kotak_gabungan_forklift"] = s.merged_n
        if self.fk_scorer is None:
            skip["model_forklift_tidak_ada"] = 1
        return dict(
            n=len(ps),
            min=round(min(ps), 2) if ps else None,
            med=round(float(median(ps)), 2) if ps else None,
            max=round(max(ps), 2) if ps else None,
            skip=",".join(f"{k}={v}" for k, v in sorted(skip.items(), key=lambda kv: -kv[1])),
        )

    def _tl_tick(self, s, now, dets, fks, seen):
        """[PATCH-4] Catat keadaan sesi tiap tl_interval dtk: parkir/bergerak, forklift dekat, orang di kotak, tak terlihat, skor bak."""
        c = self.cfg
        if not c.tl_log or now - s.tl_t < c.tl_interval:
            return
        s.tl_t = now
        near = expand_box(s.bbox, 0.15)
        f = any(inter_area(near, fk) > 0 for fk in fks)
        o = False
        for d in dets:
            if d.cls == 0:
                px, py = center(d)
                if s.bbox[0] <= px <= s.bbox[2] and s.bbox[1] <= py <= s.bbox[3]:
                    o = True
                    break
        p = s.samples[-1]["p"] if s.samples and now - s.samples[-1]["t"] <= c.tl_interval + 0.5 else None
        s.tl.append(f"{now - s.t_in:.0f}:{'P' if s.parked else 'm'}{'F' if f else '-'}{'O' if o else '-'}{'' if seen else 'x'}"
                    + ("" if p is None else format(p, ".2f")))

    def _collect_persons(self, s, frame, now, dets):
        """[DATA-ORANG] Simpan crop orang di dekat kendaraan ke CAM1_PERSON_DIR/<id sesi>/. Nama berkas memuat urutan, detik sejak masuk,
        jarak pusat orang ke pusat kendaraan (dalam satuan ukuran kendaraan) dan confidence. Pasif: tidak mengubah keputusan apa pun."""
        c = self.cfg
        if not c.person_dir or now - getattr(s, "pc_t", -1e9) < c.person_interval or getattr(s, "pc_n", 0) >= c.person_max:
            return
        near = expand_box(s.bbox, c.person_near)
        vx, vy = center(s.bbox)
        vsize = max(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1], 1.0)
        saved = 0
        for d in dets:
            if d.cls != 0:
                continue
            px, py = center(d)
            if not (near[0] <= px <= near[2] and near[1] <= py <= near[3]):
                continue
            crop = crop_vehicle(frame, (d.x1, d.y1, d.x2, d.y2), pad=0.25)
            if crop.size == 0 or min(crop.shape[:2]) < c.person_min_side:
                continue
            dist = math.hypot(px - vx, py - vy) / vsize
            n = getattr(s, "pc_n", 0) + 1
            dd = os.path.join(c.person_dir, s.vid)
            try:
                os.makedirs(dd, exist_ok=True)
                cv2.imwrite(os.path.join(dd, f"{s.vid}_{n:04d}_t{now - s.t_in:.0f}_d{dist:.2f}_c{d.conf:.2f}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                return
            s.pc_n = n
            saved += 1
            if n == 1:
                log.info("DATA-ORANG %s: mulai menyimpan crop orang ke %s", s.vid, dd)
            if saved >= 3 or n >= c.person_max:
                break
        if saved:
            s.pc_t = now

    def _person_observe(self, s, frame, now, dets):
        """[ARAH-ORANG] Lacak tiap orang di dekat kendaraan yang sedang parkir dan beri skor P(bawa barang) per crop.
        Crop memakai pad 0.25 SAMA dengan _collect_persons (data latih person_carry_model). Tidak mengubah apa pun sampai sesi ditutup."""
        c = self.cfg
        if not c.pc_enable or self.pc_scorer is None:
            return
        near = expand_box(s.bbox, c.pc_box)
        used = set()
        for d in dets:
            if d.cls != 0 or d.conf < c.pc_conf:
                continue
            pb = (d.x1, d.y1, d.x2, d.y2)
            px, py = center(pb)
            if not (near[0] <= px <= near[2] and near[1] <= py <= near[3]):
                continue
            psz = max(1.0, pb[2] - pb[0], pb[3] - pb[1])
            best, best_d = None, None
            for i, tr in enumerate(s.pc_tracks):
                if i in used or now - tr["t_last"] > c.pc_link_gap:
                    continue
                tcx, tcy = center(tr["box"])
                dist = ((px - tcx) ** 2 + (py - tcy) ** 2) ** 0.5 / psz
                if dist <= c.pc_link_dist and (best is None or dist < best_d):
                    best, best_d = i, dist
            if best is None:
                s.pc_tracks.append(dict(box=pb, t_last=now, last_obs=-1e9, obs=[], dist=[], stg=[], pos=[], wh=[]))
                best = len(s.pc_tracks) - 1
            used.add(best)
            tr = s.pc_tracks[best]
            tr["box"], tr["t_last"] = pb, now
            if now - tr["last_obs"] < c.pc_interval:
                continue
            crop = crop_vehicle(frame, pb, 0.25)
            if crop.size == 0 or min(crop.shape[:2]) < c.person_min_side:
                continue
            tr["last_obs"] = now
            try:
                p = self.pc_scorer(crop)
            except Exception as e:
                log.warning("pc_scorer error: %s", e)
                p = None
            if p is None:
                continue
            tr["obs"].append((now, float(p)))
            # Gunakan titik kaki (tengah-bawah box): lebih mewakili lokasi orang di lantai
            # dan konsisten dengan titik yang dipakai untuk poligon stage.
            fx, fy = (d.x1 + d.x2) / 2.0, d.y2
            tr["dist"].append(point_box_dist_norm((fx, fy), s.bbox))
            tr["stg"].append(bool(self.zones.in_stage(fx, fy)) if self.zones.has_stage() else None)
            tr["pos"].append((round(fx / frame.shape[1], 3), round(fy / frame.shape[0], 3)))
            tr["wh"].append((round((pb[2] - pb[0]) / frame.shape[1], 3), round((pb[3] - pb[1]) / frame.shape[0], 3)))   # [UKURAN-ORANG] lebar/tinggi kotak orang (data uji: dus di depan badan melebarkan kotak)
            if len(tr["obs"]) > 400:
                for k in ("obs", "dist", "stg", "pos", "wh"):
                    tr[k] = tr[k][:5] + tr[k][5:-5][::2] + tr[k][-5:]

    def _fk_owner(self, fk):
        """Sesi yang paling tumpang-tindih dengan kotak forklift ini (supaya satu forklift tidak dihitung di dua sesi)."""
        best, best_a = None, 0.0
        for s in self.sessions:
            a = inter_area(expand_box(s.bbox, 0.15), fk)
            if a > best_a:
                best, best_a = s, a
        if self.cfg.fk_owner_nearest and best is not None:
            # [FIX-PEMILIK-FK] dua kendaraan berdampingan: forklift milik kendaraan yang kotaknya paling dekat dengan pusat forklift
            # (jarak titik-ke-kotak; 0 bila pusat di dalam kotak). Seri -> yang tumpang-tindihnya terbesar.
            fx, fy = center(fk)
            cands = []
            for s in self.sessions:
                if inter_area(expand_box(s.bbox, 0.15), fk) <= 0:
                    continue
                dx = max(s.bbox[0] - fx, 0.0, fx - s.bbox[2])
                dy = max(s.bbox[1] - fy, 0.0, fy - s.bbox[3])
                cands.append(((dx * dx + dy * dy) ** 0.5 / max(1.0, max(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1])),
                              -inter_area(expand_box(s.bbox, 0.15), fk), s))
            if cands:
                best = min(cands, key=lambda x: (round(x[0], 3), x[1]))[2]
        return best

    def _fk_observe(self, s, frame, now, fks):
        """Lacak tiap forklift yang dekat kendaraan ini. Bila ada model: skor 'bawa/kosong' per crop. Bila
        CAM1_FK_AUTOLABEL_DIR diisi: simpan crop kecil + jarak ke kendaraan (bahan pelabelan otomatis saat sesi ditutup)."""
        c = self.cfg
        collect = bool(c.fk_autolabel_dir)
        if self.fk_scorer is None and not collect:
            return
        near = expand_box(s.bbox, 0.15)
        used = set()
        for fk in fks:
            if inter_area(near, fk) <= 0 or self._fk_owner(fk) is not s:
                continue
            fcx, fcy = center(fk)
            fsz = max(1.0, fk[2] - fk[0], fk[3] - fk[1])
            best, best_d = None, None
            for i, tr in enumerate(s.fk_tracks):
                if i in used or now - tr["t_last"] > c.fk_link_gap:
                    continue
                tcx, tcy = center(tr["box"])
                dist = ((fcx - tcx) ** 2 + (fcy - tcy) ** 2) ** 0.5 / fsz
                if dist <= c.fk_link_dist and (best is None or dist < best_d):
                    best, best_d = i, dist
            if best is None:
                s.fk_tracks.append(dict(box=tuple(fk), t_last=now, last_obs=-1e9, obs=[], dist=[], crops=[]))
                best = len(s.fk_tracks) - 1
            used.add(best)
            tr = s.fk_tracks[best]
            tr["box"], tr["t_last"] = tuple(fk), now
            if now - tr["last_obs"] < c.fk_carry_interval:
                continue
            crop = crop_vehicle(frame, fk, c.fk_crop_pad)
            if crop.size == 0 or min(crop.shape[:2]) < c.fk_min_crop:
                continue
            tr["last_obs"] = now
            p = None
            if self.fk_scorer is not None:
                try:
                    p = self.fk_scorer(crop)
                except Exception as e:
                    log.warning("fk_scorer error: %s", e)
                    p = None
            if c.fk_sample_dir and s.fk_dump_n < 1000:
                try:
                    s.fk_dump_n += 1
                    dd = os.path.join(c.fk_sample_dir, s.vid)
                    os.makedirs(dd, exist_ok=True)
                    cv2.imwrite(os.path.join(dd, f"{s.vid}_f{s.fk_dump_n:03d}_p{'NA' if p is None else format(p, '.2f')}.jpg"), crop)
                except Exception:
                    pass
            bx, by = center(s.bbox)
            bsz = max(1.0, s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1])
            dist_truck = ((fcx - bx) ** 2 + (fcy - by) ** 2) ** 0.5 / bsz
            if p is not None:
                tr["obs"].append((now, float(p)))
                tr["dist"].append(dist_truck)
                tr.setdefault("stg", []).append(bool(self.zones.in_stage(fcx, fcy)) if self.zones.has_stage() else None)
                if c.fk_trans_dir:
                    tr.setdefault("snap", []).append(self._snap(frame, s.bbox, fk, p, dist_truck))
                if len(tr["obs"]) > 400:
                    tr["obs"] = tr["obs"][:5] + tr["obs"][5:-5][::2] + tr["obs"][-5:]
                    tr["dist"] = tr["dist"][:5] + tr["dist"][5:-5][::2] + tr["dist"][-5:]
                    tr["stg"] = tr["stg"][:5] + tr["stg"][5:-5][::2] + tr["stg"][-5:]
                    if tr.get("snap"):
                        tr["snap"] = tr["snap"][:5] + tr["snap"][5:-5][::2] + tr["snap"][-5:]
            if collect:
                jpg = to_jpg(crop, 320, 85)
                if jpg:
                    tr["crops"].append((now, dist_truck, jpg))
                    if len(tr["crops"]) > 120:
                        tr["crops"] = tr["crops"][:5] + tr["crops"][5:-5][::2] + tr["crops"][-5:]

    @staticmethod
    def _snap(frame, truck, fk, p, dist):
        """Foto penuh diperkecil (lebar 800) + kotak truk (merah) + kotak forklift (oranye) + skor & jarak. Hanya bila CAM1_FK_TRANS_DIR diisi."""
        try:
            k = 800.0 / max(1, frame.shape[1])
            im = cv2.resize(frame, (800, int(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
            cv2.rectangle(im, (int(truck[0] * k), int(truck[1] * k)), (int(truck[2] * k), int(truck[3] * k)), (0, 0, 255), 2)
            cv2.rectangle(im, (int(fk[0] * k), int(fk[1] * k)), (int(fk[2] * k), int(fk[3] * k)), (0, 140, 255), 2)
            cv2.putText(im, f"bawa={p:.2f} jarak={dist:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def _dump_transitions(self, s, max_n=20):
        """Simpan foto sesaat sebelum & sesudah tiap transisi bawa/kosong (nama: jenis + jarak ke truk) supaya terlihat DI MANA
        forklift menaruh/mengambil. Folder: CAM1_FK_TRANS_DIR/<id sesi>/"""
        c = self.cfg
        if not c.fk_trans_dir:
            return
        d = os.path.join(c.fk_trans_dir, s.vid)
        n = 0
        for ti, tr in enumerate(s.fk_tracks):
            snaps = tr.get("snap") or []
            runs = _fk_runs(tr["obs"], c)
            for r0, r1 in zip(runs, runs[1:]):
                if r0[0] == r1[0] or n >= max_n:
                    continue
                kind = "taruh" if r0[0] else "ambil"
                i0, i1 = r0[2], r1[1]
                dd = tr.get("dist") or []
                seg = dd[i0:i1 + 1] if len(dd) == len(tr["obs"]) else []
                dist = f"{sum(seg) / len(seg):.2f}" if seg else "NA"
                os.makedirs(d, exist_ok=True)
                n += 1
                for tag, idx in (("1sebelum", i0), ("2sesudah", i1)):
                    if 0 <= idx < len(snaps) and snaps[idx]:
                        with open(os.path.join(d, f"{n:02d}_{kind}_t{ti}_jarak{dist}_{tag}.jpg"), "wb") as f:
                            f.write(snaps[idx])
        if n:
            log.info("FK-TRANS %s: %d transisi difoto -> %s", s.vid, n, d)

    def _autolabel_forklift(self, s, res):
        """Label gratis: sesi dengan skor bak yang yakin menentukan arah -> forklift sebelum menyentuh kendaraan =
        bawa (Loading) / kosong (Unloading); sesudahnya kebalikan. Disimpan ke CAM1_FK_AUTOLABEL_DIR."""
        c = self.cfg
        if not c.fk_autolabel_dir or res.get("bed_dir") not in (LOADING, UNLOADING) or res.get("bed_conf", 0.0) < c.fk_autolabel_conf:
            return
        loading = res["bed_dir"] == LOADING
        lab_before, lab_after = ("forklift_bawa", "forklift_kosong") if loading else ("forklift_kosong", "forklift_bawa")
        saved = {"forklift_bawa": 0, "forklift_kosong": 0}
        for ti, tr in enumerate(s.fk_tracks):
            split = fk_trip_split(tr.get("crops") or [], c.fk_touch_margin, c.fk_autolabel_min_side)
            if split is None:
                continue
            for lab, items in ((lab_before, split[0]), (lab_after, split[1])):
                d = os.path.join(c.fk_autolabel_dir, lab)
                os.makedirs(d, exist_ok=True)
                for j, (_, _, jpg) in enumerate(items):
                    if saved[lab] >= c.fk_autolabel_max:
                        break
                    with open(os.path.join(d, f"{s.vid}_k{ti}_{j:03d}.jpg"), "wb") as f:
                        f.write(jpg)
                    saved[lab] += 1
        if any(saved.values()):
            log.info("[autolabel] %s (%s, keyakinan skor bak %.2f): bawa=%d kosong=%d -> %s", s.vid, res["bed_dir"],
                     res["bed_conf"], saved["forklift_bawa"], saved["forklift_kosong"], c.fk_autolabel_dir)

    def _tick_forklift(self, s, frame, now, fks):
        dt = min(max(0.0, now - s.prev_t), 3.0)
        s.prev_t = now
        near = expand_box(s.bbox, 0.15)
        if any(inter_area(near, fk) > 0 for fk in fks):
            s.fk_seconds += dt
            if now - s.last_fk_photo_t >= 3.0:
                s.last_fk_photo_t = now
                s.photo_fk = to_jpg(crop_vehicle(frame, s.bbox, 0.08), 640, 78) or s.photo_fk
            self._fk_observe(s, frame, now, fks)

    def _tick_stage(self, s, frame, now, dets):
        """[TINGKAT-A] Hitung detik di mana truk parkir DAN ada orang (titik kaki) di poligon 'stage' yang berada dekat truk.
        Pasif: tidak mengubah keputusan apa pun sampai sesi ditutup (lihat _close)."""
        c = self.cfg
        dt = min(max(0.0, now - s.stg_prev), 3.0)
        s.stg_prev = now
        if c.stage_act_min <= 0 or not s.parked or not self.zones.has_stage():
            return
        near = expand_box(s.bbox, c.stage_act_near)
        n = 0
        for d in dets:
            if d.cls != 0 or d.conf < c.stage_act_conf:
                continue
            fx, fy = (d.x1 + d.x2) / 2.0, d.y2               # titik kaki orang
            if self.zones.in_stage(fx, fy) and near[0] <= fx <= near[2] and near[1] <= fy <= near[3]:
                n += 1
        if not n:
            return
        s.stg_sec += dt
        s.stg_max = max(s.stg_max, n)
        if s.stg_first is None:
            s.stg_first = now
            log.info("STAGE-ORANG %s: truk parkir dan mulai ada orang di zona stage (%d orang)", s.vid, n)
        if now - s.stg_photo_t >= 3.0:
            s.stg_photo_t = now
            s.photo_stg = to_jpg(crop_vehicle(frame, s.bbox, 0.25), 640, 78) or s.photo_stg

    def _confirmed(self, s):
        return s.strong_hits >= self.cfg.confirm_hits

    def _update(self, s, cd, frame, now, dets, fks):
        c, d = self.cfg, cd["det"]
        s.t_last = now
        if cd["merged"]:
            # kotak gabungan forklift+kendaraan: sesi tetap hidup, tapi bbox/gerak/sampel/foto TIDAK dipercaya
            s.merged_n += 1
        else:
            s.bbox = tuple(d[:4])
            s.conf, s.cls = d.conf, d.cls
            s.max_conf = max(s.max_conf, d.conf)
            if d.conf >= c.confirm_conf:
                s.strong_hits += 1
            self._update_motion(s, now, s.bbox)
            self._depart_mark(s)
            _big = c.dock_min_ratio <= 0 or area(s.bbox) / max(1.0, self.zones.roi_area) >= c.dock_min_ratio
            if s.parked and not _big:
                s.far_n += 1
                if not s.far_logged:
                    s.far_logged = True
                    log.info("JAUH-DARI-DERMAGA %s | parkir dengan kotak %.1f%% luas ROI (< %.1f%%): waktu parkir ini tidak dihitung untuk MASUK",
                             s.vid, 100.0 * area(s.bbox) / max(1.0, self.zones.roi_area), 100.0 * c.dock_min_ratio)
            if s.parked and _big:
                if s.big_since is None:
                    s.big_since = now
            else:
                s.big_since = None
            if s.parked and s.parked_since is not None and _big:
                s.park_best = max(s.park_best, now - max(s.parked_since, s.big_since if s.big_since is not None else s.parked_since))
            if s.parked and s.park_tick is not None and _big:
                s.park_sum += max(0.0, min(now - s.park_tick, 5.0))
            s.park_tick = now
            s.n_upd += 1
            a_s = max(1.0, area(s.bbox))
            _fkx = list(fks)
            if c.fk_ovl_veto > 0:    # [FK-OVL-VETO] + forklift yang dibuang gerbang & forklift yang baru lewat
                _fkx += [x[1] for x in (getattr(self, "_fk_veto_mem", []) or []) if now - x[0] <= c.fk_veto_mem]
                _fkx += [b_ for _, b_ in (getattr(self, "_fk_mem", []) or [])]
            if any(inter_area(s.bbox, fk) / a_s >= 0.6 for fk in _fkx):
                s.fk_ovl_n += 1
            if s.sig is not None and now - s.sig_t >= 5.0:
                new = appearance_sig(frame, s.bbox)
                if new is not None:
                    s.sig = 0.85 * s.sig + 0.15 * new
                    s.sig /= max(s.sig.sum(), 1e-9)
                s.sig_t = now
            if c.presence > 0 and now - s.ref_t >= 5.0:
                _th = thumb_norm(frame, s.bbox)
                if _th is not None:
                    s.ref_thumb, s.ref_t = _th, now
            s.t_present, s.pres_n = None, 0
            s.pres_prev, s.pres_chain = None, False
            s.pres_miss_n = 0
            cx, cy = center(s.bbox)
            if s.parked and self.zones.in_dock(cx, cy):
                s.t_dock_in = s.t_dock_in or (s.parked_since or now)
                s.t_dock_out = now
            if now - s.last_photo_t >= c.photo_interval:
                s.last_photo_t = now
                _cr = crop_vehicle(frame, s.bbox, 0.08)
                _o = orange_pct(_cr)
                s.orange_n += 1
                if c.orange_big_pct > 0 and _o >= c.orange_big_pct:
                    s.orange_hi += 1          # crop didominasi forklift: jangan dipakai sebagai foto
                else:
                    s.photo_last = to_jpg(_cr) or s.photo_last
            self._maybe_sample(s, frame, now, s.bbox, dets, fks)
        if not s.announced and now - s.t_in >= c.entry_seconds and self._confirmed(s) and self._vehicle_ok(s):
            self._announce_entry(s)

    def _depart_mark(self, s):
        """[FIX-PERGI] tandai sesi terkonfirmasi yang kotaknya sudah mengecil (<= depart_ratio x luas parkir) DAN berada di luar kotak parkir
        (menjauh; kardus di dalam bak tidak dihitung). Ditandai ulang tiap pembaruan, jadi hilang sendiri bila kendaraan kembali besar/ke titik parkir."""
        c = self.cfg
        ref = s.last_parked_bbox
        if c.depart_ratio <= 0 or not s.announced or ref is None:
            s.departing = False
            return
        a = area(s.bbox)
        small = a <= c.depart_ratio * max(1.0, area(ref))
        outside = inter_area(s.bbox, ref) / max(1.0, a) < 0.5
        dep = bool(small and outside)
        if dep and not s.departing:
            log.info("PERGI-TERLIHAT %s | kotak mengecil ke %.0f%% luas parkir dan menjauh dari titik parkir -> ditutup bila hilang > %.0f dtk",
                     s.vid, 100.0 * a / max(1.0, area(ref)), c.depart_grace)
        s.departing = dep

    def _parked_enough(self, s):
        c = self.cfg
        if getattr(s, "edge", False) and c.edge_park > 0:
            return s.park_best >= c.edge_park    # sesi tepi: cukup parkir diam berturut-turut >= edge_park (10 dtk), terlepas dari park_confirm; jalur total tidak berlaku
        if c.park_confirm <= 0:
            return True
        if s.park_best >= c.park_confirm:
            return True
        return c.park_total > 0 and s.park_best >= c.park_min_run and s.park_sum >= c.park_total

    def _vehicle_ok(self, s):
        """Gerbang anti sesi-hantu: kendaraan sejati (a) pernah parkir diam >= park_confirm dtk dan (b) bukan sekadar kotak
        yang menempel pada forklift (forklift + palet terbaca 'truk')."""
        c = self.cfg
        why = None
        if not self._parked_enough(s):
            _need = c.edge_park if getattr(s, "edge", False) and c.edge_park > 0 else c.park_confirm
            why = (f"belum pernah parkir diam >= {_need:.0f} dtk berturut-turut{' (kendaraan tepi)' if getattr(s, 'edge', False) else f' atau total >= {c.park_total:.0f} dtk'} "
                   f"(terpanjang {s.park_best:.0f} dtk, total {s.park_sum:.0f} dtk)")
            if s.far_n:
                why += f"; {s.far_n} pembaruan parkir dengan kotak < {100.0 * c.dock_min_ratio:.1f}% luas ROI (masih jauh dari dermaga, tidak dihitung)"
        elif c.fk_ovl_max > 0 and s.n_upd >= c.fk_ovl_min_n and s.fk_ovl_n / s.n_upd > c.fk_ovl_max:
            why = f"kotaknya tertutup kotak forklift pada {s.fk_ovl_n}/{s.n_upd} pembaruan (bukan kendaraan)"
        elif c.sess_orange_frac > 0 and s.orange_n >= 5 and s.orange_hi / s.orange_n >= c.sess_orange_frac:
            why = f"kotaknya didominasi warna oranye forklift pada {s.orange_hi}/{s.orange_n} crop (bukan kendaraan)"
        elif (c.announce_min_samples > 0 and len(s.samples) < c.announce_min_samples and (s.t_last - s.t_in) < c.announce_wait
              and not (c.announce_strong > 0 and s.strong_hits >= c.announce_strong and c.park_confirm > 0 and self._parked_enough(s))):
            why = (f"bukti bak belum cukup ({len(s.samples)} < {c.announce_min_samples} sampel; bak tertutup orang/forklift terus-menerus), "
                   f"menunggu sampai {c.announce_wait:.0f} dtk")
        if why and not s.gate_logged:
            s.gate_logged = True
            log.info("GERBANG-KENDARAAN %s | MASUK ditahan: %s", s.vid, why)
        return why is None

    def _guard_fragment_of(self, bb, now):
        """[SESI-GANDA] sesi aktif terkonfirmasi yang kotaknya memuat kandidat kecil ini (pecahan truk), selain itu None."""
        c = self.cfg
        if c.open_guard <= 0:
            return None
        a = max(1.0, area(bb))
        cx, cy = center(bb)
        for old in self.sessions:
            if not old.announced or now - old.t_last > c.open_guard:
                continue
            for ref in (old.last_parked_bbox, old.bbox):
                if ref is None:
                    continue
                a_ref = max(1.0, area(ref))
                if a >= c.open_guard_ratio * a_ref:
                    continue
                rx, ry = center(ref)
                hw, hh = (ref[2] - ref[0]) * c.open_guard_expand / 2.0, (ref[3] - ref[1]) * c.open_guard_expand / 2.0
                if abs(cx - rx) <= hw and abs(cy - ry) <= hh:
                    return old
        return None

    def _open_session(self, p, now):
        c = self.cfg
        bb = p["items"][-1][1]
        _g = self._guard_fragment_of(bb, now)
        if _g is not None:
            self.guard_n = getattr(self, "guard_n", 0) + 1
            if self.guard_n in (1, 5, 20) or self.guard_n % 100 == 0:
                log.info("SESI-GANDA-DITOLAK kotak kecil (luas %.0f%% kotak %s, conf=%.2f, pusat (%d,%d)) dianggap pecahan kendaraan yang sama, TIDAK dibuka sesi baru / tidak diserahterimakan | total=%d",
                         100.0 * area(bb) / max(1.0, area(_g.last_parked_bbox or _g.bbox)), _g.vid, float(p["items"][-1][2]), *center(bb), self.guard_n)
            return
        # sesi lama yang masih "menggantung" di tempat yang sama = kendaraan lama sudah pergi
        for old in list(self.sessions):
            ref = old.last_parked_bbox or old.bbox
            if now - old.t_last > c.swap_gap and iou(bb, ref) >= 0.30:
                self._close(old, "kendaraan_lain_menggantikan", old.t_last)
        while len(self.sessions) >= c.max_sessions:
            # sesi yang belum dikonfirmasi (kemungkinan noise) dikorbankan lebih dulu, baru yang tertua
            oldest = min(self.sessions, key=lambda x: (x.announced, x.t_last))
            self._close(oldest, "batas_sesi_bersamaan", oldest.t_last)
        t0 = p["items"][0][0]
        log.info("BUKA-SESI-DIAG | %d item | kelas=%s | conf=%s | gabungan=%s | pusat_normal=(%.2f,%.2f) rasio_luas=%.3f",
                 len(p["items"]), sorted({int(it[3]) for it in p["items"]}), [round(float(it[2]), 2) for it in p["items"]],
                 [bool(it[4]) for it in p["items"]], center(bb)[0] / max(1, self.zones.w), center(bb)[1] / max(1, self.zones.h),
                 area(bb) / max(1.0, self.zones.roi_area))
        # SERAH-TERIMA: satu kendaraan yang pindah tempat (mis. dari spot dekat kamera ke dermaga jauh) hilang sebentar lalu muncul dengan
        # kotak yang tidak bertumpuk sama sekali, sehingga _match_score menolaknya. Bila sesi terkonfirmasi yang TIDAK terlihat pada frame ini
        # baru hilang <= handover_gap dan tampilannya mirip, lanjutkan sesi itu alih-alih membuka sesi baru.
        if c.handover_gap > 0 and all(not it[4] for it in p["items"]):   # hanya dari deteksi bersih (kotak gabungan forklift tidak boleh mengambil alih sesi)
            cands = []
            for old in self.sessions:
                gap = t0 - old.t_last
                if not old.announced or old.t_last >= now - 1.0 or gap > c.handover_gap or gap < -1.0:
                    continue
                if any(o is not old and o.t_last >= now - 1.0 and iou(o.bbox, bb) >= 0.3 for o in self.sessions):
                    continue
                if c.handover_size_ratio > 0:
                    _q = max(1.0, area(bb)) / max(1.0, area(old.last_parked_bbox or old.bbox))
                    if _q > c.handover_size_ratio or _q < 1.0 / c.handover_size_ratio:
                        continue                  # [FIX-UKURAN] ukuran kotak terlalu beda = bukan kendaraan yang sama
                _oc, _nc = center(old.bbox), center(bb)
                if math.hypot(_oc[0] - _nc[0], _oc[1] - _nc[1]) / max(1.0, gap) > c.handover_max_speed * max(1, self.zones.w):
                    continue
                dist = appearance_dist(old.sig, p["sig"]) if (old.sig is not None and p["sig"] is not None) else None
                if dist is not None and dist <= c.handover_dist:
                    cands.append((dist, old, gap))
            if cands:
                dist, old, gap = min(cands, key=lambda x: x[0])
                log.info("SERAH-TERIMA %s: kendaraan sama pindah posisi (hilang %.0f dtk, jarak tampilan %.2f <= %.2f, pusat (%d,%d) -> (%d,%d)) "
                         "-> sesi dilanjutkan, bukan sesi baru", old.vid, gap, dist, c.handover_dist, *center(old.bbox), *center(bb))
                old.bbox, old.t_last, old.prev_t = bb, now, now
                old.parked, old.parked_since, old.last_parked_bbox = False, None, None
                old.moving_hist.clear()
                old.handovers = getattr(old, "handovers", 0) + 1
                return
        _ph0 = p["photo"]
        try:
            if _ph0 and c.orange_big_pct > 0:
                _im0 = cv2.imdecode(np.frombuffer(_ph0, np.uint8), cv2.IMREAD_COLOR)
                if _im0 is not None and orange_pct(_im0) >= c.orange_big_pct:
                    _ph0 = None
        except Exception:
            pass
        s = Session(self._new_vid(t0), t0, bb, _ph0, p["sig"])
        s.max_conf = max(it[2] for it in p["items"])
        s.strong_hits = sum(1 for it in p["items"] if it[2] >= c.confirm_conf)
        s.cls = p["items"][-1][3]
        s.t_last = now
        s.prev_t = now
        s.edge = bool(p.get("edge", False))
        if s.edge:
            log.info("KENDARAAN-TEPI %s dibuka di pinggir area -> baru diumumkan bila parkir diam >= %.0f dtk berturut-turut", s.vid, c.edge_park)
        self.sessions.append(s)
        log.info("sesi baru %s (terlihat sejak %s)", s.vid, fmt_ts(t0))

    def _near_recent_forklift(self, bb, now):
        """[PATCH-1] True bila kotak kendaraan ini hampir seluruhnya berada di (atau sangat mirip dengan) kotak forklift yang terlihat
        dalam beberapa detik terakhir -> kemungkinan forklift yang lolos pada frame ini."""
        a = max(1.0, area(bb))
        for _, fb in self._fk_mem:
            if inter_area(bb, fb) / a >= 0.6 or iou(bb, fb) >= 0.4:
                return True
        return False

    def _feed_pending(self, frame, now, cands):
        c = self.cfg
        self.pending = [p for p in self.pending if now - p["t_last"] <= 3.0]
        for cd in cands:
            d = cd["det"]
            if d.conf < c.min_conf:
                self._rej("conf<min_conf(tak_bisa_buka_sesi)", d)
                continue  # deteksi lemah hanya boleh melanjutkan sesi, tidak boleh membuka
            bb = tuple(d[:4])
            if cd.get("outside"):
                continue                      # [HISTERESIS-ZONA] kotak di tepi luar area hanya boleh melanjutkan sesi, bukan membuka sesi baru
            if c.fk_mem_sec > 0 and not cd["merged"] and self._near_recent_forklift(bb, now):
                self._rej("forklift_ingatan(tak_bisa_buka_sesi)", d)
                continue
            # fragmen kendaraan yang sudah punya sesi terbuka (pintu/bak/kabin terbaca terpisah) bukan kendaraan baru
            if c.fragment_contain > 0 and any(
                    inter_area(bb, os_.bbox) / max(1.0, min(area(bb), area(os_.bbox))) >= c.fragment_contain
                    for os_ in self.sessions):
                self._rej("fragmen_dalam_sesi_terbuka", d)
                continue
            best = None
            for p in self.pending:
                sc = iou(bb, p["bbox"])
                if sc >= 0.15 and (best is None or sc > best[0]):
                    best = (sc, p)
            if best is None:
                p = dict(items=[], edge=True, bbox=bb, t_last=now, photo=to_jpg(crop_vehicle(frame, bb, 0.08)),
                         sig=None if cd["merged"] else appearance_sig(frame, bb))
                self.pending.append(p)
            else:
                p = best[1]
            p["items"].append((now, bb, d.conf, d.cls, cd["merged"]))
            if not cd.get("edge"):
                p["edge"] = False             # satu saja sighting di dalam area = bukan kandidat tepi
            p["items"] = [it for it in p["items"] if now - it[0] <= 8.0]
            p["bbox"], p["t_last"] = bb, now
            if p["sig"] is None and not cd["merged"]:
                p["sig"] = appearance_sig(frame, bb)
            n_clean = sum(1 for it in p["items"] if not it[4])
            if len(p["items"]) >= c.entry_hits and n_clean >= 1:
                self.pending.remove(p)
                self._open_session(p, now)

    def _fk_garpu_ok(self, box, conf):
        """[FK-GARPU] True bila deteksi forklift tidak-oranye ini boleh diterima (hanya garpu tampak, dekat truk parkir / di stage)."""
        c = self.cfg
        if not c.fk_garpu:
            return False
        cx, cy = center(box)
        a_box = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        in_stage = self.zones.has_stage() and self.zones.in_stage(cx, cy)
        for s in self.sessions:
            if not (s.announced and s.parked):
                continue
            sb = s.bbox
            if a_box > c.fk_garpu_area * max(1.0, (sb[2] - sb[0]) * (sb[3] - sb[1])):
                continue                                      # sebesar truk = kemungkinan truk boks yang salah baca
            ex = expand_box(sb, c.fk_garpu_pad)
            if in_stage or (ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3]):
                self.fk_garpu_n = getattr(self, "fk_garpu_n", 0) + 1
                if self.fk_garpu_n in (1, 10, 100) or self.fk_garpu_n % 500 == 0:
                    log.info("FK-GARPU diterima (tidak oranye, conf=%.2f) pusat=(%d,%d) dekat sesi %s total=%d", conf, cx, cy, s.vid, self.fk_garpu_n)
                return True
        return False

    # ---- langkah utama --------------------------------------------------------------------
    def step(self, frame, now, dets, fk_dets=None):
        """fk_dets: opsional, kotak forklift dari detektor model [(x1, y1, x2, y2, conf), ...] (frame penuh)."""
        h, w = frame.shape[:2]
        self.zones.bind(w, h)
        c = self.cfg
        fks = self._forklift_boxes(frame)               # untuk OBSERVASI (arah forklift): hanya yang di dalam area
        fks_extra = []                                  # forklift di luar area/zona exclude: hanya untuk MENOLAK kandidat kendaraan
        self._fk_conf = []
        if c.fk_veto > 0 or c.fk_ovl_veto > 0:
            self._fk_veto_mem = [x for x in self._fk_veto_mem if now - x[0] <= c.fk_veto_mem]
        for fd in fk_dets or []:
            # [FIX-FK-HANTU] buang "forklift" yang tidak oranye & tidak sangat yakin (truk boks putih terbaca forklift)
            if c.fk_orange_gate > 0:
                _cf = float(fd[4]) if len(fd) > 4 else 1.0
                if _cf < c.fk_orange_gate and orange_pct(crop_vehicle(frame, tuple(fd[:4]))) < c.orange_min_pct and not self._fk_garpu_ok(tuple(fd[:4]), _cf):
                    if (c.fk_veto > 0 or c.fk_ovl_veto > 0) and _cf >= c.fk_veto_conf:
                        self._fk_veto_mem.append((now, tuple(fd[:4]), _cf))
                    self.fk_ghost_n = getattr(self, "fk_ghost_n", 0) + 1
                    if self.fk_ghost_n in (1, 10, 100) or self.fk_ghost_n % 500 == 0:
                        log.info("FK-HANTU dibuang (tidak oranye, conf=%.2f < %.2f) pusat=(%d,%d) total=%d",
                                 _cf, c.fk_orange_gate, *center(tuple(fd[:4])), self.fk_ghost_n)
                    continue
            fb = expand_box(tuple(fd[:4]), c.fk_det_expand)
            cx, cy = center(fb)
            if len(fd) > 4:
                self._fk_conf.append((cx / max(1, w), cy / max(1, h), float(fd[4])))
            if self.zones.in_area(cx, cy):
                fks.append(fb)
            else:
                fks_extra.append(fb)
        self.last_fk = fks
        fks_reject = fks + fks_extra
        if c.fk_mem_sec > 0:
            self._fk_mem = [(t_, b_) for t_, b_ in self._fk_mem if now - t_ <= c.fk_mem_sec] + [(now, b_) for b_ in fks_reject]
        cands = self._candidates(frame, dets, fks_reject)
        if c.det_log:
            self._det_log(now, fks)

        for s in list(self.sessions):
            if now - s.t_in > c.hard_max:
                self._close(s, "batas_maksimum_sesi", s.t_last)

        pairs = []
        for si, s in enumerate(self.sessions):
            for ci, cd in enumerate(cands):
                sc = self._match_score(s, cd, now, frame)
                if sc is not None:
                    pairs.append((sc, si, ci))
        pairs.sort(key=lambda x: -x[0])
        used_s, used_c, matched = set(), set(), []
        for sc, si, ci in pairs:
            if si in used_s or ci in used_c:
                continue
            used_s.add(si)
            used_c.add(ci)
            matched.append((self.sessions[si], cands[ci]))

        for s, cd in matched:
            self._update(s, cd, frame, now, dets, fks)
        if c.relocate_hits > 0 and self.sessions:
            try:
                self._relocate(frame, now, cands, used_s, used_c)
            except Exception as _ex:
                log.debug("PINDAH-POSISI gagal: %s", _ex)
        for si, s in enumerate(list(self.sessions)):
            self._tick_forklift(s, frame, now, fks)
            try:
                self._tick_stage(s, frame, now, dets)
            except Exception as _ex:
                log.debug("STAGE-ORANG gagal: %s", _ex)
            self._tl_tick(s, now, dets, fks, si in used_s)
            if si in used_s:
                try:
                    self._collect_persons(s, frame, now, dets)
                except Exception as _ex:
                    log.debug("DATA-ORANG gagal: %s", _ex)
                if s.parked:
                    try:
                        self._person_observe(s, frame, now, dets)
                    except Exception as _ex:
                        log.warning("ARAH-ORANG gagal: %s", _ex)
            if si not in used_s and s in self.sessions:
                if c.depart_ratio > 0 and s.departing and now - s.t_last > c.depart_grace:
                    log.info("PERGI-TERLIHAT %s | hilang %.0f dtk setelah tampak pergi (> %.0f dtk) -> sesi ditutup sebagai KELUAR",
                             s.vid, now - s.t_last, c.depart_grace)
                    self._close(s, "terlihat_pergi", s.t_last)
                    continue
                grace = c.parked_grace if s.parked else c.moving_grace
                last = s.t_last
                if c.presence > 0 and s.announced and s.parked:
                    try:
                        self._presence_tick(s, frame, now)
                    except Exception as _ex:
                        log.debug("PRESENSI gagal: %s", _ex)
                    if s.t_present is not None and now - s.t_last <= c.presence_max_hold:
                        last = max(last, s.t_present)     # area parkir masih tampak sama: anggap kendaraan masih ada
                if now - last > grace:
                    self._close(s, "tidak_terlihat_lagi", last)

        self._feed_pending(frame, now, [cd for ci, cd in enumerate(cands) if ci not in used_c])

    def _relocate(self, frame, now, cands, used_s, used_c):
        """[PINDAH-POSISI] sesi terkonfirmasi yang tempat parkirnya SUDAH kosong (PRESENSI) + sighting lemah beruntun di tempat lain = pindah posisi."""
        c = self.cfg
        free = [(ci, cd) for ci, cd in enumerate(cands) if ci not in used_c and not cd.get("merged") and cd["det"].conf < c.min_conf]
        if not free:
            return
        for si, s in enumerate(self.sessions):
            ref = s.last_parked_bbox
            if si in used_s or not s.announced or ref is None or s.departing:
                continue
            if c.presence > 0 and s.pres_miss_n < c.relocate_presence_misses:
                continue                              # satu cek gagal bisa karena occlusion; spot lama harus terbukti kosong berulang
            gap = now - s.t_last
            if gap < 2.0 or gap > c.relocate_window:
                continue
            best = None
            for ci, cd in free:
                if ci in used_c:
                    continue
                bb = tuple(cd["det"][:4])
                if iou(bb, ref) >= 0.3 or any(o is not s and iou(o.bbox, bb) >= 0.3 for o in self.sessions):
                    continue
                if self._near_recent_forklift(bb, now):
                    continue
                q = max(1.0, area(bb)) / max(1.0, area(ref))
                if q > c.relocate_size_ratio or q < 1.0 / c.relocate_size_ratio:
                    continue
                oc, nc = center(ref), center(bb)
                if math.hypot(oc[0] - nc[0], oc[1] - nc[1]) / max(1.0, gap) > c.handover_max_speed * max(1, self.zones.w):
                    continue
                dist = appearance_dist(s.sig, appearance_sig(frame, bb)) if s.sig is not None else None
                if dist is None or dist > c.handover_dist:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, ci, bb)
            if best is None:
                continue
            dist, ci, bb = best
            hist = [h for h in s.reloc_hist if now - h[0] <= 8.0] + [(now, center(bb))]
            s.reloc_hist = hist
            if len(hist) < c.relocate_hits or any(
                    math.hypot(a[1][0] - b_[1][0], a[1][1] - b_[1][1]) > 0.10 * max(1, self.zones.w) for a, b_ in zip(hist, hist[1:])):
                continue
            log.info("PINDAH-POSISI %s: tempat parkir lama kosong, %d sighting lemah beruntun di tempat baru (hilang %.0f dtk, jarak tampilan %.2f <= %.2f, "
                     "pusat (%d,%d) -> (%d,%d)) -> sesi dilanjutkan", s.vid, len(hist), gap, dist, c.handover_dist, *center(ref), *center(bb))
            s.bbox, s.t_last, s.prev_t = bb, now, now
            s.parked, s.parked_since, s.last_parked_bbox = False, None, None
            s.pres_miss_n = 0
            s.moving_hist.clear()
            s.reloc_hist = []
            s.handovers = getattr(s, "handovers", 0) + 1
            used_s.add(si)
            used_c.add(ci)

    def _spot_covered(self, bb, now):
        """[PRESENSI-TUTUP] benar bila kotak forklift (frame ini atau beberapa dtk terakhir) menutup >= presence_occl_cover luas area parkir bb."""
        c = self.cfg
        A = max(1.0, area(bb))
        boxes = list(getattr(self, "last_fk", []) or []) + [b_ for _, b_ in (getattr(self, "_fk_mem", []) or [])]
        for fb in boxes:
            iw = min(bb[2], fb[2]) - max(bb[0], fb[0])
            ih = min(bb[3], fb[3]) - max(bb[1], fb[1])
            if iw > 0 and ih > 0:
                frac = (iw * ih) / A
                if frac >= c.presence_occl_cover_max:
                    return False                  # area parkir hampir seluruhnya = forklift: sesi ini kemungkinan forklift itu sendiri
                if frac >= c.presence_occl_cover:
                    return True
        return False

    def _presence_tick(self, s, frame, now):
        """[PRESENSI] sesi parkir yang tak terdeteksi pada frame ini: apakah area parkirnya masih tampak seperti kendaraan tadi?"""
        c = self.cfg
        if s.ref_thumb is None or now - s.pres_t < c.presence_interval:
            return
        s.pres_t = now
        bb = s.last_parked_bbox or s.bbox
        cur = thumb_norm(frame, bb)
        ncc = float(np.dot(s.ref_thumb, cur)) if cur is not None else -1.0
        dist = appearance_dist(s.sig, appearance_sig(frame, bb)) if s.sig is not None else 0.0
        ok_ref = ncc >= c.presence_ncc and dist <= c.presence_dist
        ncc_prev = float(np.dot(s.pres_prev, cur)) if (s.pres_prev is not None and cur is not None) else -1.0
        s.pres_prev = cur
        if ok_ref:
            s.pres_chain = True                       # rantai dimulai/diperbarui dari kecocokan dengan acuan
        elif not (s.pres_chain and c.presence_chain > 0 and ncc_prev >= c.presence_chain):
            s.pres_chain = False                      # berubah mendadak (mis. truk pergi) -> rantai putus
        ok = ok_ref or s.pres_chain
        occl_hold = False
        if c.presence_occl > 0:
            if ok:
                s.occl_t0 = None
            elif self._spot_covered(bb, now):
                if getattr(s, "occl_t0", None) is None:
                    s.occl_t0 = now
                if now - s.occl_t0 <= c.presence_occl:
                    ok, occl_hold = True, True
            else:
                s.occl_t0 = None
        if ok:
            s.t_present = now
            s.pres_miss_n = 0
        else:
            s.pres_miss_n += 1
        if s.pres_n == 0 or ok != s.pres_ok or now - s.pres_log_t >= 60.0:
            s.pres_log_t = now
            if occl_hold:
                _msg = "TERTUTUP forklift, bukan bukti pergi (sesi ditahan %.0f/%.0f dtk)" % (now - s.occl_t0, c.presence_occl)
            else:
                _msg = ("MASIH ADA (sesi ditahan)" if ok_ref else "MASIH ADA (berubah pelan, rantai)") if ok else "tidak sama (hitung mundur keluar)"
            log.info("PRESENSI %s | tak terdeteksi %.0f dtk | korelasi=%.2f (>=%.2f) antar-cek=%.2f (>=%.2f) warna=%.2f (<=%.2f) -> %s", s.vid, now - s.t_last,
                     ncc, c.presence_ncc, ncc_prev, c.presence_chain, dist, c.presence_dist, _msg)
        s.pres_n += 1
        s.pres_ok = ok

    def flush(self, now=None):
        """Dipanggil saat program berhenti / video habis: tutup semua sesi yang masih terbuka."""
        log.info("KANDIDAT-DITOLAK (car/truck yang tidak jadi sesi) | %s", self.rej_summary())
        for s in list(self.sessions):
            if not s.announced and s.t_last - s.t_in >= self.cfg.entry_seconds and self._confirmed(s) and self._vehicle_ok(s):
                self._announce_entry(s)
            self._close(s, "program_berhenti", s.t_last)


# ----------------------------------------------------------------------------------------------
# Detektor, scorer, sumber video
# ----------------------------------------------------------------------------------------------
class YoloDetector:
    """YOLO11n (OpenVINO) hanya pada ROI zona 'area' -> hemat CPU. Kelas: person, car, truck."""

    def __init__(self, cfg, zones):
        from ultralytics import YOLO
        self.cfg, self.zones = cfg, zones
        self.model = YOLO(cfg.model_path, task="detect")

    def __call__(self, frame):
        self.zones.bind(frame.shape[1], frame.shape[0])
        x1, y1, x2, y2 = self.zones.roi
        if self.cfg.roi_ext_top > 0:
            y1 = max(0, y1 - int(self.cfg.roi_ext_top * frame.shape[0]))   # [ROI-ATAS] hanya untuk YOLO kendaraan; luas rujukan rasio tetap
        if self.cfg.roi_ext_right > 0:
            x2 = min(frame.shape[1], x2 + int(self.cfg.roi_ext_right * frame.shape[1]))
        res = self.model.predict(frame[y1:y2, x1:x2], conf=self.cfg.det_conf, iou=self.cfg.iou,
                                 classes=[0, 2, 7], verbose=False)[0]
        out = []
        for b in res.boxes:
            cls_ = int(b.cls[0]); cf_ = float(b.conf[0])
            if cls_ == 0 and cf_ < self.cfg.person_conf:
                continue
            bx = b.xyxy[0].tolist()
            out.append(Det(bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1, float(b.conf[0]), int(b.cls[0])))
        return out


class ForkliftDetector:
    """Detektor forklift (model OpenVINO, kelas bernama 'forklift') pada ROI zona 'area'.
    Mengembalikan kotak (x1, y1, x2, y2, conf) koordinat frame penuh. TIDAK PERNAH melempar error ke loop utama:
    bila model gagal dimuat / gagal jalan, cam1 tetap jalan dengan blob warna saja.

    Backend (CAM1_FK_BACKEND):
      direct      (default) keluaran mentah OpenVINO dibaca & di-decode sendiri (fk_openvino.py). Tidak bergantung pada
                  postprocess ultralytics, yang bisa salah menebak format untuk model 2 kelas (4 kotak + 2 kelas = 6 kolom).
      ultralytics postprocess bawaan ultralytics (YOLO(folder).predict)."""

    def __init__(self, cfg, zones):
        self.cfg, self.zones, self.model, self.direct, self.ids = cfg, zones, None, None, []
        self.fail_logged = False
        backend = cfg.fk_backend
        try:
            if backend == "ultralytics":
                from ultralytics import YOLO
                self.model = YOLO(cfg.fk_model_path, task="detect")
                names = dict(getattr(self.model, "names", None) or {})
                self.ids = [int(k) for k, v in names.items() if "forklift" in str(v).lower()]
                if not self.ids:
                    raise ValueError(f"tidak ada kelas 'forklift' di model (kelas: {names})")
            else:
                from fk_openvino import DirectForklift
                self.direct = DirectForklift(cfg.fk_model_path, conf=cfg.fk_det_conf, iou=cfg.iou)
                names, self.ids = self.direct.names, self.direct.class_ids
            log.info("detektor forklift model dimuat: %s | backend=%s | kelas=%s | conf>=%.2f", cfg.fk_model_path,
                     backend, {i: names[i] for i in self.ids}, cfg.fk_det_conf)
        except Exception as ex:
            log.error("CAM1_FK_MODEL gagal dimuat (%s) -> detektor forklift DIMATIKAN, pakai blob warna saja", ex)
            self.model = self.direct = None

    @property
    def ok(self):
        return self.model is not None or self.direct is not None

    def __call__(self, frame):
        if not self.ok:
            return []
        try:
            self.zones.bind(frame.shape[1], frame.shape[0])
            x1, y1, x2, y2 = self.zones.roi
            roi = frame[y1:y2, x1:x2]
            out = []
            if self.direct is not None:
                for bx in self.direct(roi):
                    out.append((bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1, float(bx[4])))
                return out
            res = self.model.predict(roi, conf=self.cfg.fk_det_conf, iou=self.cfg.iou, classes=self.ids, verbose=False)[0]
            for b in res.boxes:
                bx = b.xyxy[0].tolist()
                out.append((bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1, float(b.conf[0])))
            return out
        except Exception as ex:
            if not self.fail_logged:
                self.fail_logged = True
                log.error("detektor forklift error (%s); frame ini dilewati", ex)
            return []


def make_scorer():
    try:
        import cargo_classifier
        if not cargo_classifier.is_available():
            log.warning("cargo_model.pkl tidak ditemukan -> status muatan TIDAK bisa ditentukan")
            return lambda crop: None
        try:
            import pickle
            with open(cargo_classifier.CARGO_MODEL_PATH, "rb") as f:
                classes = [str(x) for x in pickle.load(f)["model"].classes_]
            if not any(x in classes for x in ("tertutup", "forklift", "bukan_truk")):
                log.warning("cargo_model.pkl hanya punya kelas %s. Tanpa kelas 'tertutup'/'forklift' akurasi "
                            "Loading/Unloading turun. Latih ulang: python train_classifier.py", classes)
        except Exception:
            pass
        return cargo_classifier.loaded_probability
    except Exception as e:
        log.warning("cargo_classifier tidak tersedia (%s)", e)
        return lambda crop: None


def make_fk_scorer():
    """Skor 'forklift bawa barang' (forklift_carry_model.pkl). None bila model belum ada -> arah dari forklift mati."""
    try:
        import forklift_carry
        if not forklift_carry.is_available():
            log.warning("forklift_carry_model.pkl tidak ditemukan -> arah dari gerakan forklift MATI "
                        "(latih: python train_fk_carry.py)")
            return None
        return forklift_carry.carry_probability
    except Exception as e:
        log.warning("forklift_carry tidak tersedia (%s)", e)
        return None


class Grabber(threading.Thread):
    """Baca RTSP terus-menerus di thread sendiri, simpan hanya frame TERBARU, sambung ulang otomatis."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.frame, self.ts, self.seq = url, None, 0.0, 0
        self._lock = threading.Lock()
        self.alive = True

    def run(self):
        while self.alive:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                log.warning("RTSP gagal dibuka, coba lagi 5 dtk")
                time.sleep(5)
                continue
            log.info("RTSP tersambung")
            fails = 0
            while self.alive:
                ok, fr = cap.read()
                if not ok or fr is None:
                    fails += 1
                    if fails > 30:
                        break
                    time.sleep(0.1)
                    continue
                fails = 0
                with self._lock:
                    self.frame, self.ts, self.seq = fr, time.time(), self.seq + 1
            cap.release()
            log.warning("RTSP terputus, menyambung ulang...")
            time.sleep(3)

    def latest(self):
        with self._lock:
            return self.frame, self.ts, self.seq


def overlay(frame, zones, mon, dets, now):
    img = zones.draw(frame.copy())
    for d in dets:
        col = (255, 255, 0) if d.cls == 0 else (0, 255, 0)
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 2)
    for fk in getattr(mon, "last_fk", []):
        cv2.rectangle(img, (int(fk[0]), int(fk[1])), (int(fk[2]), int(fk[3])), (0, 140, 255), 3)
        cv2.putText(img, "FORKLIFT", (int(fk[0]), max(30, int(fk[1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 3)
    lines = []
    for s in mon.sessions:
        st = s.info["status"] if s.info else "-"
        fkd = fk_direction(Cam1Monitor._fk_obs(s), mon.cfg, Cam1Monitor._fk_dist(s), Cam1Monitor._fk_stage(s))
        lines.append(f"{s.vid[-4:]} {'PARKIR' if s.parked else 'bergerak'} sampel={len(s.samples)} "
                     f"fk={s.fk_seconds:.0f}s taruh/ambil={fkd['drops']}/{fkd['picks']} {st}")
        cv2.rectangle(img, (int(s.bbox[0]), int(s.bbox[1])), (int(s.bbox[2]), int(s.bbox[3])), (0, 0, 255), 4)
    if not lines:
        lines = ["tidak ada sesi"]
    for i, ln in enumerate(lines):
        y = 60 + i * 70
        txt = (f"{fmt_ts(now)}  " if i == 0 else "") + ln
        cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
        cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return cv2.resize(img, (1280, int(1280 * img.shape[0] / img.shape[1])))


# ----------------------------------------------------------------------------------------------
# Mode jalan
# ----------------------------------------------------------------------------------------------
def run_live(cfg, zones, args):
    sender = N8NSender(cfg, enabled=not args.no_send)
    mon = Cam1Monitor(cfg, zones, make_scorer(), sender, make_fk_scorer(), make_pc_scorer())
    det = YoloDetector(cfg, zones)
    fkdet = ForkliftDetector(cfg, zones) if cfg.fk_model_path else None
    grab = Grabber(args.rtsp or build_rtsp_url())
    grab.start()
    period, last_seq, last_log = 1.0 / cfg.fps, -1, time.time()
    log.info("cam1 live berjalan (%.1f fps, n8n=%s)", cfg.fps, "AKTIF" if sender.enabled else "mati")
    try:
        while True:
            t0 = time.time()
            frame, ts, seq = grab.latest()
            if frame is not None and seq != last_seq and t0 - ts < 10:
                last_seq = seq
                try:
                    dets = det(frame)
                    mon.step(frame, ts, dets, fkdet(frame) if fkdet else None)
                    if args.show:
                        cv2.imshow("cam1", overlay(frame, zones, mon, dets, ts))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                except Exception:
                    log.error("error di langkah deteksi:\n%s", traceback.format_exc())
            elif frame is not None and t0 - ts >= 10 and t0 - last_log > 30:
                last_log = t0
                log.warning("tidak ada frame baru %.0f dtk (stream bermasalah?)", t0 - ts)
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        log.info("dihentikan (Ctrl+C)")
    finally:
        grab.alive = False
        mon.flush()
        sender.close()


def run_video(cfg, zones, args):
    sender = N8NSender(cfg, enabled=not args.no_send)
    mon = Cam1Monitor(cfg, zones, make_scorer(), sender, make_fk_scorer(), make_pc_scorer())
    det = YoloDetector(cfg, zones)
    fkdet = ForkliftDetector(cfg, zones) if cfg.fk_model_path else None
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log.error("tidak bisa membuka video %s", args.video)
        return
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step_n = max(1, int(round(vfps / cfg.fps)))
    base = datetime.strptime(args.video_start, "%Y-%m-%d %H:%M:%S").timestamp() if args.video_start else time.time()
    writer, idx, t_wall = None, 0, time.time()
    log.info("video %s | %.1f fps | proses tiap %d frame | mulai %s", args.video, vfps, step_n, fmt_ts(base))
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step_n == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                now = base + idx / vfps
                dets = det(frame)
                mon.step(frame, now, dets, fkdet(frame) if fkdet else None)
                if args.debug_video:
                    vis = overlay(frame, zones, mon, dets, now)
                    if writer is None:
                        writer = cv2.VideoWriter(args.debug_video, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if total and (idx // step_n) % 200 == 0:
                    log.info("progres %.1f%% | MASUK=%d SELESAI=%d", 100.0 * idx / total, mon.stats["masuk"], mon.stats["selesai"])
            idx += 1
        mon.flush()
    finally:
        cap.release()
        if writer:
            writer.release()
        sender.close()
    log.info("selesai: MASUK=%d SELESAI=%d dibuang=%d | %.0f dtk", mon.stats["masuk"], mon.stats["selesai"],
             mon.stats["dibuang"], time.time() - t_wall)


# ---- editor zona ----------------------------------------------------------------------------
class ZoneModel:
    NAMES = {"1": "area", "2": "exclude", "3": "dock", "4": "stage"}

    def __init__(self, data):
        self.z = {"area": list(data.get("area") or []), "exclude": [list(p) for p in data.get("exclude") or []],
                  "dock": list(data.get("dock") or []), "stage": list(data.get("stage") or [])}
        self.name, self.cur = "area", []

    def select(self, key):
        if key in self.NAMES:
            self.name, self.cur = self.NAMES[key], []

    def click(self, x, y, w, h):
        self.cur.append([round(x / w, 4), round(y / h, 4)])

    def undo(self):
        if self.cur:
            self.cur.pop()

    def finish(self):
        if len(self.cur) < 3:
            return False
        if self.name == "exclude":
            self.z["exclude"].append(self.cur)
        else:
            self.z[self.name] = self.cur
        self.cur = []
        return True

    def delete_last_exclude(self):
        if self.z["exclude"]:
            self.z["exclude"].pop()

    def as_dict(self):
        return dict(self.z)


def grab_one_frame(args, cfg):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"gagal membaca {args.image}")
        return img
    cap = cv2.VideoCapture(args.rtsp or build_rtsp_url(), cv2.CAP_FFMPEG)
    for _ in range(10):
        ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("gagal mengambil frame dari RTSP; pakai --image frame.jpg")
    return fr


def run_preview(cfg, args):
    img = grab_one_frame(args, cfg)
    zones = Zones.load(cfg.zones_path)
    zones.bind(img.shape[1], img.shape[0])
    cv2.imwrite("zones_preview.jpg", zones.draw(img))
    print("ditulis: zones_preview.jpg (hijau=area deteksi, merah=dikecualikan, oranye=dock, biru=tumpukan palet)")


def run_draw(cfg, args):
    img = grab_one_frame(args, cfg)
    h, w = img.shape[:2]
    scale = 1400.0 / w
    path = cfg.zones_path
    data = {} if getattr(args, "kosong", False) else (json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {})
    model = ZoneModel(data)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            model.click(x / scale, y / scale, w, h)

    cv2.namedWindow("zona", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("zona", on_mouse)
    print("1=area  2=exclude  3=dock  4=stage(tumpukan palet) | klik=titik | Enter=selesai poligon | u=undo | d=hapus exclude terakhir | s=simpan | q=keluar")
    while True:
        z = Zones(model.as_dict())
        z.bind(w, h)
        vis = z.draw(img.copy())
        if model.cur:
            pts = np.array([[x * w, y * h] for x, y in model.cur], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (255, 0, 255), 3)
            for p in pts:
                cv2.circle(vis, tuple(int(v) for v in p), 8, (255, 0, 255), -1)
        cv2.putText(vis, f"mode: {model.name}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.imshow("zona", cv2.resize(vis, None, fx=scale, fy=scale))
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        elif chr(k) in ZoneModel.NAMES if k < 128 else False:
            model.select(chr(k))
        elif k in (13, ord("n")):
            model.finish()
        elif k == ord("u"):
            model.undo()
        elif k == ord("d"):
            model.delete_last_exclude()
        elif k == ord("s"):
            out = model.as_dict()
            if os.path.isfile(path) and not os.path.isfile(path + ".bak"):
                import shutil
                shutil.copyfile(path, path + ".bak")  # simpan zona lama sekali sebagai cadangan
            out["_catatan"] = "Koordinat normal 0..1 terhadap frame penuh. Edit dengan: python cam1.py --draw"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"tersimpan: {path}")
    cv2.destroyAllWindows()


def setup_logging(debug=False, log_dir=None, log_name=None):
    """Folder log: --log-dir  >  CAM1_LOG_DIR di .env  >  'logs' (di samping cam1.py).
    Nama file: --log-name  >  CAM1_LOG_NAME  >  cam1.log. Path relatif dihitung dari folder tempat perintah dijalankan
    (bukan dari folder cam1.py), jadi 'tes/logs' bekerja bila dijalankan dari folder induk tes."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger("cam1")
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    d = log_dir or os.environ.get("CAM1_LOG_DIR", "").strip().strip('"') or "logs"
    name = log_name or os.environ.get("CAM1_LOG_NAME", "").strip().strip('"') or "cam1.log"
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.abspath(os.path.join(d, name))
        fh = RotatingFileHandler(path, maxBytes=int(_env("LOG_MAX_MB", 100.0) * 1024 * 1024),
                                 backupCount=int(_env("LOG_KEEP_FILES", 14.0)), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("log ditulis ke: %s", path)
    except Exception as ex:
        root.warning("gagal membuat file log di '%s' (%s) -> log hanya ke layar", d, ex)


def run_forever():
    """Dipanggil dari aw.py (thread cam1). Tidak pernah kembali; crash -> restart otomatis 5 dtk."""
    setup_logging()
    args = argparse.Namespace(no_send=False, rtsp=None, show=False)
    while True:
        try:
            # dimuat DI DALAM loop: bila cam1_zones.json hilang/rusak, thread tidak mati; dicoba lagi tiap 5 dtk
            cfg = Cfg()
            zones = Zones.load(cfg.zones_path)
            run_live(cfg, zones, args)
            log.warning("run_live berhenti, restart dalam 5 detik")
        except Exception:
            log.error("CRASH cam1, restart dalam 5 detik:\n%s", traceback.format_exc())
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser(description="CAM1 loading dock monitor")
    ap.add_argument("--video", help="uji offline dengan file video")
    ap.add_argument("--video-start", help='jam awal video, "YYYY-MM-DD HH:MM:SS" (default: sekarang)')
    ap.add_argument("--rtsp", help="override URL RTSP")
    ap.add_argument("--no-send", action="store_true", help="jangan kirim ke n8n (tetap tulis events_cam1.jsonl)")
    ap.add_argument("--debug-video", help="tulis video beranotasi (mp4) saat --video")
    ap.add_argument("--show", action="store_true", help="tampilkan jendela live")
    ap.add_argument("--draw", action="store_true", help="editor zona")
    ap.add_argument("--kosong", action="store_true", help="--draw mulai dari kosong (tanpa zona lama)")
    ap.add_argument("--preview", action="store_true", help="tulis zones_preview.jpg")
    ap.add_argument("--image", help="gambar frame untuk --draw/--preview")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log-dir", help="folder log (mis. tes/logs); default CAM1_LOG_DIR atau logs")
    ap.add_argument("--log-name", help="nama file log (default cam1.log)")
    ap.add_argument("--events-log", help="file events jsonl (default events_cam1.jsonl)")
    args = ap.parse_args()

    # path yang diberi pengguna lewat argumen dihitung dari folder kerja SAAT perintah dijalankan; ubah jadi absolut sebelum chdir
    args.log_dir = os.path.abspath(args.log_dir) if args.log_dir else None
    if args.events_log:
        os.environ["CAM1_EVENTS_LOG"] = os.path.abspath(args.events_log)
    if args.video:
        args.video = os.path.abspath(args.video)
    if args.debug_video:
        args.debug_video = os.path.abspath(args.debug_video)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    setup_logging(args.debug, args.log_dir, args.log_name)
    cfg = Cfg()
    if args.draw:
        return run_draw(cfg, args)
    if args.preview:
        return run_preview(cfg, args)
    zones = Zones.load(cfg.zones_path)
    if args.video:
        run_video(cfg, zones, args)
    else:
        run_live(cfg, zones, args)


if __name__ == "__main__":
    main()
