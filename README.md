# Vehicle Traffic & Cargo Dock Monitoring System

Sistem pemantauan lalu lintas kendaraan dan aktivitas bongkar-muat (*loading / unloading*) berbasis Computer Vision dual-camera untuk area industri, pergudangan, dan loading dock. Sistem ini dioptimalkan menggunakan model YOLOv11 dan MobileNetV2 dengan inferensi terakselerasi via Intel OpenVINO.

---

## 📌 Fitur Utama

### 1. Dual-Camera Architecture
* **CAM 1 (Loading Dock / Dermaga Muatan):**
  * Dijalankan oleh `cam1.py`.
  * Pelacakan berbasis keberadaan sesi kendaraan (*presence session tracking*) di area dock tanpa ketergantungan pada track ID yang rapuh.
  * Estimasi status muatan (*Loaded / Empty / Closed / Forklift*) menggunakan crop bak kendaraan dan embedding MobileNetV2.
  * Deteksi arah bongkar-muat (*Loading vs Unloading*) berbasis fusi bukti log-odds:
    * Analisis tren skor muatan bak (tiba vs berangkat).
    * Gerakan & status muatan forklift (*carrying cargo vs empty forks*).
    * Status aktivitas orang yang membawa barang.
  * Pencegahan *false positive* terhadap forklift oranye (Heli) dan kendaraan tumpang-tindih.
* **CAM 2 (Halaman Depan / Gate Kendaraan):**
  * Dijalankan oleh pipeline `aw.py`.
  * Pelacakan kendaraan berbasis tracker OC-SORT (`ocsort_custom.yaml`) dengan fitur relink kendaraan dan penanganan *stationary lost*.
  * Estimasi aktivitas kendaraan (`activity_estimator.py`) saat berada di area transit.

### 2. Integrasi & Output
* **n8n Webhook Integration:** Mengirim payload event kendaraan (waktu masuk, waktu keluar, durasi, status muatan, estimasi aktivitas, dan link/bukti visual) secara otomatis ke workflow n8n.
* **Robust Outbox Queue:** Mekanisme antrean lokal (`n8n_outbox.jsonl`) dengan *automatic retry* dan pemindahan *dead letter* saat jaringan terganggu.
* **Rotasi Log Otomatis:** Penyimpanan log stdout/stderr dengan batas ukuran file dan retensi otomatis di folder `logs/`.

---

## 📁 Struktur Repositori

```text
├── aw.py                              # Komando utama (Dual-Cam Coordinator & CAM 2 pipeline)
├── cam1.py                            # Pipeline monitoring CAM 1 (Loading Dock)
├── activity_estimator.py              # Logika estimasi aktivitas (Loading / Unloading / Transit)
├── cargo_classifier.py                # Wrapper klasifikasi muatan bak kendaraan
├── dock_events.py                     # Manajemen event transisi muatan per sesi
├── feature_extractor.py               # Ekstraksi fitur visual embedding MobileNetV2 (OpenVINO)
├── fk_openvino.py                     # Direct OpenVINO detector untuk forklift
├── forklift_carry.py                  # Classifier status bawaan forklift (bawa barang vs kosong)
├── forklift_filter.py                 # Filter penolak objek forklift di CAM 2
├── person_carry.py                    # Classifier status orang (bawa barang vs tangan kosong)
├── cam1_zones.json                    # Konfigurasi koordinat poligon zona CAM 1
├── ocsort_custom.yaml                 # Konfigurasi parameter tracker OC-SORT
├── requirement.txt                    # Daftar dependensi paket Python
├── .env.example                       # Template konfigurasi environment (RTSP, n8n, ambang batas)
├── training_tools/                    # Koleksi script pelatihan model, anotasi, dan tool ROI
│   ├── roi.py                         # Tool interaktif penentuan zona poligon via RTSP
│   ├── train_classifier.py            # Script pelatihan classifier muatan
│   ├── train_fk_carry.py              # Script pelatihan classifier status forklift
│   ├── train_carry.py                 # Script pelatihan classifier status orang
│   ├── label_crops.py                 # Tool GUI pelabelan gambar crop
│   └── README.md                      # Panduan penggunaan tool training
└── README.md                          # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Prasyarat Sistem
* Python 3.10 atau 3.11 (disarankan)
* CPU dengan instruksi AVX2/AVX-512 atau Intel iGPU/NPU (untuk akselerasi OpenVINO)
* Akses stream RTSP kamera IP / NVR

### 2. Instalasi Dependensi
Buat virtual environment dan pasang pustaka yang diperlukan:

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Instal dependensi
pip install -r requirement.txt
```

### 3. Konfigurasi Lingkungan (`.env`)
Salin file template `.env.example` menjadi `.env`:

```bash
cp .env.example .env
```

Buka file `.env` dan sesuaikan parameter berikut:
* `CAMERA_USER`, `CAMERA_PASS`, `CAMERA_HOST`, `CAM1_CHANNEL`, `CAM2_CHANNEL`: Kredensial dan endpoint RTSP kamera/NVR Anda.
* `N8N_WEBHOOK_URL`: URL endpoint webhook n8n untuk penerimaan data log kendaraan.
* `MODEL_PATH`: Lokasi folder model deteksi YOLO (OpenVINO).
* `CARGO_MODEL_PATH`: Lokasi file model klasifikasi muatan (`cargo_model.pkl`).

> ⚠️ **PERHATIAN KEAMANAN:** File `.env` berisi data sensitif (IP kamera, user, password, webhook secret). **JANGAN PERNAH** menambahkan atau mem-push file `.env` asli ke repositori publik. File ini sudah otomatis dimasukkan ke dalam `.gitignore`.

---

## 💻 Penggunaan

### 1. Menjalankan Sistem Penuh (Dual-Camera)
Untuk menjalankan pemantauan kedua kamera secara bersamaan:

```bash
python aw.py
```

`aw.py` akan memulai thread independen untuk CAM 1 (`cam1.run_forever()`) dan CAM 2 (`camera_loop`).

### 2. Menjalankan CAM 1 Secara Mandiri
Jika ingin menguji atau menjalankan CAM 1 saja:

```bash
# Mode live RTSP (berdasarkan konfigurasi di .env)
python cam1.py

# Mode uji coba dengan rekaman video offline (tanpa kirim webhook ke n8n)
python cam1.py --video path/ke/rekaman.mp4 --no-send

# Mode preview zona CAM 1
python cam1.py --preview --image frame.jpg
```

### 3. Kalibrasi Zona Poligon (ROI)
Gunakan tool interaktif di folder `training_tools/` untuk menggambar poligon area pantau:

```bash
python training_tools/roi.py rtsp://user:password@<ip-kamera>:554/stream
```

---

## 🔒 Privasi & Keamanan Data
Repositori ini dikonfigurasi dengan aturan pengabaian file ketat (`.gitignore`):
* Kredensial otentikasi RTSP, alamat IP, dan token n8n diisolasi dalam file `.env`.
* Bobot model berukuran besar (`*.pkl`, `*.pt`, `*_openvino_model/`) dan rekaman video/dataset tidak diikutsertakan ke dalam version control.
* Log harian dan bukti foto tangkapan disimpan secara lokal pada folder `logs/` dan `evidence_cam1/`.
