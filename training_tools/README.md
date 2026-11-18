# Training & Helper Tools

Folder ini berisi script dan file pendukung yang digunakan untuk:
1. Pelatihan model (training classifier muatan, forklift carry, person carry, dan YOLO forklift).
2. Anotasi data dan ekstraksi crop (`label_crops.py`, `extract_crops.py`, `terapkan_label.py`).
3. Tool kalibrasi interaktif (`roi.py`).
4. Dataset config (`data_forklift.yaml`) dan bobot awal (`yolo11n.pt`).

Script-script di folder ini **tidak diimpor atau dibutuhkan** saat `aw.py` berjalan.
Jika ingin menjalankan salah satu script dari folder ini, jalankan dari root project:
misalnya:
    python training_tools/roi.py <rtsp_url>
