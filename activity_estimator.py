# activity_estimator.py - penentu Loading/Unloading dari selisih skor P(bermuatan) saat tiba vs saat pergi.
# Skor naik = Loading, turun = Unloading, perubahan kecil = Transit / Lewat Kosong.
# Boks tertutup, sampel kurang, atau skor di batas = Tidak Dapat Dipastikan.
# Forklift di dekat truk hanya jadi keterangan di field reason, bukan penentu arah.

import os
import threading
from collections import deque

LOADING = "Mengambil Muatan (Loading)"
UNLOADING = "Bongkar Muatan (Unloading)"
TRANSIT = "Transit / membawa Muatan"
EMPTY_PASS = "Lewat Kosong"
UNKNOWN = "Tidak Dapat Dipastikan"
FORKLIFT_ACTIVITY = "Aktivitas Forklift (arah belum pasti)"


def _f(key, default):
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _i(key, default):
    try:
        return int(float(os.environ.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


ACT_DELTA_MIN = _f("ACT_DELTA_MIN", 0.30)
# [FIX-SMALL-LOAD] Satu kotak di bak hampir tidak menggeser skor global MobileNetV2 (delta kecil).
# Bila forklift terbukti bekerja di dekat kendaraan (>= ACT_FORKLIFT_NEAR_SECONDS), ambang delta diturunkan.
ACT_DELTA_MIN_FK = _f("ACT_DELTA_MIN_FK", 0.12)
ACT_PHASE_SAMPLES = _i("ACT_PHASE_SAMPLES", 6)
ACT_MIN_PHASE_SAMPLES = _i("ACT_MIN_PHASE_SAMPLES", 2)

# Di loading dock forklift/orang TIDAK terlihat kamera, jadi bukti hanya perubahan skor itu sendiri. Lonjakan "tiba-tiba terisi"
# berupa anak tangga yang stabil, maka ambangnya lebih rendah tetapi jendela harus lebih rapat (ACT_TR_MAX_SPREAD_DOCK).
ACT_DELTA_MIN_DOCK = _f("ACT_DELTA_MIN_DOCK", 0.20)
ACT_TR_MAX_SPREAD_DOCK = _f("ACT_TR_MAX_SPREAD_DOCK", 0.25)
# Skor muatan MobileNetV2 berisik (truk kosong statis pun berayun 0.07-0.43 di log uji; orang/forklift di crop mendorongnya).
# Ambang 0.12 terlalu rendah -> kejadian palsu Loading/Unloading bergantian. Ambang transisi sendiri, lebih tinggi.
ACT_TR_DELTA = _f("ACT_TR_DELTA", 0.30)
ACT_TR_DELTA_EVIDENCE = _f("ACT_TR_DELTA_EVIDENCE", 0.22)
ACT_TR_STATIONARY_ONLY = os.environ.get("ACT_TR_STATIONARY_ONLY", "1").strip().lower() not in ("0", "false", "no", "off")
ACT_TR_WINDOW = _i("ACT_TR_WINDOW", 4)
ACT_TR_MAX_SPREAD = _f("ACT_TR_MAX_SPREAD", 0.35)
ACT_TR_SETTLE_SECONDS = _f("ACT_TR_SETTLE_SECONDS", 12.0)
ACT_TR_EVIDENCE_LOOKBACK = _f("ACT_TR_EVIDENCE_LOOKBACK", 30.0)
# [FIX-UNLOAD-BASE] dulu 0.30: lonjakan palsu 0.16->0.39 (masih wilayah "kosong") lalu turun 0.39->0.12 lolos sebagai
# Unloading. Unloading hanya masuk akal bila baseline-nya benar-benar "bermuatan" (>= 0.5).
ACT_TR_MIN_BASE_UNLOAD = _f("ACT_TR_MIN_BASE_UNLOAD", 0.50)
ACT_TR_MAX_BASE_LOAD = _f("ACT_TR_MAX_BASE_LOAD", 0.90)
ACT_MIN_SEPARATION_SECONDS = _f("ACT_MIN_SEPARATION_SECONDS", 8.0)
BOX_CONFIRM_HITS = _i("CARGO_BOX_CONFIRM_HITS", 5)
ACT_FORKLIFT_NEAR_SECONDS = _f("ACT_FORKLIFT_NEAR_SECONDS", 15.0)
ACT_FORKLIFT_NEAR_MARGIN = _f("ACT_FORKLIFT_NEAR_MARGIN", 0.30)

# [FIX-CROSS-BOUNDARY] Selisih skor sedikit di bawah ACT_DELTA_MIN, TAPI median skor melewati batas 0.5
# (tiba < 0.5 lalu pergi >= 0.5 = Loading; kebalikannya = Unloading) dan selisihnya >= nilai ini.
# Ini murni dari bukti skor (bukan asumsi). Nilai > ACT_DELTA_MIN atau < 0 = matikan aturan ini.
ACT_CROSS_DELTA_MIN = _f("ACT_CROSS_DELTA_MIN", 0.20)  # [DIHAPUS] aturan lintas-batas 0.5 dicabut (lihat [FIX-ARRIVAL-STATE]); konstanta dibiarkan agar _env lama tidak error

# [FIX-ARRIVAL-STATE] Skor tiba itu BERISIK (orang/forklift di dalam crop bisa mendorong skor truk kosong ke 0.8-0.9).
# Median saja bisa tipis di sekitar 0.5 (kasus nyata: 0.502 padahal truk kosong -> salah "Unloading").
# Jadi keadaan tiba dibaca dengan dua ukuran:
#   s_low = median dari separuh sampel TERENDAH (tahan terhadap lonjakan palsu ke atas)
#   "kosong": s_low < 0.5 dan median < ACT_ARRIVAL_FULL_MIN   -> tidak mungkin Unloading
#   "penuh" : median >= ACT_ARRIVAL_FULL_MIN dan s_low >= 0.5 -> boleh dianggap Unloading (asumsi)
#   selain itu "ragu": tidak dipakai untuk menebak arah.
ACT_ARRIVAL_FULL_MIN = _f("ACT_ARRIVAL_FULL_MIN", 0.65)

# [FIX-FORKLIFT-FALLBACK] Bila forklift aktif di sekitar kendaraan >= ACT_FORKLIFT_NEAR_SECONDS tetapi skor muatan
# tidak bisa memutuskan arah (Transit / Lewat Kosong / Tidak Dapat Dipastikan), jangan lapor "Transit":
#   - ACT_FORKLIFT_DEFAULT_DIRECTION kosong      -> "Aktivitas Forklift (arah belum pasti)"
#   - ACT_FORKLIFT_DEFAULT_DIRECTION=LOADING     -> Loading   (hanya ASUMSI konfigurasi, tertulis di reason)
#   - ACT_FORKLIFT_DEFAULT_DIRECTION=UNLOADING   -> Unloading (idem)
# Bukti skor yang jelas berlawanan dengan arah default tidak ditimpa.
ACT_FORKLIFT_DEFAULT_DIRECTION = os.environ.get("ACT_FORKLIFT_DEFAULT_DIRECTION", "").strip().upper()

# [FIX-CLOSED-DOOR] Pintu belakang van TERTUTUP -> isi tidak terlihat, tidak ada sampel saat pergi. Tetapi bila
# kendaraan tiba KOSONG (skor tiba <= nilai ini, terbaca jelas saat pintu masih terbuka) dan forklift bekerja di
# sekitarnya >= ACT_FORKLIFT_NEAR_SECONDS, satu-satunya kemungkinan adalah LOADING (tidak ada yang bisa
# dibongkar). Hanya berlaku bila TIDAK ada bukti saat pergi sama sekali. Nilai < 0 = matikan aturan ini.
ACT_EMPTY_ARRIVAL_MAX = _f("ACT_EMPTY_ARRIVAL_MAX", 0.30)

# [FIX-EMPTY-STAY] Truk TIBA kosong (median skor tiba < 0.5), forklift bekerja di sekitarnya >= ACT_FORKLIFT_NEAR_SECONDS,
# tetapi skor saat pergi tetap rendah (Lewat Kosong) atau naik tidak cukup jelas (Tidak Dapat Dipastikan):
# truk yang tiba kosong TIDAK mungkin Unloading, jadi hasilnya dilaporkan Loading (isi bak sering tidak terbaca
# classifier: barang rendah/kecil, sudut kamera, tertutup forklift). Ini menimpa bukti "tetap kosong" -> hanya
# berlaku bila forklift terbukti dekat. Matikan dengan ACT_EMPTY_STAY_LOADING=0.
ACT_EMPTY_STAY_LOADING = _i("ACT_EMPTY_STAY_LOADING", 1)

# [FIX-LOADED-ARRIVAL] Truk TIBA bermuatan (median skor tiba >= 0.5), forklift bekerja di sekitarnya >= ACT_FORKLIFT_NEAR_SECONDS,
# dan skor TIDAK naik jelas (selisih <= 0.10 atau bukti saat pergi tidak ada): kemungkinan terbesar Unloading.
# Ini ASUMSI (truk bisa saja menambah muatan atau parkir dekat forklift), selalu ditulis di reason dengan confidence
# rendah. Bukti skor yang jelas (selisih >= ACT_DELTA_MIN) tetap menang. Kendaraan bodi boks tidak terkena aturan ini.
# Matikan dengan ACT_LOADED_ARRIVAL_UNLOADING=0.
ACT_LOADED_ARRIVAL_UNLOADING = _i("ACT_LOADED_ARRIVAL_UNLOADING", 1)

# [FIX-ORANGE] Crop yang memuat forklift oranye (>= nilai ini) tidak dinilai: yang terbaca palet/forklift,
# bukan isi truk. Crop van tanpa forklift = 0%, crop dengan forklift = 2.7-7.9% (diukur dari log uji).
ACT_ORANGE_CROP_PCT = _f("ACT_ORANGE_CROP_PCT", 1.0)


def _orange_pct(crop):
    try:
        import cv2
        small = cv2.resize(crop, (160, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (3, 140, 120), (20, 255, 255))
        return float(m.mean()) / 255.0 * 100.0
    except Exception:
        return 0.0

_forklift_lock = threading.Lock()
_forklift_recent = deque(maxlen=300)


def note_forklift(bbox, now):
    try:
        with _forklift_lock:
            _forklift_recent.append((float(now), tuple(float(v) for v in bbox)))
    except Exception:
        pass


def note_skip(vehicle, reason):
    """Dipanggil aw.py tiap kali satu frame milik sesi dibuang gerbang forklift (reason: 'oranye' / 'yolo')."""
    try:
        sk = _state(vehicle).setdefault("skips", {})
        sk[reason] = sk.get(reason, 0) + 1
    except Exception:
        pass


def _forklift_near(vehicle_bbox, now):
    if not vehicle_bbox:
        return False
    x1, y1, x2, y2 = vehicle_bbox
    mx = (x2 - x1) * ACT_FORKLIFT_NEAR_MARGIN
    my = (y2 - y1) * ACT_FORKLIFT_NEAR_MARGIN
    x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    with _forklift_lock:
        items = list(_forklift_recent)
    for t, (fx1, fy1, fx2, fy2) in items:
        if now - t > 1.5:
            continue
        if fx1 < x2 and fx2 > x1 and fy1 < y2 and fy2 > y1:
            return True
    return False


def _arrival_state(scores):
    """Return (state, s_low). state: 'kosong' | 'penuh' | 'ragu' (lihat [FIX-ARRIVAL-STATE])."""
    vals = sorted(scores)
    if not vals:
        return None, None
    med = _median(vals)
    half = vals[:max(1, (len(vals) + 1) // 2)]
    s_low = _median(half)
    if s_low < 0.5 and med < ACT_ARRIVAL_FULL_MIN:
        return "kosong", s_low
    if med >= ACT_ARRIVAL_FULL_MIN and s_low >= 0.5:
        return "penuh", s_low
    return "ragu", s_low


def _median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


# ---- Debug: lihat persis crop apa yang dinilai pengklasifikasi muatan ----
# Tiap sampel dicetak ke log ([ACT-SAMPLE]) dan crop-nya disimpan beserta skor P(bermuatan)
# ke ACT_DEBUG_DIR/<vehicle_id>/ . Sampel TIBA semuanya disimpan; sampel lanjutan tiap >=10 dtk video.
# Matikan penyimpanan gambar dengan ACT_DEBUG_DIR= (kosong).
# [DEPLOY] Default MATI di produksi (dulu selalu menyimpan gambar). Nyalakan: ACT_DEBUG_DIR=forklift_debug/activity
# atau DEBUG_ARTIFACTS=1 (video.py menyalakannya otomatis). ACT_DEBUG_MAX_TOTAL membatasi jumlah file per proses.
_DBG_ON = os.environ.get("DEBUG_ARTIFACTS", "0").strip().lower() in ("1", "true", "yes", "on")
ACT_DEBUG_DIR = os.environ.get("ACT_DEBUG_DIR", os.path.join("forklift_debug", "activity") if _DBG_ON else "")
ACT_DEBUG_MAX_TOTAL = int(float(os.environ.get("ACT_DEBUG_MAX_TOTAL", "20000")))
_dbg_total = 0
ACT_DEBUG_LATER_GAP = _f("ACT_DEBUG_LATER_GAP", 10.0)
ACT_DEBUG_MAX_PER_VEHICLE = _i("ACT_DEBUG_MAX_PER_VEHICLE", 60)
T0 = None


def set_t0(t):
    global T0
    if T0 is None and t is not None:
        T0 = float(t)


def _debug_sample(vehicle, st, crop, sample, is_arrival):
    global _dbg_total
    try:
        ts, score = sample
        rel = ts - (T0 if T0 is not None else ts)
        vid = str(vehicle.get("vehicle_id", "?"))
        short = vid.split("_")[-1]
        phase = "TIBA" if is_arrival else "lanjut"
        n_in = len(st["arrival"])
        if is_arrival or ts - st.get("dbg_last_t", -1e9) >= ACT_DEBUG_LATER_GAP:
            if not is_arrival:
                st["dbg_last_t"] = ts
            print(f"[ACT-SAMPLE] {vid} {phase} t+{rel:.0f}s P(bermuatan)={score:.2f} "
                  f"(sampel tiba terkumpul={n_in}/{ACT_PHASE_SAMPLES})")
            if (ACT_DEBUG_DIR and crop is not None and st.get("dbg_saved", 0) < ACT_DEBUG_MAX_PER_VEHICLE
                    and _dbg_total < ACT_DEBUG_MAX_TOTAL):
                _dbg_total += 2
                import cv2
                folder = os.path.join(ACT_DEBUG_DIR, vid)
                os.makedirs(folder, exist_ok=True)
                img = crop.copy()
                cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
                cv2.putText(img, f"#{short} {phase} t+{rel:.0f}s P(bermuatan)={score:.2f}", (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                st["dbg_saved"] = st.get("dbg_saved", 0) + 1
                _name = f"{st['dbg_saved']:02d}_{phase}_t{rel:04.0f}s_p{score:.2f}.jpg"
                cv2.imwrite(os.path.join(folder, _name), img)
                # Salinan BERSIH (tanpa bar teks) di subfolder clean/ -> bahan label_tool.py / train_classifier.py
                _clean = os.path.join(folder, "clean")
                os.makedirs(_clean, exist_ok=True)
                cv2.imwrite(os.path.join(_clean, f"{vid}_{_name}"), crop)
    except Exception as e:
        print(f"[ACT-SAMPLE] gagal debug: {e}")


def _state(vehicle):
    st = vehicle.get("_act")
    if st is None:
        st = {
            "arrival": [],
            "latest": deque(maxlen=ACT_PHASE_SAMPLES * 2),
            "last_crop_ts": 0.0,
            "box_streak": 0,
            "fk_seconds": 0.0,
            "last_fk_t": None,
            "orange_skipped": 0,
            # [MULTI-EVENT] seluruh deret skor sesi (bukan hanya 6 pertama/terakhir) untuk deteksi perubahan berulang
            "series": deque(maxlen=4000),
            "tr_win": deque(maxlen=max(2, ACT_TR_WINDOW)),
            "tr_last_ts": -1e18,
            "tr_base": None,
            "tr_pending": None,
        }
        vehicle["_act"] = st
    return st


def _bump(st, key):
    d = st.setdefault("diag", {})
    d[key] = d.get(key, 0) + 1


def add_samples(vehicle, now, score_fn, box_fn):
    st = _state(vehicle)
    _bump(st, "panggilan")

    last = st["last_fk_t"]
    st["last_fk_t"] = now
    if last is not None and _forklift_near(vehicle.get("last_bbox"), now):
        st["fk_seconds"] += min(3.0, max(0.0, now - last))

    if vehicle.get("is_box_body"):
        _bump(st, "bodi_boks")
        return

    crops = [c for c in (vehicle.get("act_crops") or vehicle.get("recent_crops") or [])
             if c.get("usable_cargo", True) and c.get("timestamp", 0.0) > st["last_crop_ts"]]
    if not crops:
        _bump(st, "tanpa_crop_baru")
        return
    item = max(crops, key=lambda c: c.get("timestamp", 0.0))
    st["last_crop_ts"] = item.get("timestamp", now)
    crop = item.get("crop")
    if crop is None:
        return

    if ACT_ORANGE_CROP_PCT > 0 and _orange_pct(crop) >= ACT_ORANGE_CROP_PCT:
        st["orange_skipped"] = st.get("orange_skipped", 0) + 1
        _bump(st, "oranye")
        return

    try:
        is_box = bool(box_fn(crop))
    except Exception:
        is_box = False
    if is_box:
        _bump(st, "dinilai_boks")
        st["box_streak"] += 1
        if st["box_streak"] >= BOX_CONFIRM_HITS:
            vehicle["is_box_body"] = True
            print(f"[ACTIVITY] {vehicle.get('vehicle_id', '?')}: dikunci BODI BOKS TERTUTUP "
                  f"({st['box_streak']}x berturut) -> status 'Tidak Dapat Dipastikan'")
        return
    st["box_streak"] = 0

    try:
        score = score_fn(crop)
    except Exception:
        score = None
    if score is None:
        _bump(st, "skor_none")
        return
    _bump(st, "sampel_masuk")

    sample = (float(item.get("timestamp", now)), float(score))
    is_arrival = len(st["arrival"]) < ACT_PHASE_SAMPLES
    if is_arrival:
        st["arrival"].append(sample)
    st["latest"].append(sample)
    # [FIX-GERAK] Skor saat kendaraan BERGERAK (datang/pergi/mundur) tidak mencerminkan isi bak: sudut & ukuran crop berubah.
    # Log 08 Okt: Granmax (aslinya Loading) mendapat "Unloading 0.77->0.18" ~6 dtk sebelum pergi. Sampel bergerak tetap dipakai
    # decide() lama (arrival/latest), tetapi TIDAK masuk deret deteksi perubahan. Matikan: ACT_TR_STATIONARY_ONLY=0
    if (not ACT_TR_STATIONARY_ONLY) or item.get("is_stationary", True):
        st["series"].append(sample)
    else:
        _bump(st, "seri_bergerak")
    _debug_sample(vehicle, st, crop, sample, is_arrival)


def arrival_label(vehicle):
    """Label muatan SAAT TIBA untuk event MASUK: 'Kosong' | 'Bermuatan' | None (belum cukup sampel / ragu)."""
    st = vehicle.get("_act")
    if not st or vehicle.get("is_box_body"):
        return None
    arr = st.get("arrival") or []
    if len(arr) < ACT_MIN_PHASE_SAMPLES:
        return None
    state, _ = _arrival_state([sc for _, sc in arr])
    return {"kosong": "Kosong", "penuh": "Bermuatan"}.get(state)


# ---------------------------------------------------------------------------------------------------------------
# [MULTI-EVENT] Deteksi PERUBAHAN muatan berulang di dalam SATU sesi kendaraan.
# decide() lama hanya membandingkan 6 sampel pertama vs 6 terakhir -> satu sesi = satu kesimpulan, sehingga
# truk yang diisi lalu dibongkar (atau diisi dua kali) tetap "Lewat Kosong". poll_transitions() memantau deret skor:
#   - jendela ACT_TR_WINDOW sampel berurutan yang KONSISTEN (selisih max-min <= ACT_TR_MAX_SPREAD)
#   - median jendela bergeser >= ambang dari baseline  -> perubahan (naik = Loading, turun = Unloading)
#   - ambang turun ke ACT_DELTA_MIN_FK bila forklift/orang terlihat di dekat kendaraan dalam ACT_TR_EVIDENCE_LOOKBACK dtk
#   - perubahan searah yang berdekatan digabung (barang diangkut bertahap), baru dikirim setelah ACT_TR_SETTLE_SECONDS
#     tanpa perubahan lagi
# Unloading diabaikan bila baseline sudah kosong (< ACT_TR_MIN_BASE_UNLOAD), Loading diabaikan bila sudah penuh.
# ---------------------------------------------------------------------------------------------------------------
def poll_transitions(vehicle, now, evidence_fn=None, flush=False, in_dock=False):
    """Return list event perubahan yang SELESAI (sudah settle). evidence_fn(t0, t1) -> (detik_forklift, detik_orang)."""
    st = _state(vehicle)
    out = []
    K = max(2, ACT_TR_WINDOW)

    def _evidence(t0, t1):
        if evidence_fn is None:
            return 0.0, 0.0
        try:
            return evidence_fn(t0, t1)
        except Exception:
            return 0.0, 0.0

    def _finalize(pend):
        before, after = pend["before"], pend["after"]
        delta = after - before
        # Perubahan harus BERTAHAN: bila sampel sesudahnya kembali ke level semula, itu orang/forklift lewat di crop.
        later = [sc for t_, sc in st["series"] if t_ > pend["t_last"]]
        if len(later) >= 2 and abs(_median(later) - after) > 0.5 * abs(delta):
            st["tr_loaded"] = st.get("tr_loaded_before", st.get("tr_loaded"))
            st["tr_base"] = _median(later)
            return
        # [FIX-LOADED-STATE] Loading yang berakhir di bawah 0.5 (masih wilayah "kosong") BUKAN bukti kendaraan terisi;
        # dulu langsung tr_loaded=True sehingga noise berikutnya boleh jadi "Unloading".
        if pend["dir"] > 0:
            st["tr_loaded"] = True if after >= 0.5 else st.get("tr_loaded_before", st.get("tr_loaded"))
        else:
            st["tr_loaded"] = False
        verified = len(later) >= 2
        fk_s, pr_s = _evidence(pend["t_start"] - ACT_TR_EVIDENCE_LOOKBACK, pend["t_last"] + ACT_TR_SETTLE_SECONDS)
        has_ev = (fk_s + pr_s) >= 2.0
        thr = ACT_TR_DELTA_EVIDENCE if has_ev else ACT_TR_DELTA
        if in_dock:
            thr = min(thr, ACT_DELTA_MIN_DOCK)
        conf = max(0.3, min(0.95, abs(delta) / (2.0 * thr))) + (0.1 if has_ev else 0.0)
        pelaku = "forklift" if fk_s >= pr_s and fk_s >= 2.0 else ("orang" if pr_s >= 2.0 else "tidak terlihat")
        act = LOADING if pend["dir"] > 0 else UNLOADING
        out.append({
            "activity": act,
            "source": "skor+forklift" if has_ev else "skor",
            "confidence": round(min(0.99, conf), 3),
            "verified": verified,
            "reason": (
                f"muatan {'bertambah' if pend['dir'] > 0 else 'berkurang'} "
                f"({before:.2f} -> {after:.2f}, selisih {delta:+.2f}, ambang {thr:.2f}); "
                + ("di loading dock (pelaku tidak terlihat kamera)" if in_dock else
                   f"pelaku terlihat: {pelaku} (forklift ~{fk_s:.0f}s, orang ~{pr_s:.0f}s)")
                + ("" if verified else "; BELUM TERVERIFIKASI (tak ada sampel stabil sesudahnya, mis. kendaraan keburu pergi)")
            ),
            "t_start": pend["t_start"], "t_end": pend["t_last"],
            "score_before": round(before, 3), "score_after": round(after, 3), "delta": round(delta, 3),
            "pelaku": pelaku,
            "di_dock": bool(in_dock),
        })

    fresh = [x for x in st["series"] if x[0] > st["tr_last_ts"]]
    for ts, sc in fresh:
        st["tr_last_ts"] = ts
        st["tr_win"].append((ts, sc))
        if st["tr_base"] is None:
            if len(st["tr_win"]) >= K:  # baseline awal = K sampel pertama sesi
                arr = [sc for _, sc in (st.get("arrival") or [])]
                a_state, a_low = _arrival_state(arr) if len(arr) >= ACT_MIN_PHASE_SAMPLES else (None, None)
                # tiba kosong: baseline = separuh sampel terendah (tahan lonjakan palsu ke atas dari orang/forklift)
                st["tr_base"] = a_low if a_state == "kosong" else _median([v for _, v in st["tr_win"]])
                st["tr_loaded"] = True if a_state == "penuh" else (False if a_state == "kosong" else None)
                st["tr_win"].clear()
            continue
        if len(st["tr_win"]) < K:
            continue
        win = list(st["tr_win"])
        vals = [v for _, v in win]
        if max(vals) - min(vals) > (ACT_TR_MAX_SPREAD_DOCK if in_dock else ACT_TR_MAX_SPREAD):
            continue  # jendela belum stabil (barang sedang bergerak / orang lewat)
        m = _median(vals)
        base = st["tr_base"]
        d = m - base
        fk_s, pr_s = _evidence(ts - ACT_TR_EVIDENCE_LOOKBACK, ts)
        thr = ACT_TR_DELTA_EVIDENCE if (fk_s + pr_s) >= 2.0 else ACT_TR_DELTA
        if in_dock:
            thr = min(thr, ACT_DELTA_MIN_DOCK)
        if abs(d) < thr:
            continue
        direction = 1 if d > 0 else -1
        if direction < 0 and base < ACT_TR_MIN_BASE_UNLOAD:
            continue  # tidak ada yang bisa dibongkar: noise
        if direction < 0 and st.get("tr_loaded") is False and st["tr_pending"] is None:
            continue  # tiba kosong & belum pernah terisi: tidak mungkin Unloading
        if direction > 0 and base > ACT_TR_MAX_BASE_LOAD:
            continue
        pend = st["tr_pending"]
        st["tr_loaded_before"] = st.get("tr_loaded")
        if pend is not None and pend["dir"] != direction:
            _finalize(pend)
            pend = None
        if pend is None:
            st["tr_pending"] = {"dir": direction, "before": base, "after": m,
                                "t_start": win[0][0], "t_last": ts}
        else:
            pend["after"], pend["t_last"] = m, ts
        st["tr_base"] = m
        st["tr_win"].clear()

    pend = st["tr_pending"]
    if pend is not None and (flush or now - pend["t_last"] >= ACT_TR_SETTLE_SECONDS):
        _finalize(pend)
        st["tr_pending"] = None
    return out


def _forklift_fallback(activity, reason, delta, conf, fk_seconds, s_in=None, s_out=None, n_in=0, a_state=None):
    # Mengembalikan (activity, reason, conf, source). source=None -> hasil asli dari skor (tidak ditimpa).
    # CATATAN: aturan EMPTY_STAY_LOADING di bawah MEMANG menimpa EMPTY_PASS bila forklift terbukti dekat
    # (komentar lama yang bilang EMPTY_PASS tidak ditimpa sudah tidak berlaku); hasilnya diberi source="asumsi".
    if fk_seconds < ACT_FORKLIFT_NEAR_SECONDS:
        return activity, reason, conf, None
    note = f"forklift aktif di sekitar kendaraan ~{fk_seconds:.0f}s"
    if (ACT_EMPTY_STAY_LOADING and activity in (EMPTY_PASS, UNKNOWN, TRANSIT)
            and s_in is not None and s_out is not None and a_state == "kosong" and n_in >= ACT_MIN_PHASE_SAMPLES):
        return (LOADING, f"tiba kosong (skor {s_in:.2f}), pergi {s_out:.2f}, lalu {note}; tidak mungkin Unloading "
                         f"-> dilaporkan Loading meski isi bak tidak terbaca jelas (semula: {activity})",
                max(0.45, min(0.6, conf)), "asumsi")
    if (ACT_LOADED_ARRIVAL_UNLOADING and activity in (TRANSIT, UNKNOWN)
            and s_in is not None and a_state == "penuh" and n_in >= ACT_MIN_PHASE_SAMPLES
            and (delta is None or delta <= 0.10)):
        _out = "tidak terlihat" if s_out is None else f"{s_out:.2f}"
        return (UNLOADING, f"tiba bermuatan (skor {s_in:.2f}), pergi {_out}, lalu {note}; kemungkinan terbesar "
                           f"Unloading (ASUMSI, bukan pengamatan; semula: {activity})", 0.4, "asumsi")
    if activity not in (TRANSIT, UNKNOWN):
        return activity, reason, conf, None
    if (ACT_EMPTY_ARRIVAL_MAX >= 0 and s_out is None and s_in is not None
            and s_in <= ACT_EMPTY_ARRIVAL_MAX and n_in >= ACT_MIN_PHASE_SAMPLES):
        return (LOADING, f"tiba KOSONG (skor {s_in:.2f}) lalu {note}; isi saat pergi tidak terlihat "
                         f"(pintu tertutup / tertutup forklift) -> hanya mungkin Loading (semula: {activity})",
                max(0.6, conf), "asumsi")
    direction = ACT_FORKLIFT_DEFAULT_DIRECTION
    if direction in ("LOADING", "UNLOADING"):
        sign = 1.0 if direction == "LOADING" else -1.0
        if delta is not None and delta * sign < -0.5 * ACT_DELTA_MIN:
            return activity, reason, conf, None  # skor cukup jelas berlawanan: jangan ditimpa
        act = LOADING if direction == "LOADING" else UNLOADING
        return (act, f"{note}; arah dari konfigurasi ACT_FORKLIFT_DEFAULT_DIRECTION={direction}, "
                     f"BUKAN dari pengamatan (semula: {activity})", min(0.5, conf if conf else 0.5), "asumsi")
    return FORKLIFT_ACTIVITY, f"{note}; skor muatan tidak bisa menentukan arah (semula: {activity})", 0.3, "tidak_pasti"


def decide(vehicle):
    st = vehicle.get("_act") or _state(vehicle)
    # [FIX-FK-EVIDENCE] Forklift yang terlihat DI DALAM crop truk sendiri (crop dibuang karena oranye) adalah bukti
    # kontak yang langsung: tiap crop ~1 detik. Ambil yang terbesar dari dua ukuran kedekatan itu.
    fk = max(st.get("fk_seconds", 0.0), float(st.get("orange_skipped", 0)))
    fk_note = (f"; forklift aktif di sekitar truk ~{fk:.0f}s" if fk >= ACT_FORKLIFT_NEAR_SECONDS
               else f"; forklift dekat ~{fk:.0f}s (< ambang {ACT_FORKLIFT_NEAR_SECONDS:.0f}s)")
    ctx = {"a_state": None}
    _sk = st.get("skips") or {}
    if _sk:
        fk_note += (f"; {sum(_sk.values())} frame sesi dibuang gerbang forklift "
                    f"(oranye={_sk.get('oranye', 0)}, yolo={_sk.get('yolo', 0)})")
    _dg = st.get("diag")
    if _dg:
        fk_note += "; diag add_samples: " + ", ".join(f"{k}={v}" for k, v in sorted(_dg.items()))
    _osk = st.get("orange_skipped", 0)
    if _osk:
        fk_note += f"; {_osk} crop dibuang karena ada forklift di dalam crop"

    def out(activity, reason, s_in=None, s_out=None, delta=None, conf=0.0, n_in=0, n_out=0):
        activity, reason, conf, source = _forklift_fallback(activity, reason, delta, conf, fk, s_in, s_out, n_in,
                                                            a_state=ctx["a_state"])
        if source is None:
            source = "tidak_pasti" if activity == UNKNOWN else "skor"
        return {
            "activity": activity,
            "source": source,
            "arrival_state": ctx["a_state"],  # kosong | penuh | ragu | None
            "reason": reason + fk_note,
            "score_in": None if s_in is None else round(s_in, 3),
            "score_out": None if s_out is None else round(s_out, 3),
            "delta": None if delta is None else round(delta, 3),
            "confidence": round(float(conf), 3),
            "n_in": n_in,
            "n_out": n_out,
        }

    if vehicle.get("is_box_body"):
        return out(UNKNOWN, "bodi boks tertutup, isi tidak terlihat")

    arrival = st["arrival"]
    if len(arrival) < ACT_MIN_PHASE_SAMPLES:
        return out(UNKNOWN, f"sampel saat tiba kurang ({len(arrival)}<{ACT_MIN_PHASE_SAMPLES})",
                   n_in=len(arrival))

    ctx["a_state"], _s_low = _arrival_state([s for _, s in arrival])
    t_arrival_end = arrival[-1][0]
    departure = [s for s in st["latest"] if s[0] >= t_arrival_end + ACT_MIN_SEPARATION_SECONDS]
    departure = departure[-ACT_PHASE_SAMPLES:]
    s_in = _median([s for _, s in arrival])
    if len(departure) < ACT_MIN_PHASE_SAMPLES:
        return out(UNKNOWN, f"sampel saat pergi kurang ({len(departure)}<{ACT_MIN_PHASE_SAMPLES}) "
                            f"atau jeda < {ACT_MIN_SEPARATION_SECONDS:.0f}s dari sampel tiba",
                   s_in=s_in, n_in=len(arrival), n_out=len(departure))

    s_out = _median([s for _, s in departure])
    delta = s_out - s_in
    n_in, n_out = len(arrival), len(departure)
    # [FIX-SMALL-LOAD] ambang efektif: lebih rendah kalau ada bukti forklift
    d_min = min(ACT_DELTA_MIN, ACT_DELTA_MIN_FK) if fk >= ACT_FORKLIFT_NEAR_SECONDS else ACT_DELTA_MIN
    sample_factor = min(1.0, min(n_in, n_out) / float(max(1, ACT_PHASE_SAMPLES)))

    if delta >= d_min:
        return out(LOADING, f"muatan bertambah (+{delta:.2f}, ambang {d_min:.2f})", s_in, s_out, delta,
                   min(1.0, delta / (2 * d_min)) * sample_factor, n_in, n_out)
    if delta <= -d_min and ctx["a_state"] == "kosong":
        # Truk yang tiba KOSONG tidak mungkin dibongkar: skor turun itu noise (mis. orang lewat di crop tiba).
        return out(UNKNOWN, f"tiba kosong (median {s_in:.2f}, terendah {_s_low:.2f}) tetapi skor turun {delta:.2f}: "
                            f"dianggap noise", s_in, s_out, delta, 0.0, n_in, n_out)
    if delta <= -d_min:
        return out(UNLOADING, f"muatan berkurang ({delta:.2f}, ambang {d_min:.2f})", s_in, s_out, delta,
                   min(1.0, -delta / (2 * d_min)) * sample_factor, n_in, n_out)

    # [FIX-ARRIVAL-STATE] aturan lintas-batas 0.5 dicabut: median tiba yang tipis di 0.5 membuatnya salah membalik arah.

    stable_conf = max(0.0, 1.0 - abs(delta) / ACT_DELTA_MIN) * sample_factor
    if s_in >= 0.5 and s_out >= 0.5:
        return out(TRANSIT, "muatan tetap ada (perubahan kecil)", s_in, s_out, delta, stable_conf, n_in, n_out)
    if s_in < 0.5 and s_out < 0.5:
        return out(EMPTY_PASS, "tetap kosong (perubahan kecil)", s_in, s_out, delta, stable_conf, n_in, n_out)
    return out(UNKNOWN, "skor di sekitar batas, perubahan kecil", s_in, s_out, delta, 0.0, n_in, n_out)