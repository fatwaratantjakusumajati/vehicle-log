"""
boks_activity.py - deteksi AKTIVITAS di sekitar kendaraan TERTUTUP (mobil boks) tanpa classifier muatan & tanpa forklift.

Idenya: isi boks tidak terlihat, jadi yang dibaca adalah KEGIATANNYA:
    kendaraan PARKIR diam  +  ada orang di dekat kendaraan (atau di zona 'stage' yang dekat kendaraan)
    terus-menerus >= --min-active detik  ->  satu AKTIVITAS.
ARAH (kolom 'arah' di CSV) dibaca dari ISI ZONA STAGE (tanpa model tambahan): seberapa banyak lantai kuning stage tertutup barang
di awal vs di akhir aktivitas.  Stage jadi lebih penuh = Unloading, jadi lebih kosong = Loading, tidak berubah = Tidak Dapat Dipastikan.
Aktivitas selesai bila syarat hilang > --end-gap detik. Dua kendaraan = ID berbeda.

Pakai (hanya butuh model YOLO umum + extract_crops.py + cam1_zones.json di folder yang sama):
    python boks_activity.py klip/2-vec-load.mp4 --zones cam1_zones.json --snap-dir snap_boks --csv hasil_boks.csv
    python boks_activity.py klip/2-act.mp4 --start 60 --end 180 --every 1.0        # hanya bagian tertentu
    python boks_activity.py klip/x.mp4 --verbose                                   # satu baris per frame

Di akhir tiap video dicetak DIAGNOSIS (berapa frame ada kendaraan / parkir / orang dekat) sehingga kalau hasilnya 0
langsung kelihatan syarat mana yang tidak pernah terpenuhi.
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np


# ----------------------------------------------------------------------------- geometri
def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    i = _inter(a, b)
    return i / max(1e-6, _area(a) + _area(b) - i)


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _size(b):
    return max(b[2] - b[0], b[3] - b[1], 1.0)


def _expand(b, r):
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - w * r, b[1] - h * r, b[2] + w * r, b[3] + h * r)


def _near_poly(x, y, poly, margin):
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        u = 0.0 if L == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L))
        if ((x - (x1 + u * dx)) ** 2 + (y - (y1 + u * dy)) ** 2) ** 0.5 <= margin:
            return True
    return False


def _stage_rect(stage, margin=0.03):
    xs = [p[0] for p in stage]
    ys = [p[1] for p in stage]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def _fmt(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


# ----------------------------------------------------------------------------- isi zona stage
class StageWatch:
    """Mengukur persentase lantai KUNING stage yang terlihat (tidak tertutup orang/kendaraan) pada tiap frame.
    Barang di atas stage menutupi lantai kuning -> kuning turun = stage lebih penuh."""

    def __init__(self, stage, h_lo=20, h_hi=38, s_min=50, v_min=110, width=240, keep=150.0,
                 min_visible=0.25, delta_thr=0.12, max_delta=0.45, pre=15.0, keep_img=False):
        self.stage, self.hsv = stage, (h_lo, h_hi, s_min, v_min)
        self.width, self.keep, self.min_visible = width, keep, min_visible
        self.delta_thr, self.pre, self.keep_img, self.max_delta = delta_thr, pre, keep_img, max_delta
        self.buf = deque()                       # (t, yellow, valid, bgr|None)
        self.series = []                         # (t, kuning_terlihat, bagian_stage_terlihat)
        self.sheet = []                          # (t, crop, valid) tiap ~10 dtk untuk lembar kontak
        self._last_sheet = -1e9
        self._geo = None

    def _setup(self, W, H):
        xs = [p[0] * W for p in self.stage]
        ys = [p[1] * H for p in self.stage]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(W, int(max(xs)) + 1), min(H, int(max(ys)) + 1)
        sc = min(1.0, self.width / max(1, x1 - x0))
        w, h = max(8, int((x1 - x0) * sc)), max(8, int((y1 - y0) * sc))
        poly = np.array([[(x * W - x0) * w / (x1 - x0), (y * H - y0) * h / (y1 - y0)] for x, y in self.stage], np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly], 255)
        self._geo = (W, H, x0, y0, x1, y1, w, h, m > 0)

    def observe(self, t, frame, veh, per):
        H, W = frame.shape[:2]
        if self._geo is None or self._geo[:2] != (W, H):
            self._setup(W, H)
        _, _, x0, y0, x1, y1, w, h, pm = self._geo
        crop = cv2.resize(frame[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lo, hi, smin, vmin = self.hsv
        yel = (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)
        occ = np.zeros((h, w), np.uint8)
        for b, r in [(b, 0.0) for b in veh] + [(b, 0.12) for b in per]:
            bw, bh = b[2] - b[0], b[3] - b[1]
            a = int((b[0] - bw * r - x0) * w / (x1 - x0))
            c = int((b[1] - bh * r - y0) * h / (y1 - y0))
            e = int((b[2] + bw * r - x0) * w / (x1 - x0)) + 1
            f = int((b[3] + bh * r - y0) * h / (y1 - y0)) + 1
            occ[max(0, c):max(0, f), max(0, a):max(0, e)] = 1
        valid = pm & (occ == 0)
        if t - self._last_sheet >= 10.0:
            self._last_sheet = t
            self.sheet.append((t, crop.copy(), valid.copy()))
        self.series.append((t, float((yel & valid).sum()) / max(1, int(valid.sum())), float(valid.sum()) / max(1, int(pm.sum()))))
        ctrl = (~pm) & (occ == 0)                # piksel pembanding di luar poligon stage (aspal), untuk mendeteksi cahaya berubah
        cv_ = float(hsv[..., 2][ctrl].mean()) if int(ctrl.sum()) >= 50 else float("nan")
        self.buf.append((t, yel & valid, valid, crop if self.keep_img else None, cv_))
        while self.buf and t - self.buf[0][0] > self.keep:
            self.buf.popleft()

    def ref(self, t0, t1, last_k=None):
        """Gabungan (suara mayoritas per piksel) frame t0 <= t <= t1.  -> (yellow, valid, n, frame_contoh) atau None"""
        if self._geo is None:
            return None
        pm = self._geo[8]
        es = [e for e in self.buf if t0 <= e[0] <= t1 and e[2].sum() >= self.min_visible * pm.sum()]
        if last_k:
            es = es[-last_k:]
        if not es:
            return None
        vc = sum(e[2].astype(np.int32) for e in es)
        yc = sum(e[1].astype(np.int32) for e in es)
        valid = vc >= max(1, 0.4 * len(es))
        yellow = valid & (yc >= 0.5 * np.maximum(vc, 1))
        best = max(es, key=lambda e: e[2].sum())
        return {"yellow": yellow, "valid": valid, "n": len(es), "ex": best, "es": es}

    def before(self, start_t):
        r = self.ref(start_t - self.pre, start_t, last_k=6)
        if r is None:                              # klip dimulai langsung dengan aktivitas
            r = self.ref(start_t, start_t + 4.0, last_k=4)
        return r

    def after(self, end_t, now):
        return self.ref(end_t, now)

    def verdict(self, b, a):
        info = {"arah": "Tidak Dapat Dipastikan", "awal": None, "akhir": None, "delta": None, "terlihat": 0.0, "alasan": ""}
        if b is None or a is None:
            info["alasan"] = "frame stage awal/akhir tidak ada yang cukup bersih (tertutup orang/kendaraan)"
            return info
        pm = self._geo[8]
        common = b["valid"] & a["valid"]
        vis = common.sum() / max(1, pm.sum())
        info["terlihat"] = float(vis)
        if vis < self.min_visible:
            info["alasan"] = f"bagian stage yang terlihat di awal DAN akhir hanya {vis:.0%} (< {self.min_visible:.0%})"
            return info
        yb = float(b["yellow"][common].mean())
        ya = float(a["yellow"][common].mean())
        info.update(awal=yb, akhir=ya, delta=ya - yb)
        if abs(ya - yb) < self.delta_thr:
            info["alasan"] = f"perubahan stage {ya - yb:+.0%} di bawah ambang {self.delta_thr:.0%}"
            return info
        vb = [e[4] for e in b["es"] if e[4] == e[4]]
        va = [e[4] for e in a["es"] if e[4] == e[4]]
        if vb and va:
            ratio = float(np.median(va)) / max(1.0, float(np.median(vb)))
            info["cahaya"] = ratio
            if abs(ratio - 1.0) > 0.10:
                info["alasan"] = f"kecerahan sekitar stage berubah {ratio - 1:+.0%} antara awal dan akhir (cahaya/kamera), perubahan kuning tidak bisa dipercaya"
                return info
        if abs(ya - yb) > self.max_delta:
            info["alasan"] = (f"stage berubah {ya - yb:+.0%}: terlalu besar untuk barang, kemungkinan kendaraan/benda besar "
                              f"menutupi stage (kendaraan tidak terdeteksi)")
            return info
        # bukti harus KONSISTEN: frame-frame akhir harus terpisah dari frame-frame awal (bukan sekadar rata-rata bergeser
        # karena cahaya / kamera berubah)
        need = 0.5 * common.sum()
        fb = [float(e[1][common].sum()) / max(1, int((e[2] & common).sum())) for e in b["es"] if (e[2] & common).sum() >= need]
        fa = [float(e[1][common].sum()) / max(1, int((e[2] & common).sum())) for e in a["es"] if (e[2] & common).sum() >= need]
        if len(fb) < 2 or len(fa) < 2:
            info["alasan"] = "frame bersih terlalu sedikit di awal/akhir untuk memastikan perubahan konsisten"
            return info
        if ya < yb:   # kuning turun = stage lebih penuh
            ok = np.percentile(fb, 25) - np.percentile(fa, 75) > 0.02
            arah, why = "Unloading", "stage lebih penuh di akhir"
        else:
            ok = np.percentile(fa, 25) - np.percentile(fb, 75) > 0.02
            arah, why = "Loading", "stage lebih kosong di akhir"
        if ok:
            info["arah"], info["alasan"] = arah, why
        else:
            info["alasan"] = f"selisih {ya - yb:+.0%} tetapi tidak konsisten antar frame (kemungkinan cahaya/kamera berubah)"
        return info

    def montage(self, b, a, info):
        if b is None or a is None or b["ex"][3] is None or a["ex"][3] is None:
            return None
        outs = []
        for r, tag in ((b, "AWAL"), (a, "AKHIR")):
            img = r["ex"][3].copy()
            dim = ~r["valid"]
            img[dim] = (img[dim] * 0.25).astype(np.uint8)
            img[r["yellow"]] = (0, 0, 255)
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, tag, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            outs.append(img)
        h = max(o.shape[0] for o in outs)
        outs = [cv2.copyMakeBorder(o, 0, h - o.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for o in outs]
        m = np.hstack(outs)
        txt = f"{info['arah']}  awal={info['awal'] if info['awal'] is None else round(info['awal'], 2)} akhir={info['akhir'] if info['akhir'] is None else round(info['akhir'], 2)}"
        cv2.putText(m, txt, (6, m.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return m


# ----------------------------------------------------------------------------- inti
class _Track:
    def __init__(self, tid, box, t):
        self.id, self.box, self.first_seen, self.last_seen = tid, box, t, t
        self.hist = deque()              # (t, cx, cy, size)
        self.parked = False
        self.cond_since = None
        self.last_cond = None
        self.active = False
        self.start_t = None
        self.hits = {"orang_dekat": 0, "stage": 0}
        self.max_people = 0
        self.pos = []
        self.n_near = 0                  # jumlah orang terhitung pada frame terakhir
        self.n_stage = 0
        self.near_stage = False          # kendaraan menempel zona stage pada frame terakhir
        self.cond_frames = 0             # frame syarat aktivitas terpenuhi (selama aktivitas ini)
        self.cond_near_stage = 0         # ... di antaranya kendaraan menempel stage
        self.stage_before = None


class BoksActivity:
    def __init__(self, zones, min_active=8.0, end_gap=15.0, grace=4.0, iou_thr=0.25, ttl=40.0,
                 expand=0.20, person_overlap=0.30, park_window=4.0, park_move=0.06,
                 stage_margin=0.04, veh_min_frac=0.10, need_parked=True, use_stage=True,
                 stage_delta=0.12, stage_min_visible=0.25, near_stage_frac=0.5, keep_img=False,
                 virtual=True, virt_min=20.0, virt_ratio=0.5, virt_pad=0.35):
        self.z = zones
        self.min_active, self.end_gap, self.grace = min_active, end_gap, grace
        self.iou_thr, self.ttl, self.expand = iou_thr, ttl, expand
        self.person_overlap = person_overlap
        self.park_window, self.park_move = park_window, park_move
        self.stage_margin, self.veh_min_frac = stage_margin, veh_min_frac
        self.need_parked = need_parked
        stage = getattr(zones, "stage", None) or []
        self.stage = stage if (use_stage and len(stage) >= 3) else []
        self.near_stage_frac = near_stage_frac
        self.sw = StageWatch(self.stage, delta_thr=stage_delta, min_visible=stage_min_visible, keep_img=keep_img) if self.stage else None
        self._now = 0.0
        self.virtual, self.virt_min, self.virt_ratio, self.virt_pad = virtual, virt_min, virt_ratio, virt_pad
        self.base, self.block_since, self.block_last = 0.8, None, -1e9
        self.tracks, self._next = {}, 1
        self.diag = dict(frames=0, f_veh=0, f_parked=0, f_person_near=0, f_stage=0, f_cond=0, f_virt=0)

    # --- pelacakan kendaraan
    def _match(self, t, boxes):
        pairs = []
        for tid, tr in self.tracks.items():
            if t - tr.last_seen > self.ttl:
                continue
            for j, b in enumerate(boxes):
                v = _iou(tr.box, b)
                if v >= self.iou_thr:
                    pairs.append((1.0 + v, tid, j))
                    continue
                (cx, cy), (tx, ty) = _center(b), _center(tr.box)
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / max(_size(b), _size(tr.box))
                if d <= 0.5:
                    pairs.append((1.0 - d, tid, j))
        pairs.sort(reverse=True)
        used_t, used_b, seen = set(), set(), []
        for _, tid, j in pairs:
            if tid in used_t or j in used_b:
                continue
            used_t.add(tid)
            used_b.add(j)
            tr = self.tracks[tid]
            tr.box, tr.last_seen = boxes[j], t
            seen.append(tr)
        for j, b in enumerate(boxes):
            if j in used_b:
                continue
            # pecahan kotak (pintu/bak terbaca terpisah) yang berada di dalam kendaraan lain bukan kendaraan baru
            if any(t - tr.last_seen <= 5.0 and _inter(b, tr.box) / max(1.0, _area(b)) >= 0.7 for tr in self.tracks.values()):
                continue
            tr = _Track(self._next, b, t)
            self._next += 1
            self.tracks[tr.id] = tr
            seen.append(tr)
        return seen

    def _update_motion(self, tr, t):
        cx, cy = _center(tr.box)
        tr.hist.append((t, cx, cy, _size(tr.box)))
        while tr.hist and t - tr.hist[0][0] > self.park_window:
            tr.hist.popleft()
        if tr.hist[-1][0] - tr.hist[0][0] < self.park_window * 0.7:
            return
        xs, ys = [h[1] for h in tr.hist], [h[2] for h in tr.hist]
        move = max(max(xs) - min(xs), max(ys) - min(ys))
        tr.parked = move <= self.park_move * _size(tr.box)

    # --- pemicu
    def _people_near(self, tr, per, W, H):
        ex = _expand(tr.box, self.expand)
        n_near = n_stage = 0
        veh_near_stage = False
        tr.near_stage = False
        if self.stage:
            r = _stage_rect(self.stage)
            nb = (tr.box[0] / W, tr.box[1] / H, tr.box[2] / W, tr.box[3] / H)
            veh_near_stage = _inter(nb, r) / max(1e-6, _area(r)) >= self.veh_min_frac
            tr.near_stage = veh_near_stage
        for p in per:
            if _inter(p, ex) / max(1.0, _area(p)) >= self.person_overlap:
                n_near += 1
            if veh_near_stage:
                fx, fy = (p[0] + p[2]) / 2 / W, p[3] / H
                if self._in_poly(fx, fy) or _near_poly(fx, fy, self.stage, self.stage_margin):
                    n_stage += 1
        return n_near, n_stage

    def _in_poly(self, x, y):
        poly = np.array(self.stage, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    # --- mesin kejadian
    def _close(self, tr):
        n = max(1, len(tr.pos))
        ev = {"type": "selesai", "id": tr.id, "start": tr.start_t, "end": tr.last_cond,
              "hits": dict(tr.hits), "max_people": tr.max_people,
              "pos": (sum(p[0] for p in tr.pos) / n, sum(p[1] for p in tr.pos) / n)}
        ev["arah_info"], ev["stage_img"] = self._arah(tr)
        tr.cond_frames = tr.cond_near_stage = 0
        tr.stage_before = None
        tr.active, tr.start_t, tr.cond_since = False, None, None
        tr.max_people = 0
        return ev

    def _arah(self, tr):
        none = {"arah": "Tidak Dapat Dipastikan", "awal": None, "akhir": None, "delta": None, "terlihat": 0.0, "alasan": ""}
        if self.sw is None:
            none["alasan"] = "poligon stage tidak ada / --no-stage"
            return none, None
        if tr.cond_frames == 0 or tr.cond_near_stage < self.near_stage_frac * tr.cond_frames:
            none["alasan"] = "kendaraan tidak menempel zona stage (arah dari stage tidak berlaku)"
            return none, None
        b = tr.stage_before
        a = self.sw.after(tr.last_cond, self._now)
        info = self.sw.verdict(b, a)
        return info, self.sw.montage(b, a, info)

    def _step(self, tr, t, cond):
        if cond:
            tr.last_cond = t
        if not tr.active:
            if cond:
                if tr.cond_since is None:
                    tr.cond_since = t
                    tr.hits = {"orang_dekat": 0, "stage": 0}
                    tr.max_people = 0
                    tr.cond_frames = tr.cond_near_stage = 0
                if t - tr.cond_since >= self.min_active:
                    tr.active, tr.start_t = True, tr.cond_since
                    tr.stage_before = self.sw.before(tr.start_t) if self.sw else None
                    return {"type": "mulai", "id": tr.id, "start": tr.start_t}
            elif tr.cond_since is not None and (tr.last_cond is None or t - tr.last_cond > self.grace):
                tr.cond_since = None
        else:
            if tr.last_cond is not None and t - tr.last_cond > self.end_gap:
                return self._close(tr)
        return None

    def _virtual_box(self, t, W, H, vehicle_boxes):
        """Kendaraan yang tidak terbaca YOLO (mis. atap truk boks dari atas) tetapi menutupi stage lama -> 1 kotak 'kendaraan virtual'."""
        if not (self.virtual and self.sw is not None and self.sw.series):
            return []
        _, y, v = self.sw.series[-1]
        if v >= 0.4:                              # stage cukup terlihat -> keadaannya bisa dinilai
            if v >= 0.7 and (self.block_since is None) and y > self.base:
                self.base = y
            if y < self.virt_ratio * self.base:
                if self.block_since is None:
                    self.block_since = t
                self.block_last = t
            elif self.block_since is not None and t - self.block_last > 6.0:
                self.block_since = None
        if self.block_since is None or t - self.block_since < self.virt_min:
            return []
        xs = [p[0] for p in self.stage]
        ys = [p[1] for p in self.stage]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        pw, ph = (x1 - x0) * self.virt_pad, (y1 - y0) * self.virt_pad
        box = (max(0.0, x0 - pw) * W, max(0.0, y0 - ph) * H, min(1.0, x1 + pw) * W, min(1.0, y1 + ph) * H)
        if any(_inter(box, b) / max(1.0, _area(box)) >= 0.3 for b in vehicle_boxes):
            return []                              # sudah ada kendaraan nyata di sana
        return [box]

    def update(self, t, W, H, vehicle_boxes, person_boxes, frame=None):
        d = self.diag
        d["frames"] += 1
        self._now = t
        if self.sw is not None and frame is not None:
            self.sw.observe(t, frame, list(vehicle_boxes), list(person_boxes))
        virt = self._virtual_box(t, W, H, list(vehicle_boxes))
        if virt:
            d["f_virt"] += 1
        seen = self._match(t, list(vehicle_boxes) + virt)
        seen_ids = {tr.id for tr in seen}
        if seen:
            d["f_veh"] += 1
        if any(tr.parked for tr in seen):
            d["f_parked"] += 1
        events, any_near, any_stage, any_cond = [], False, False, False
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            cond = False
            if tid in seen_ids:
                self._update_motion(tr, t)
                tr.n_near, tr.n_stage = self._people_near(tr, person_boxes, W, H)
                if len(tr.pos) < 5000:
                    tr.pos.append(((tr.box[0] + tr.box[2]) / 2 / W, (tr.box[1] + tr.box[3]) / 2 / H))
                ok_motion = tr.parked or not self.need_parked
                cond = ok_motion and (tr.n_near > 0 or tr.n_stage > 0)
                if cond:
                    tr.cond_frames += 1
                    tr.cond_near_stage += int(tr.near_stage)
                    if tr.n_near:
                        tr.hits["orang_dekat"] += 1
                    if tr.n_stage:
                        tr.hits["stage"] += 1
                    tr.max_people = max(tr.max_people, tr.n_near, tr.n_stage)
                    any_cond = True
                any_near |= tr.n_near > 0
                any_stage |= tr.n_stage > 0
            else:
                tr.n_near = tr.n_stage = 0
            ev = self._step(tr, t, cond)
            if ev:
                events.append(ev)
            if t - tr.last_seen > self.ttl:
                if tr.active:
                    events.append(self._close(tr))
                del self.tracks[tid]
        d["f_person_near"] += int(any_near)
        d["f_stage"] += int(any_stage)
        d["f_cond"] += int(any_cond)
        return events

    def finish(self):
        return [self._close(tr) for tr in self.tracks.values() if tr.active]


# ----------------------------------------------------------------------------- gambar bukti
def _draw(frame, zones, ba, veh, per, W, H):
    img = frame.copy()
    for poly, col in ((getattr(zones, "area", None), (0, 220, 0)), (getattr(zones, "stage", None), (255, 120, 0))):
        if poly and len(poly) >= 3:
            p = np.array([[x * W, y * H] for x, y in poly], dtype=np.int32)
            cv2.polylines(img, [p], True, col, 4)
    for tr in ba.tracks.values():
        if tr.last_seen != ba.tracks[tr.id].last_seen:
            continue
        b = tr.box
        col = (0, 140, 255) if tr.active else ((0, 255, 255) if tr.parked else (200, 200, 200))
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 5)
        cv2.putText(img, f"#{tr.id} {'PARKIR' if tr.parked else 'gerak'}{' AKTIF' if tr.active else ''}",
                    (int(b[0]) + 8, int(b[1]) + 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4)
    for p in per:
        cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255, 0, 255), 4)
    return cv2.resize(img, (1280, int(1280 * H / W)))


# ----------------------------------------------------------------------------- uji di video
def run_video(path, det, zones, args, writer):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("  ! tidak bisa dibuka:", path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 120:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * args.every)))
    idx = int(args.start * fps) if args.start > 0 else 0
    if idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    end_idx = int(args.end * fps) if args.end > 0 else (total or 10 ** 12)
    print(f"  fps={fps:.1f} total_frame={total} (~{total / fps:.0f} dtk) | analisis tiap {step} frame "
          f"| rentang {_fmt(idx / fps)} - {_fmt(min(end_idx, total or end_idx) / fps)}", flush=True)

    ba = BoksActivity(zones, min_active=args.min_active, end_gap=args.end_gap, expand=args.expand,
                      person_overlap=args.person_overlap, park_move=args.park_move,
                      need_parked=not args.no_parked, use_stage=not args.no_stage,
                      virtual=not args.no_virtual, virt_min=args.virt_min, virt_ratio=args.virt_ratio,
                      stage_delta=args.stage_delta, stage_min_visible=args.stage_visible, keep_img=bool(args.snap_dir))
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0].replace(" ", "_")
    n_ev, n_done, t_start, last_snap = 0, 0, time.time(), {}
    spans = []

    def snap(tag, tid, t, frame, veh, per):
        if not args.snap_dir:
            return
        H, W = frame.shape[:2]
        os.makedirs(args.snap_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.snap_dir, f"{stem}_k{tid}_{int(t):05d}_{tag}.jpg"), _draw(frame, zones, ba, veh, per, W, H))

    def handle(ev, frame=None, veh=(), per=()):
        nonlocal n_ev
        if ev["type"] == "mulai":
            print(f"  MULAI    kendaraan #{ev['id']}  {_fmt(ev['start'])}")
            if frame is not None:
                snap("mulai", ev["id"], ev["start"], frame, veh, per)
        else:
            h, (px, py) = ev["hits"], ev["pos"]
            dur = ev["end"] - ev["start"]
            if ev["max_people"] < args.min_people:
                print(f"  (dibuang) kendaraan #{ev['id']} {_fmt(ev['start'])}: orang_maks={ev['max_people']} < --min-people {args.min_people}")
                return
            ai = ev["arah_info"]
            if args.snap_dir and ev.get("stage_img") is not None:
                os.makedirs(args.snap_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.snap_dir, f"{stem}_k{ev['id']}_{int(ev['start']):05d}_stage.jpg"), ev["stage_img"])
            print(f"  SELESAI  kendaraan #{ev['id']}  {_fmt(ev['start'])} - {_fmt(ev['end'])}  ({dur:.0f} dtk)  "
                  f"pemicu: orang_dekat={h['orang_dekat']} stage={h['stage']}  orang_maks={ev['max_people']}  "
                  f"posisi=({px:.2f},{py:.2f})")
            dl = "-" if ai["delta"] is None else f"{ai['delta']:+.2f}"
            print(f"           ARAH: {ai['arah']}  (kuning stage awal->akhir: "
                  f"{'-' if ai['awal'] is None else format(ai['awal'], '.2f')} -> {'-' if ai['akhir'] is None else format(ai['akhir'], '.2f')}, "
                  f"selisih {dl}, stage terlihat {ai['terlihat']:.0%})  {ai['alasan']}")
            writer.writerow([name, ev["id"], f"{ev['start']:.1f}", f"{ev['end']:.1f}", f"{dur:.1f}", _fmt(ev["start"]),
                             h["orang_dekat"], h["stage"], ev["max_people"], f"{px:.3f}", f"{py:.3f}",
                             ai["arah"], "" if ai["awal"] is None else f"{ai['awal']:.3f}", "" if ai["akhir"] is None else f"{ai['akhir']:.3f}",
                             "" if ai["delta"] is None else f"{ai['delta']:+.3f}", f"{ai['terlihat']:.2f}", ai["alasan"]])
            n_ev += 1
            spans.append((ev["start"], ev["end"], ai["arah"]))

    while idx < end_idx:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                t = idx / fps
                H, W = frame.shape[:2]
                veh, per = [], []
                for (x1, y1, x2, y2, conf, cls) in det.detect(frame):
                    box = (x1, y1, x2, y2)
                    if cls == 0:
                        if conf >= args.conf:
                            per.append(box)
                    elif cls in (2, 7) and conf >= args.veh_conf:
                        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                        if zones.area and not zones.in_area(cx, cy):
                            continue
                        if _area(box) / (W * H) < args.min_ratio:
                            continue
                        veh.append(box)
                events = ba.update(t, W, H, veh, per, frame)
                for ev in events:
                    handle(ev, frame, veh, per)
                # bukti berkala selama aktivitas berjalan
                for tr in ba.tracks.values():
                    if tr.active and args.snap_dir and t - last_snap.get(tr.id, -1e9) >= args.snap_every:
                        last_snap[tr.id] = t
                        snap("jalan", tr.id, t, frame, veh, per)
                n_done += 1
                if args.verbose:
                    st = " ".join(f"#{tr.id}:{'P' if tr.parked else 'g'}o{tr.n_near}s{tr.n_stage}" for tr in ba.tracks.values()
                                  if tr.last_seen == t) or "-"
                    print(f"  t={_fmt(t)} kendaraan={len(veh)} orang={len(per)} | {st}", flush=True)
                elif n_done == 1 or n_done % 10 == 0:
                    el = (time.time() - t_start) / n_done
                    sisa = (min(end_idx, total or end_idx) - idx) / fps / args.every * el
                    print(f"  ... video {_fmt(t)} | {n_done} frame | {el:.1f} dtk/frame | sisa ~{sisa / 60:.1f} menit", flush=True)
        idx += 1
    cap.release()
    for ev in ba.finish():
        handle(ev)
    if args.stage_log and ba.sw is not None and ba.sw.series:
        _stage_plot(ba.sw.series, spans, os.path.join(args.snap_dir or ".", f"{stem}_stage_grafik"))
        _stage_sheet(ba.sw.sheet, os.path.join(args.snap_dir or ".", f"{stem}_stage_kontak.jpg"))
    d = ba.diag
    print(f"  -> {n_ev} aktivitas")
    print(f"  DIAGNOSIS ({d['frames']} frame dianalisis): ada kendaraan={d['f_veh']} | kendaraan PARKIR={d['f_parked']} | "
          f"orang dekat kendaraan={d['f_person_near']} | orang di stage={d['f_stage']} | syarat aktivitas terpenuhi={d['f_cond']} | "
          f"kendaraan virtual (benda besar di stage)={d['f_virt']}")
    if d["f_veh"] == 0:
        print("     -> kendaraan tidak pernah lolos: turunkan --veh-conf / --min-ratio, atau cek poligon 'area'.")
    elif d["f_parked"] == 0:
        print("     -> kendaraan tidak pernah dianggap PARKIR: naikkan --park-move (mis. 0.12) atau pakai --no-parked.")
    elif d["f_person_near"] == 0 and d["f_stage"] == 0:
        print("     -> tidak ada orang dekat kendaraan: turunkan --conf (mis. 0.2) / naikkan --expand (mis. 0.4).")
    elif d["f_cond"] > 0:
        print("     -> syarat terpenuhi tetapi tidak sampai --min-active berturut-turut: turunkan --min-active / naikkan --end-gap.")


def _stage_plot(series, spans, base):
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_dtk", "mmss", "kuning_terlihat", "stage_terlihat"])
        for t, y, v in series:
            w.writerow([f"{t:.1f}", _fmt(t), f"{y:.3f}", f"{v:.2f}"])
    W, H, L, B, T = 1600, 420, 60, 40, 20
    img = np.full((H, W, 3), 255, np.uint8)
    t1 = max(s[0] for s in series) or 1.0
    X = lambda t: int(L + (W - L - 10) * t / t1)
    Y = lambda y: int(H - B - (H - B - T) * y)
    for a, b, arah in spans:
        cv2.rectangle(img, (X(a), T), (X(b), H - B), (235, 245, 255), -1)
        cv2.putText(img, arah, (X(a) + 4, T + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 220), 1)
    for k in range(0, int(t1) + 1, 10):
        x = X(k)
        cv2.line(img, (x, H - B), (x, H - B + (8 if k % 60 == 0 else 4)), (0, 0, 0), 1)
        if k % 20 == 0:
            cv2.putText(img, _fmt(k)[:5], (x - 18, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for y in (0, 0.25, 0.5, 0.75, 1.0):
        cv2.line(img, (L, Y(y)), (W - 10, Y(y)), (220, 220, 220), 1)
        cv2.putText(img, f"{y:.2f}", (8, Y(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    prev = None
    for t, y, v in series:
        ok = v >= 0.5
        p = (X(t), Y(y))
        if prev is not None and ok and prev[2]:
            cv2.line(img, prev[:2], p, (0, 0, 200), 2)
        cv2.circle(img, p, 3, (0, 0, 200) if ok else (180, 180, 180), -1)
        prev = (p[0], p[1], ok)
    cv2.putText(img, "kuning stage (merah = stage terlihat >= 50%, abu = tertutup)  naik = stage lebih kosong, turun = lebih penuh",
                (L, H - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(base + ".png", img)
    print("  grafik stage:", base + ".png / .csv")


def _stage_sheet(sheet, path, cols=4):
    if not sheet:
        return
    tiles = []
    for t, crop, valid in sheet:
        img = crop.copy()
        img[~valid] = (img[~valid] * 0.35).astype(np.uint8)
        cv2.rectangle(img, (0, 0), (62, 20), (0, 0, 0), -1)
        cv2.putText(img, _fmt(t)[:5], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    h, w = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.full((h, w, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))
    print("  lembar kontak stage:", path)


def main():
    ap = argparse.ArgumentParser(description="Aktivitas di sekitar kendaraan tertutup (mobil boks).")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--zones", default="cam1_zones.json")
    ap.add_argument("--model", default="yolo11n_openvino_model")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--every", type=float, default=1.0, help="detik antar frame yang dianalisis")
    ap.add_argument("--start", type=float, default=0.0, help="mulai dari detik ke-")
    ap.add_argument("--end", type=float, default=0.0, help="berhenti di detik ke- (0 = sampai habis)")
    ap.add_argument("--conf", type=float, default=0.30, help="conf orang")
    ap.add_argument("--veh-conf", type=float, default=0.30)
    ap.add_argument("--min-ratio", type=float, default=0.012)
    ap.add_argument("--min-active", type=float, default=8.0)
    ap.add_argument("--end-gap", type=float, default=15.0)
    ap.add_argument("--expand", type=float, default=0.20, help="perlebar kotak kendaraan (pecahan) untuk 'orang dekat'")
    ap.add_argument("--person-overlap", type=float, default=0.30, help="bagian kotak orang yang harus masuk kotak kendaraan yang diperlebar")
    ap.add_argument("--park-move", type=float, default=0.06, help="gerak maksimum (x ukuran kendaraan) dalam 4 dtk agar dianggap parkir")
    ap.add_argument("--no-parked", action="store_true", help="tidak mensyaratkan kendaraan parkir")
    ap.add_argument("--no-stage", action="store_true", help="abaikan poligon stage (hanya 'orang dekat kendaraan')")
    ap.add_argument("--min-people", type=int, default=1, help="buang aktivitas yang orang_maks-nya kurang dari ini (default 1 = tidak membuang apa pun)")
    ap.add_argument("--stage-delta", type=float, default=0.12, help="perubahan kuning stage minimum (0-1) agar arah diputuskan")
    ap.add_argument("--stage-visible", type=float, default=0.25, help="bagian stage minimum yang harus terlihat di awal DAN akhir")
    ap.add_argument("--no-virtual", action="store_true", help="matikan 'kendaraan virtual' (benda besar menutupi stage yang tidak terbaca YOLO)")
    ap.add_argument("--virt-min", type=float, default=20.0, help="detik stage tertutup benda besar sebelum dianggap kendaraan")
    ap.add_argument("--virt-ratio", type=float, default=0.5, help="kuning stage < rasio x kuning normal = tertutup")
    ap.add_argument("--stage-log", action="store_true", help="tulis grafik+CSV kuning stage terhadap waktu (<video>_stage_grafik.png / .csv)")
    ap.add_argument("--csv", default="hasil_boks.csv")
    ap.add_argument("--snap-dir", default="", help="simpan gambar beranotasi (mulai & berkala)")
    ap.add_argument("--snap-every", type=float, default=10.0)
    ap.add_argument("--verbose", action="store_true", help="satu baris per frame")
    args = ap.parse_args()

    from extract_crops import YoloDet, Zones, collect_videos
    zones = Zones(args.zones)
    det = YoloDet(args.model, min(args.conf, args.veh_conf), 0.45, [0, 2, 7], args.imgsz)
    vids = collect_videos(args.inputs)
    if not vids:
        print("Tidak ada video.")
        return 1
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "kendaraan_id", "mulai_dtk", "selesai_dtk", "durasi_dtk", "mulai_mmss",
                    "hit_orang_dekat", "hit_stage", "orang_maks", "x", "y",
                    "arah", "stage_kuning_awal", "stage_kuning_akhir", "stage_selisih", "stage_terlihat", "alasan_arah"])
        for i, v in enumerate(vids, 1):
            print(f"[{i}/{len(vids)}] {v}", flush=True)
            run_video(v, det, zones, args, w)
            fh.flush()
    print("Daftar aktivitas tersimpan di", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())