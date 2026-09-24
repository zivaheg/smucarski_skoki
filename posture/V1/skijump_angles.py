#!/usr/bin/env python3
"""
skijump_angles.py - measure the body-to-ski angle of a ski jumper in flight.

Input : a local video file or a URL (YouTube etc., downloaded with yt-dlp)
Output: JSON with one data point every --step seconds (default 0.1 s) while the
        jumper is airborne:  time, angle between body line and skis, + diagnostics.

Pipeline (per sampled frame)
  1. shot-cut detection (HSV histogram distance)        -> resets tracking
  2. person localisation + 2D pose (MediaPipe PoseLandmarker, heavy model),
     run on a tracked crop first, then full frame, then detector-guided crops
  3. ski detection: line segments (FastLineDetector / LSD) in a region around
     the feet, with the jumper's own limbs masked out; segments must be
     collinear with a boot (heel / ankle / toe) -> dominant orientation per ski
  4. body line  = mid-ankle -> mid-shoulder   (also mid-ankle -> mid-hip)
     angle      = angle between body line and ski axis in the image plane
  5. view check (side / oblique / frontal).  In frontal views the 2-D angle is
     meaningless, so it is reported as null.
  6. airborne classification (legs extended + skis at the feet + small
     body-ski angle), temporal clean-up, split into flight segments.

The angle is a 2-D (image-plane) measurement: it is exact for a pure side
view and increasingly distorted as the camera moves towards the front/back.

Usage
  python skijump_angles.py https://www.youtube.com/watch?v=Wij38rVUTEY -o out.json
  python skijump_angles.py jump.mp4 -o out.json --debug-video dbg.mp4 --plot plot.png
  python skijump_angles.py jump.mp4 --start 60 --end 90 --all-frames

Requirements:  pip install opencv-contrib-python mediapipe numpy yt-dlp
               (matplotlib only for --plot)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
MODEL_URLS = {
    "pose_heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "pose_full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "detector": "https://storage.googleapis.com/mediapipe-models/object_detector/"
                "efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
}


def ensure_model(name: str, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    url = MODEL_URLS[name]
    p = model_dir / url.rsplit("/", 1)[-1]
    if not p.exists():
        print(f"[models] downloading {p.name} ...", file=sys.stderr)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(p)
    return p


# --------------------------------------------------------------------------- #
# video input
# --------------------------------------------------------------------------- #
def resolve_video(src: str, cache_dir: Path, max_height: int = 720) -> Path:
    """Return a local video path; download with yt-dlp if src is a URL."""
    if os.path.exists(src):
        return Path(src)
    if not src.lower().startswith(("http://", "https://")):
        raise FileNotFoundError(src)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp  # noqa: F401
        use_module = True
    except ImportError:
        use_module = False
        if not shutil.which("yt-dlp"):
            raise RuntimeError("URL input needs yt-dlp:  pip install yt-dlp")
    # prefer H.264 (avc1): OpenCV cannot decode AV1/VP9 on many installs
    fmt = (f"bv*[vcodec^=avc1][height<={max_height}]/"
           f"b[vcodec^=avc1][height<={max_height}]/"
           f"bv*[height<={max_height}]/b[height<={max_height}]/b")
    outtmpl = str(cache_dir / "%(id)s.%(ext)s")
    if use_module:
        import yt_dlp
        opts = {"format": fmt, "outtmpl": outtmpl, "merge_output_format": "mp4",
                "retries": 10, "fragment_retries": 10, "quiet": True, "noprogress": True}
        last = None
        for _ in range(4):                      # YouTube sometimes 403s mid-way; resume
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(src, download=True)
                    path = Path(ydl.prepare_filename(info))
                break
            except Exception as e:              # pragma: no cover
                last = e
        else:
            raise RuntimeError(f"download failed: {last}")
        if not path.exists():
            cands = sorted(cache_dir.glob(f"{info['id']}.*"))
            cands = [c for c in cands if c.suffix != ".part"]
            path = cands[0]
        return path
    subprocess.run(["yt-dlp", "-f", fmt, "--merge-output-format", "mp4",
                    "-o", outtmpl, src], check=True)
    cands = [c for c in sorted(cache_dir.iterdir(), key=os.path.getmtime)
             if c.suffix in (".mp4", ".mkv", ".webm")]
    return cands[-1]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def angle_at(a, b, c) -> float:
    """interior angle ABC in degrees"""
    v1, v2 = np.asarray(a) - b, np.asarray(c) - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / n, -1, 1)))


def signed_angle_to_ski(body_vec, ski_dir) -> float:
    """
    Angle (deg) between body vector and ski axis.  ski_dir is an axis (sign
    free); it is oriented to point the same way as the body.  Positive when the
    body is rotated from the ski towards image-'up' (= body above the skis, the
    normal flight position).
    """
    u = np.asarray(ski_dir, float)
    u /= np.linalg.norm(u)
    b = np.asarray(body_vec, float)
    if np.dot(u, b) < 0:
        u = -u
    cross = u[0] * b[1] - u[1] * b[0]
    ang = math.degrees(math.atan2(abs(cross), np.dot(u, b)))
    n = np.array([-u[1], u[0]])                # a normal of the ski
    if n[1] > 0:                               # make it point up (image y down)
        n = -n
    return ang if np.dot(b, n) >= 0 else -ang


def point_line_dist(p, p0, d):
    """distance from point p to infinite line through p0 with unit dir d"""
    v = np.asarray(p) - p0
    return abs(v[0] * d[1] - v[1] * d[0])


# --------------------------------------------------------------------------- #
# pose
# --------------------------------------------------------------------------- #
LM = dict(nose=0, l_sh=11, r_sh=12, l_el=13, r_el=14, l_wr=15, r_wr=16,
          l_hip=23, r_hip=24, l_kn=25, r_kn=26, l_an=27, r_an=28,
          l_heel=29, r_heel=30, l_toe=31, r_toe=32)
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
            (27, 31), (28, 30), (30, 32), (28, 32)]


class PoseEstimator:
    def __init__(self, model_dir: Path, heavy=True, use_detector=True):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.mp = mp
        mpath = ensure_model("pose_heavy" if heavy else "pose_full", model_dir)
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(mpath)),
                running_mode=vision.RunningMode.IMAGE, num_poses=1,
                min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4))
        self.det = None
        if use_detector:
            dpath = ensure_model("detector", model_dir)
            self.det = vision.ObjectDetector.create_from_options(
                vision.ObjectDetectorOptions(
                    base_options=mpt.BaseOptions(model_asset_path=str(dpath)),
                    running_mode=vision.RunningMode.IMAGE, max_results=5,
                    score_threshold=0.25, category_allowlist=["person"]))

    def close(self):
        for t in (self.pose, self.det):
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass

    def _run(self, rgb):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        r = self.pose.detect(img)
        if not r.pose_landmarks:
            return None
        h, w = rgb.shape[:2]
        lms = r.pose_landmarks[0]
        pts = np.array([[l.x * w, l.y * h] for l in lms], np.float32)
        vis = np.array([min(l.visibility, l.presence) for l in lms], np.float32)
        return pts, vis

    def _run_crop(self, rgb, box, min_side=384):
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = max(x1 - x0, y1 - y0) * 1.5
        s = max(s, 96)
        x0, x1 = int(max(0, cx - s / 2)), int(min(W, cx + s / 2))
        y0, y1 = int(max(0, cy - s / 2)), int(min(H, cy + s / 2))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        crop = rgb[y0:y1, x0:x1]
        sc = max(1.0, min_side / max(crop.shape[:2]))
        if sc > 1:
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        r = self._run(crop)
        if r is None:
            return None
        pts, vis = r
        pts = pts / sc + np.array([x0, y0], np.float32)
        return pts, vis

    def detect_people(self, rgb):
        if self.det is None:
            return []
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        out = []
        for d in self.det.detect(img).detections:
            b = d.bounding_box
            out.append((d.categories[0].score,
                        (b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height)))
        return sorted(out, reverse=True)

    def estimate(self, rgb, prev_box=None):
        """returns (pts, vis, box, method) or None"""
        tries = []
        if prev_box is not None:
            tries.append(("track", lambda: self._run_crop(rgb, prev_box)))
        tries.append(("full", lambda: self._run(rgb)))
        best = None
        for name, fn in tries:
            r = fn()
            if r is not None and _pose_quality(r[1]) > 0.5:
                best = (*r, name)
                break
        if best is None:
            for score, box in self.detect_people(rgb)[:3]:
                r = self._run_crop(rgb, box)
                if r is not None and _pose_quality(r[1]) > 0.5:
                    best = (*r, "detector")
                    break
        if best is None:
            return None
        pts, vis, name = best
        good = vis > 0.3
        if good.sum() < 6:
            return None
        p = pts[good]
        box = (float(p[:, 0].min()), float(p[:, 1].min()),
               float(p[:, 0].max()), float(p[:, 1].max()))
        return pts, vis, box, name


def _pose_quality(vis):
    idx = [11, 12, 23, 24, 25, 26, 27, 28]
    return float(np.mean(vis[idx]))


# --------------------------------------------------------------------------- #
# ski detection
# --------------------------------------------------------------------------- #
class SkiDetector:
    def __init__(self):
        self.fld = None
        if hasattr(cv2, "ximgproc"):
            try:
                self.fld = cv2.ximgproc.createFastLineDetector(
                    length_threshold=10, distance_threshold=1.414,
                    canny_th1=30, canny_th2=80, canny_aperture_size=3, do_merge=True)
            except Exception:
                self.fld = None
        self.lsd = cv2.createLineSegmentDetector() if self.fld is None and \
            hasattr(cv2, "createLineSegmentDetector") else None

    def segments(self, gray):
        if self.fld is not None:
            s = self.fld.detect(gray)
        elif self.lsd is not None:
            s = self.lsd.detect(gray)[0]
        else:
            e = cv2.Canny(gray, 40, 120)
            s = cv2.HoughLinesP(e, 1, np.pi / 360, 25, minLineLength=12, maxLineGap=4)
        if s is None:
            return np.zeros((0, 4), np.float32)
        return s.reshape(-1, 4).astype(np.float32)

    def detect(self, frame_bgr, pts, vis, prior_rel=None):
        """
        Find the ski axes near the feet.

        1. line segments in a window around the feet (jumper's body masked out)
        2. group collinear segments into candidate lines (a ski edge is often
           broken up by logos / blur)
        3. keep lines whose extent looks like a ski: 0.5L..2.4L long, passing
           under a boot and extending past it towards the tip (+body direction)
        4. score = visible length x closeness to the boot x parallel-edge bonus
        prior_rel: ski-vs-body angle (deg) from the previous sample of the same
        shot; candidates close to it are preferred (temporal consistency).
        Returns dict (full-frame pixel coords) or None.
        """
        an = (pts[27] + pts[28]) / 2
        sh = (pts[11] + pts[12]) / 2
        L = float(np.linalg.norm(sh - an))
        if L < 25:
            return None
        H, W = frame_bgr.shape[:2]
        R = 1.7 * L
        x0, y0 = int(max(0, an[0] - R)), int(max(0, an[1] - R))
        x1, y1 = int(min(W, an[0] + R)), int(min(H, an[1] + R))
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return None
        sc = 1.0
        if max(roi.shape[:2]) < 480:          # upscale small ROIs so thin skis survive
            sc = 480 / max(roi.shape[:2])
            roi = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        segs = self.segments(gray) / sc + np.array([x0, y0, x0, y0], np.float32)

        # ---- mask of the jumper's own body -------------------------------------
        mask = np.zeros((H, W), np.uint8)
        thick = max(3, int(0.12 * L))
        for a, b in [(23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (12, 24),
                     (11, 12), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16), (0, 11), (0, 12)]:
            if vis[a] > 0.2 and vis[b] > 0.2:
                cv2.line(mask, tuple(map(int, pts[a])), tuple(map(int, pts[b])), 255, thick)
        cv2.fillConvexPoly(mask, np.array([pts[11], pts[12], pts[24], pts[23]], np.int32), 255)

        body_dir = (sh - an) / L
        P, Q, D, LEN = [], [], [], []
        for xa, ya, xb, yb in segs:
            p, q = np.array([xa, ya], float), np.array([xb, yb], float)
            ln = float(np.linalg.norm(q - p))
            if ln < 0.06 * L:
                continue
            d = (q - p) / ln
            if abs(float(np.dot(d, body_dir))) < math.cos(math.radians(80)):
                continue                      # skis are never ~perpendicular to the body in flight
            n = max(4, int(ln / 3))
            ts = np.linspace(0, 1, n)
            xs = np.clip((p[0] + ts * (q[0] - p[0])).astype(int), 0, W - 1)
            ys = np.clip((p[1] + ts * (q[1] - p[1])).astype(int), 0, H - 1)
            if (mask[ys, xs] > 0).mean() > 0.5:
                continue
            if np.dot(d, body_dir) < 0:       # orient every segment tail->tip
                p, q, d = q, p, -d
            P.append(p); Q.append(q); D.append(d); LEN.append(ln)
        if not P:
            return None
        P, Q, D, LEN = map(np.array, (P, Q, D, LEN))

        # ---- group collinear segments ----------------------------------------------
        order = np.argsort(-LEN)
        used = np.zeros(len(P), bool)
        tol = max(2.5, 0.035 * L)
        lines = []
        for i in order:
            if used[i]:
                continue
            d0, p0 = D[i], P[i]
            nrm = np.array([-d0[1], d0[0]])
            cos_ok = D @ d0 > math.cos(math.radians(5))
            off_p = np.abs((P - p0) @ nrm)
            off_q = np.abs((Q - p0) @ nrm)
            members = np.where(cos_ok & (off_p < tol) & (off_q < tol) & ~used)[0]
            used[members] = True
            # weighted refit through all endpoints
            pts_m = np.vstack([P[members], Q[members]])
            w = np.concatenate([LEN[members], LEN[members]])
            c = np.average(pts_m, axis=0, weights=w)
            cov = np.cov((pts_m - c).T, aweights=w) if len(pts_m) > 2 else None
            if cov is not None and np.all(np.isfinite(cov)):
                ev, evec = np.linalg.eigh(cov)
                d = evec[:, -1]
            else:
                d = d0
            if np.dot(d, body_dir) < 0:
                d = -d
            s_int = sorted(zip(((P[members] - c) @ d).tolist(), ((Q[members] - c) @ d).tolist()))
            cover, cur_a, cur_b = 0.0, None, None      # union length along the line
            for a, b in s_int:
                a, b = min(a, b), max(a, b)
                if cur_b is None or a > cur_b:
                    if cur_b is not None:
                        cover += cur_b - cur_a
                    cur_a, cur_b = a, b
                else:
                    cur_b = max(cur_b, b)
            cover += cur_b - cur_a
            smin = min(min(a, b) for a, b in s_int)
            smax = max(max(a, b) for a, b in s_int)
            lines.append(dict(c=c, d=d, cover=cover, smin=smin, smax=smax,
                              segs=[(P[k].tolist(), Q[k].tolist()) for k in members]))

        # parallel-edge bonus: a ski is a thin band -> two close parallel lines
        for ln_ in lines:
            nrm = np.array([-ln_["d"][1], ln_["d"][0]])
            ln_["pair"] = any(
                o is not ln_ and np.dot(o["d"], ln_["d"]) > math.cos(math.radians(4))
                and 0.01 * L < abs(np.dot(o["c"] - ln_["c"], nrm)) < 0.10 * L
                for o in lines)

        feet = {"left": [pts[27], pts[29], pts[31]], "right": [pts[28], pts[30], pts[32]]}
        best = {}
        for foot, fp in feet.items():
            fc = np.mean(fp, axis=0)
            cand = []
            for k, ln_ in enumerate(lines):
                ext = ln_["smax"] - ln_["smin"]
                if ln_["cover"] < 0.35 * L or ext > 2.5 * L:
                    continue
                d, c = ln_["d"], ln_["c"]
                dist = min(point_line_dist(f, c, d) for f in fp)
                if dist > 0.35 * L:
                    continue
                s_f = float(np.dot(fc - c, d))       # boot position along the line
                if s_f < ln_["smin"] - 0.4 * L or s_f > ln_["smax"] + 0.2 * L:
                    continue                          # line does not reach the boot
                tip = ln_["smax"] - s_f               # length in front of the boot
                tail = s_f - ln_["smin"]              # length behind the boot
                if tip < 0.2 * L:
                    continue
                score = min(ln_["cover"], 2.0 * L) * math.exp(-dist / (0.12 * L)) \
                    * (1.3 if ln_["pair"] else 1.0) * (1.15 if tail > 0.15 * L else 1.0)
                if prior_rel is not None:
                    rel = signed_angle_to_ski(sh - an, d)
                    score *= 0.35 + 0.65 * math.exp(-((rel - prior_rel) / 12.0) ** 2)
                cand.append((score, k, dist, tip, tail))
            if cand:
                score, k, dist, tip, tail = max(cand)
                best[foot] = dict(line=k, score=score / L, dist=dist / L,
                                  tip=tip / L, tail=tail / L)
        if not best:
            return None

        out = {"L": L, "skis": {}}
        for foot, b in best.items():
            ln_ = lines[b["line"]]
            th = math.degrees(math.atan2(ln_["d"][1], ln_["d"][0]))
            out["skis"][foot] = dict(theta=th, support=ln_["cover"] / L, score=b["score"],
                                     dist=b["dist"], tip=b["tip"], tail=b["tail"],
                                     center=ln_["c"].tolist(), segments=ln_["segs"])
        sk = out["skis"]
        if len(sk) == 2 and best["left"]["line"] != best["right"]["line"]:
            a, b = sk["left"]["theta"], sk["right"]["theta"]
            dv = (a - b + 180) % 360 - 180
            out["v_angle"] = abs(dv)
            # both skis found: use the bisector (the V is symmetric about the flight axis)
            theta = b + dv / 2
            support = sk["left"]["support"] + sk["right"]["support"]
            out["n_skis"] = 2
        else:
            k = max(sk, key=lambda f: sk[f]["score"])
            theta, support = sk[k]["theta"], sk[k]["support"]
            out["v_angle"] = 0.0 if len(sk) == 2 else None
            out["n_skis"] = 1
        out["theta"] = theta
        out["support"] = support
        out["tail"] = max(s["tail"] for s in sk.values())
        out["tip"] = max(s["tip"] for s in sk.values())
        out["dir"] = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
        return out


# --------------------------------------------------------------------------- #
# per-frame analysis
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    t: float
    frame: int
    shot: int
    pose_ok: bool = False
    ski_ok: bool = False
    angle: float | None = None           # body line (ankle->shoulder) vs ski
    angle_leg: float | None = None       # ankle->hip vs ski
    angle_torso: float | None = None     # hip->shoulder vs ski
    n_skis: int | None = None
    leg_vis: float | None = None
    strong: bool = False
    knee: float | None = None
    hip: float | None = None
    body_incl: float | None = None       # body line vs image horizontal
    ski_incl: float | None = None
    ski_support: float | None = None
    v_angle: float | None = None
    view: str | None = None
    width_ratio: float | None = None
    pose_q: float | None = None
    airborne_raw: bool = False
    airborne: bool = False
    segment: int | None = None
    confidence: float = 0.0
    method: str | None = None
    _pts: object = field(default=None, repr=False)
    _ski: object = field(default=None, repr=False)


def classify_view(pts, L):
    sh_w = float(np.linalg.norm(pts[11] - pts[12]))
    hip_w = float(np.linalg.norm(pts[23] - pts[24]))
    torso = float(np.linalg.norm((pts[11] + pts[12]) / 2 - (pts[23] + pts[24]) / 2))
    r = (sh_w + hip_w) / 2 / max(torso, 1e-3)
    if r < 0.45:
        return "side", r
    if r < 0.8:
        return "oblique", r
    return "frontal", r


def analyse_frame(frame, t, fidx, shot, pose_est, ski_det, prev_box, prior_rel=None):
    s = Sample(t=round(t, 3), frame=fidx, shot=shot)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = pose_est.estimate(rgb, prev_box)
    if r is None:
        return s, None
    pts, vis, box, method = r
    s.pose_ok, s.method, s._pts = True, method, pts
    s.pose_q = round(_pose_quality(vis), 3)
    s.leg_vis = round(float(max(vis[27], vis[28])), 3)
    an = (pts[27] + pts[28]) / 2
    sh = (pts[11] + pts[12]) / 2
    hp = (pts[23] + pts[24]) / 2
    kn = [angle_at(pts[23], pts[25], pts[27]), angle_at(pts[24], pts[26], pts[28])]
    s.knee = round(float(np.nanmean(kn)), 1)
    s.hip = round(float(np.nanmean([angle_at(pts[11], pts[23], pts[25]),
                                    angle_at(pts[12], pts[24], pts[26])])), 1)
    body = sh - an
    L = float(np.linalg.norm(body))
    s.body_incl = round(math.degrees(math.atan2(-body[1], abs(body[0]) + 1e-9)), 1)
    s.view, wr = classify_view(pts, L)
    s.width_ratio = round(wr, 3)

    ski = ski_det.detect(frame, pts, vis, prior_rel)
    if ski is not None:
        s.ski_ok, s._ski = True, ski
        s.ski_support = round(ski["support"], 3)
        s.n_skis = ski["n_skis"]
        s.v_angle = None if ski["v_angle"] is None else round(ski["v_angle"], 1)
        d = ski["dir"]
        s.ski_incl = round(math.degrees(math.atan2(-d[1] if d[0] >= 0 else d[1], abs(d[0]))), 1)
        s.angle = round(signed_angle_to_ski(body, d), 2)
        s.angle_leg = round(signed_angle_to_ski(hp - an, d), 2)
        s.angle_torso = round(signed_angle_to_ski(sh - hp, d), 2)
        v = s.v_angle or 0.0
        if s.n_skis == 2 and v >= 22 or wr >= 0.8:
            s.view = "frontal"                  # wide V / wide body: front or rear view
        elif v >= 12 or wr >= 0.45:
            s.view = "oblique"
        else:
            s.view = "side"
        conf = min(1.0, ski["support"] / 1.0) * min(1.0, s.pose_q / 0.8)
        conf *= {"side": 1.0, "oblique": 0.6, "frontal": 0.1}[s.view]
        s.confidence = round(conf, 3)
        extended = s.knee > 140 and s.hip > 115
        big_enough = L >= 40
        if s.view == "frontal":
            s.airborne_raw = bool(extended and big_enough and s.n_skis == 2 and v >= 20)
        else:
            s.airborne_raw = bool(extended and big_enough and ski["tip"] > 0.3
                                  and -15 < s.angle < 60 and abs(s.body_incl) < 65)
        # a "strong" sample: legs clearly visible and a long ski found
        s.strong = bool(s.airborne_raw and s.leg_vis >= 0.5 and
                        (ski["support"] >= 1.2 or (s.view == "frontal" and v >= 20)))
    return s, box


# --------------------------------------------------------------------------- #
# temporal post-processing
# --------------------------------------------------------------------------- #
def postprocess(samples, step, min_len=0.6, max_gap=0.35, merge_gap=1.2):
    n = len(samples)
    raw = np.array([s.airborne_raw for s in samples])
    shots = np.array([s.shot for s in samples])
    # close small gaps and drop short blips, within shots
    gap = int(round(max_gap / step))
    air = raw.copy()
    i = 0
    while i < n:
        if not air[i]:
            j = i
            while j < n and not air[j] and shots[j] == shots[i]:
                j += 1
            if 0 < i and j < n and j - i <= gap and air[i - 1] and air[j] \
                    and shots[i - 1] == shots[j]:
                air[i:j] = True
            i = max(j, i + 1)
        else:
            i += 1
    seg_id, segments = 0, []
    i = 0
    minn = int(round(min_len / step))
    while i < n:
        if air[i]:
            j = i
            while j < n and air[j] and shots[j] == shots[i]:
                j += 1
            if j - i >= minn:
                segments.append((i, j))
            else:
                air[i:j] = False
            i = j
        else:
            i += 1
    # drop segments without enough clear evidence (pose hallucinations in
    # crowd shots, logo transitions ...)
    kept = []
    for a, b in segments:
        n_strong = sum(samples[i].strong for i in range(a, b))
        if n_strong >= max(3, 0.3 * (b - a)):
            kept.append((a, b))
        else:
            air[a:b] = False
    segments = kept
    # merge consecutive segments separated only by a cut (same jump, new camera)
    merged = []
    mgap = int(round(merge_gap / step))
    for a, b in segments:
        if merged and a - merged[-1][1] <= mgap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    for k, (a, b) in enumerate(merged):
        for i in range(a, b):
            samples[i].airborne = bool(air[i]) or True
            samples[i].segment = k
    # outlier rejection + light smoothing, inside each segment and shot
    for a, b in merged:
        idx = [i for i in range(a, b)]
        for i in idx:
            s = samples[i]
            s.__dict__["angle_raw"] = s.angle
        def usable(j):
            return samples[j].__dict__.get("angle_raw") is not None and samples[j].view != "frontal"
        for i in idx:
            s = samples[i]
            if not usable(i):
                continue
            win = [samples[j].__dict__["angle_raw"] for j in range(max(a, i - 3), min(b, i + 4))
                   if j != i and usable(j) and samples[j].shot == s.shot]
            # isolated measurements (no support from neighbours) or large jumps are rejected
            if len(win) < 2 or abs(s.angle - float(np.median(win))) > 15:
                s.__dict__["outlier"] = True
                s.angle = None
        for i in idx:
            s = samples[i]
            win = [samples[j].angle for j in range(max(a, i - 1), min(b, i + 2))
                   if samples[j].angle is not None and samples[j].view != "frontal"
                   and samples[j].shot == s.shot]
            s.__dict__["angle_smooth"] = round(float(np.median(win)), 2) \
                if win and s.angle is not None else None
    return merged


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def draw(frame, s: Sample):
    img = frame.copy()
    if s._pts is not None:
        p = s._pts
        for a, b in SKELETON:
            cv2.line(img, tuple(map(int, p[a])), tuple(map(int, p[b])), (0, 255, 255), 2)
        an = (p[27] + p[28]) / 2
        sh = (p[11] + p[12]) / 2
        cv2.line(img, tuple(map(int, an)), tuple(map(int, sh)), (0, 0, 255), 3)
        if s._ski is not None:
            for foot, sk in s._ski["skis"].items():
                for p0, p1 in sk["segments"]:
                    cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), (255, 120, 0), 3)
            d = s._ski["dir"]
            L = s._ski["L"]
            a0, a1 = an - d * L, an + d * L
            cv2.line(img, tuple(map(int, a0)), tuple(map(int, a1)), (0, 255, 0), 2)
    col = (0, 200, 0) if s.airborne else (0, 0, 220)
    txt = f"t={s.t:6.2f}s shot {s.shot} {'AIR' if s.airborne else '---'} "
    if s.angle is not None:
        txt += f"angle={s.angle:5.1f} "
    txt += f"knee={s.knee} view={s.view}"
    cv2.rectangle(img, (0, img.shape[0] - 34), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return img


# --------------------------------------------------------------------------- #
# frame access (OpenCV, with an ffmpeg fallback for codecs OpenCV cannot decode)
# --------------------------------------------------------------------------- #
def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    meta = dict(fps=fps, frames=nfr, width=w, height=h, reader="opencv" if ok else "ffmpeg")
    if not ok or fps <= 0 or w == 0:
        if not shutil.which("ffprobe"):
            raise RuntimeError("OpenCV cannot decode this video and ffprobe/ffmpeg "
                               "is not installed")
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height,r_frame_rate,nb_frames"
                              ":format=duration", "-of", "json", str(path)],
                             capture_output=True, text=True, check=True).stdout
        j = json.loads(out)
        st = j["streams"][0]
        num, den = st["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        dur = float(j["format"]["duration"])
        meta.update(fps=fps, width=int(st["width"]), height=int(st["height"]),
                    frames=int(st.get("nb_frames") or round(dur * fps)), reader="ffmpeg")
    meta["duration"] = meta["frames"] / meta["fps"]
    return meta


def iter_frames(path: Path, start: float, end: float, step: float, meta: dict):
    """yield (t, frame_index, bgr_frame) for t = start, start+step, ... < end"""
    fps = meta["fps"]
    times = np.arange(start, end - 1e-9, step)
    if len(times) == 0:
        return
    targets = [int(round(t * fps)) for t in times]
    if meta["reader"] == "opencv":
        cap = cv2.VideoCapture(str(path))
        fidx = targets[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ti = 0
        while ti < len(targets):
            if not cap.grab():
                break
            if fidx < targets[ti]:
                fidx += 1
                continue
            ok, frame = cap.retrieve()
            fidx += 1
            if not ok:
                break
            yield float(times[ti]), targets[ti], frame
            ti += 1
            while ti < len(targets) and targets[ti] < fidx:
                ti += 1
        cap.release()
        return
    # ffmpeg pipe: decode every frame from start, keep the ones we need
    w, h = meta["width"], meta["height"]
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path),
           "-t", f"{end - start + 1.0 / fps:.3f}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    fsize = w * h * 3
    fidx = int(round(start * fps))
    ti = 0
    try:
        while ti < len(targets):
            buf = proc.stdout.read(fsize)
            if len(buf) < fsize:
                break
            if fidx >= targets[ti]:
                frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
                yield float(times[ti]), targets[ti], frame
                ti += 1
                while ti < len(targets) and targets[ti] <= fidx:
                    ti += 1
            fidx += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def analyse_video(path: Path, step=0.1, start=0.0, end=None, model_dir=Path("models"),
                  heavy=True, cut_thresh=0.5, debug_video=None, debug_scale=0.5,
                  progress=True):
    meta = probe_video(path)
    fps, nfr, dur = meta["fps"], meta["frames"], meta["duration"]
    end = dur if end is None else min(end, dur)

    pose_est = PoseEstimator(model_dir, heavy=heavy)
    ski_det = SkiDetector()
    writer = None

    samples, prev_hist, prev_tiny, shot, prev_box, prior = [], None, None, 0, None, None
    for t, fidx, frame in iter_frames(path, start, end, step, meta):
        # shot-cut detection
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        tiny = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_hist is not None:
            dh = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            dp = float(np.mean(np.abs(tiny - prev_tiny)))
            if (dh > cut_thresh and dp > 40) or dh > 0.65 or dp > 75:
                shot += 1
                prev_box, prior = None, None
        prev_hist, prev_tiny = hist, tiny

        s, box = analyse_frame(frame, float(t), fidx, shot, pose_est, ski_det, prev_box, prior)
        prev_box = box
        prior = s.angle if (s.angle is not None and s.view != "frontal") else None
        samples.append((s, frame if debug_video else None))
        if progress and len(samples) % 50 == 0:
            print(f"\r[analyse] t={t:7.2f}s / {end:.1f}s", end="", file=sys.stderr)
    pose_est.close()
    if progress:
        print(file=sys.stderr)
    if not samples:
        raise RuntimeError("no frames could be decoded")

    S = [s for s, _ in samples]
    segments = postprocess(S, step)

    if debug_video:
        for s, fr in samples:
            img = draw(fr, s)
            if debug_scale != 1:
                img = cv2.resize(img, None, fx=debug_scale, fy=debug_scale)
            if writer is None:
                writer = cv2.VideoWriter(str(debug_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                         1.0 / step, (img.shape[1], img.shape[0]))
            writer.write(img)
        if writer:
            writer.release()
    return S, segments, meta


def build_json(src, path, S, segments, meta, step, all_frames=False):
    def rec(s):
        d = {k: v for k, v in asdict(s).items() if not k.startswith("_")}
        d["angle_smooth"] = s.__dict__.get("angle_smooth")
        return d

    segs = []
    for k, (a, b) in enumerate(segments):
        pts = [S[i] for i in range(a, b)]
        good = [p.angle for p in pts if p.angle is not None and p.view != "frontal"]
        segs.append(dict(
            id=k, start=pts[0].t, end=round(pts[-1].t + step, 3),
            duration=round(pts[-1].t + step - pts[0].t, 3),
            shots=sorted({p.shot for p in pts}),
            n_points=len(pts), n_measured=len(good),
            angle_mean=round(float(np.mean(good)), 2) if good else None,
            angle_min=round(float(np.min(good)), 2) if good else None,
            angle_max=round(float(np.max(good)), 2) if good else None))

    data = []
    for s in S:
        if not (s.airborne or all_frames):
            continue
        reliable = s.angle is not None and s.view != "frontal"
        d = dict(time=s.t,
                 angle=s.angle if reliable else None,
                 angle_raw=s.__dict__.get("angle_raw", s.angle),
                 outlier=bool(s.__dict__.get("outlier", False)),
                 angle_smooth=s.__dict__.get("angle_smooth"),
                 segment=s.segment, airborne=s.airborne)
        if True:
            d.update(angle_leg=s.angle_leg if reliable else None,
                     angle_torso=s.angle_torso if reliable else None, n_skis=s.n_skis,
                     body_inclination=s.body_incl, ski_inclination=s.ski_incl,
                     knee_angle=s.knee, hip_angle=s.hip, view=s.view,
                     ski_v_angle_2d=s.v_angle, confidence=s.confidence,
                     shot=s.shot, frame=s.frame)
        data.append(d)
    return {
        "source": src,
        "video_file": str(path),
        "video": {"fps": meta["fps"], "frames": meta["frames"],
                  "duration_s": round(meta["duration"], 3)},
        "sampling_step_s": step,
        "definitions": {
            "angle": "2-D angle (deg) between the body line (mid-ankle -> mid-shoulder) "
                     "and the ski axis. Positive = body above the skis. null when the "
                     "view is frontal/rear (angle not observable) or skis not found.",
            "angle_leg": "same, but body line = mid-ankle -> mid-hip",
            "angle_torso": "same, but body line = mid-hip -> mid-shoulder",
            "angle_raw": "angle before outlier rejection (always reported, even for "
                         "frontal views - use with care)",
            "outlier": "true if angle_raw deviated > 15 deg from the median of its "
                       "neighbours (+-0.3 s, same shot) or had < 2 such neighbours; "
                       "angle is then null",
            "angle_smooth": "median of 3 neighbouring samples in the same shot",
            "body_inclination": "body line vs image horizontal (deg, camera-dependent)",
            "ski_inclination": "ski axis vs image horizontal (deg, camera-dependent)",
            "knee_angle / hip_angle": "interior joint angles (deg), 180 = straight",
            "view": "side | oblique | frontal (estimated from body width / ski V)",
            "confidence": "0..1 heuristic (pose visibility x ski support x view)",
            "segment": "index of continuous airborne segment (a broadcast may show the "
                       "same jump several times: live + replays)",
        },
        "segments": segs,
        "data": data,
    }


def save_plot(out_json, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = out_json["data"]
    fig, ax = plt.subplots(figsize=(11, 4))
    for seg in out_json["segments"]:
        ax.axvspan(seg["start"], seg["end"], color="#dde8f6", lw=0)
    first = True
    for seg in out_json["segments"]:
        pts = [p for p in d if p["segment"] == seg["id"]]
        t = [p["time"] for p in pts if p["angle"] is not None]
        a = [p["angle"] for p in pts if p["angle"] is not None]
        ax.plot(t, a, ".", color="#2a6fdb", ms=5, label="angle" if first else None)
        ts = [p["time"] if p.get("angle_smooth") is not None else np.nan for p in pts]
        as_ = [p["angle_smooth"] if p.get("angle_smooth") is not None else np.nan for p in pts]
        ax.plot(ts, as_, "-", color="#1b3f7a", lw=1.2, alpha=.8,
                label="median of 3" if first else None)
        fr = [p["time"] for p in pts if p["view"] == "frontal"]
        if fr:
            ax.plot(fr, [0.03] * len(fr), "|", color="#c77", ms=6,
                    transform=ax.get_xaxis_transform(),
                    label="frontal view (no angle)" if first else None)
        first = False
    # zoom on the airborne part of the video
    if out_json["segments"]:
        t0 = out_json["segments"][0]["start"]
        t1 = out_json["segments"][-1]["end"]
        if t1 - t0 < 0.6 * out_json["video"]["duration_s"]:
            pad = max(1.0, 0.05 * (t1 - t0))
            ax.set_xlim(t0 - pad, t1 + pad)
    ax.set_xlabel("video time [s]")
    ax.set_ylabel("body-to-ski angle [deg]")
    ax.set_title("Ski jumper: body line vs skis (shaded = airborne segments)")
    ax.grid(alpha=.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video file or URL (YouTube etc.)")
    ap.add_argument("-o", "--output", default=None, help="output JSON (default <video>.angles.json)")
    ap.add_argument("--step", type=float, default=0.1, help="sampling step in seconds (0.1)")
    ap.add_argument("--start", type=float, default=0.0, help="analyse from this time [s]")
    ap.add_argument("--end", type=float, default=None, help="analyse until this time [s]")
    ap.add_argument("--all-frames", action="store_true",
                    help="write every sampled point, not only airborne ones")
    ap.add_argument("--fast", action="store_true", help="use the lighter pose model")
    ap.add_argument("--models", default=str(Path(__file__).with_name("models")))
    ap.add_argument("--cache", default=str(Path(__file__).with_name("video_cache")))
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--debug-video", default=None, help="write annotated mp4 (1 frame per step)")
    ap.add_argument("--plot", default=None, help="write PNG plot of angle over time")
    a = ap.parse_args()

    path = resolve_video(a.source, Path(a.cache), a.max_height)
    print(f"[video] {path}", file=sys.stderr)
    S, segs, meta = analyse_video(path, a.step, a.start, a.end, Path(a.models),
                                  heavy=not a.fast, debug_video=a.debug_video)
    out = build_json(a.source, path, S, segs, meta, a.step, a.all_frames)
    op = a.output or str(Path(path).with_suffix(".angles.json"))
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[done] {len(out['data'])} points, {len(segs)} airborne segments -> {op}",
          file=sys.stderr)
    for sg in out["segments"]:
        print(f"  segment {sg['id']}: {sg['start']:.1f}-{sg['end']:.1f}s  "
              f"measured {sg['n_measured']}/{sg['n_points']}  mean angle {sg['angle_mean']}",
              file=sys.stderr)
    if a.plot:
        save_plot(out, a.plot)


if __name__ == "__main__":
    main()
