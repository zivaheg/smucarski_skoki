#!/usr/bin/env python3
"""
skijump_angles.py - measure the body-to-ski angle of a ski jumper in flight.

Input : a local video file or a URL (YouTube etc., downloaded with yt-dlp)
Output: JSON with one data point every --step seconds (default 0.1 s) while the
        jumper is airborne:  time, angle between body line and skis, all 33 body
        keypoints (2-D image + 3-D model estimate), joint angles, posture measures,
        segment angles relative to the skis, centre of mass, ski geometry; plus flight
        segments with statistics, take-off / landing estimates, camera shots, and
        broadcast information read from the on-screen graphics (OCR: athlete, nation,
        speed, distances, gate ...), and the video metadata.

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
  7. full-body measures, camera motion (phase correlation), overlay OCR (RapidOCR).

The angle is a 2-D (image-plane) measurement: it is exact for a pure side
view and increasingly distorted as the camera moves towards the front/back.

Usage
  python skijump_angles.py https://www.youtube.com/watch?v=Wij38rVUTEY -o out.json
  python skijump_angles.py jump.mp4 -o out.json --debug-video dbg.mp4 --plot plot.png
  python skijump_angles.py jump.mp4 --start 60 --end 90 --all-frames

Requirements:  pip install opencv-contrib-python mediapipe numpy yt-dlp
               rapidocr-onnxruntime (overlay text, optional), matplotlib (--plot)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
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
def resolve_video(src: str, cache_dir: Path, max_height: int = 720):
    """Return (local video path, yt-dlp info dict or None); download if src is a URL."""
    if os.path.exists(src):
        return Path(src), None
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
            cands = [c for c in cands if c.suffix not in (".part", ".json")]
            path = cands[0]
        try:
            info = yt_dlp.YoutubeDL.sanitize_info(info)
        except Exception:
            info = {k: v for k, v in info.items() if isinstance(v, (str, int, float, list))}
        return path, info
    subprocess.run(["yt-dlp", "-f", fmt, "--merge-output-format", "mp4", "--write-info-json",
                    "-o", outtmpl, src], check=True)
    cands = [c for c in sorted(cache_dir.iterdir(), key=os.path.getmtime)
             if c.suffix in (".mp4", ".mkv", ".webm")]
    info = None
    ij = cands[-1].with_suffix(".info.json")
    if ij.exists():
        info = json.loads(ij.read_text(encoding="utf-8"))
    return cands[-1], info


def fetch_info(url: str):
    """video metadata only (no download) via yt-dlp; None if unavailable"""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            return yt_dlp.YoutubeDL.sanitize_info(ydl.extract_info(url, download=False))
    except Exception as e:
        print(f"[meta] could not fetch metadata: {e}", file=sys.stderr)
        return None


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
        z = np.array([l.z * w for l in lms], np.float32)          # relative depth, px
        world = None
        if r.pose_world_landmarks:
            world = np.array([[l.x, l.y, l.z] for l in r.pose_world_landmarks[0]], np.float32)
        return pts, vis, z, world

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
        pts, vis, z, world = r
        pts = pts / sc + np.array([x0, y0], np.float32)
        return pts, vis, z / sc, world

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
        """returns (pts, vis, box, method, z, world) or None"""
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
        pts, vis, z, world, name = best
        good = vis > 0.3
        if good.sum() < 6:
            return None
        p = pts[good]
        box = (float(p[:, 0].min()), float(p[:, 1].min()),
               float(p[:, 0].max()), float(p[:, 1].max()))
        return pts, vis, box, name, z, world


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
                                     center=ln_["c"].tolist(), segments=ln_["segs"],
                                     tail_pt=(ln_["c"] + ln_["d"] * ln_["smin"]).tolist(),
                                     tip_pt=(ln_["c"] + ln_["d"] * ln_["smax"]).tolist(),
                                     length_px=float(ln_["smax"] - ln_["smin"]))
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
# full-body measurements
# --------------------------------------------------------------------------- #
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb",
    "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
N = {n: i for i, n in enumerate(LANDMARK_NAMES)}

# segment mass fractions and CoM position (fraction from proximal end), de Leva (1996),
# male values; head CoM approximated by the centre of ears/nose.  Skis/boots not included.
DE_LEVA = [  # (name, proximal, distal, mass, com_frac)
    ("trunk", "mid_shoulder", "mid_hip", 0.4346, 0.45),
    ("upper_arm_l", "left_shoulder", "left_elbow", 0.0271, 0.5772),
    ("upper_arm_r", "right_shoulder", "right_elbow", 0.0271, 0.5772),
    ("forearm_l", "left_elbow", "left_wrist", 0.0162, 0.4574),
    ("forearm_r", "right_elbow", "right_wrist", 0.0162, 0.4574),
    ("hand_l", "left_wrist", "left_index", 0.0061, 0.5),
    ("hand_r", "right_wrist", "right_index", 0.0061, 0.5),
    ("thigh_l", "left_hip", "left_knee", 0.1416, 0.4095),
    ("thigh_r", "right_hip", "right_knee", 0.1416, 0.4095),
    ("shank_l", "left_knee", "left_ankle", 0.0433, 0.4459),
    ("shank_r", "right_knee", "right_ankle", 0.0433, 0.4459),
    ("foot_l", "left_heel", "left_foot_index", 0.0137, 0.4415),
    ("foot_r", "right_heel", "right_foot_index", 0.0137, 0.4415),
]
HEAD_MASS = 0.0694

# joint angle definitions: name -> (a, vertex, c)   (interior angle, 180 = straight)
JOINTS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),     # arm-trunk angle
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_wrist": ("left_elbow", "left_wrist", "left_index"),
    "right_wrist": ("right_elbow", "right_wrist", "right_index"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_ankle": ("left_knee", "left_ankle", "left_foot_index"),     # shank-foot angle
    "right_ankle": ("right_knee", "right_ankle", "right_foot_index"),
    "neck": ("head_center", "mid_shoulder", "mid_hip"),               # head-trunk angle
}

# body segments (proximal -> distal) for inclination / relative-to-ski angles
SEGMENTS = {
    "body_line": ("mid_ankle", "mid_shoulder"),
    "leg_line": ("mid_ankle", "mid_hip"),
    "trunk": ("mid_hip", "mid_shoulder"),
    "head": ("mid_shoulder", "head_center"),
    "upper_arm_l": ("left_shoulder", "left_elbow"),
    "upper_arm_r": ("right_shoulder", "right_elbow"),
    "forearm_l": ("left_elbow", "left_wrist"),
    "forearm_r": ("right_elbow", "right_wrist"),
    "thigh_l": ("left_hip", "left_knee"),
    "thigh_r": ("right_hip", "right_knee"),
    "shank_l": ("left_knee", "left_ankle"),
    "shank_r": ("right_knee", "right_ankle"),
    "foot_l": ("left_heel", "left_foot_index"),
    "foot_r": ("right_heel", "right_foot_index"),
}


def _named_points(P, V=None):
    """dict name -> point (2-D or 3-D) incl. derived mid points; visibility dict"""
    d = {n: np.asarray(P[i], float) for n, i in N.items()}
    v = {n: (float(V[i]) if V is not None else 1.0) for n, i in N.items()}

    def mid(name, a, b):
        d[name] = (d[a] + d[b]) / 2
        v[name] = min(v[a], v[b])
    mid("mid_shoulder", "left_shoulder", "right_shoulder")
    mid("mid_hip", "left_hip", "right_hip")
    mid("mid_ankle", "left_ankle", "right_ankle")
    mid("mid_ear", "left_ear", "right_ear")
    d["head_center"] = (d["left_ear"] + d["right_ear"] + d["nose"]) / 3
    v["head_center"] = max(v["left_ear"], v["right_ear"], v["nose"])
    return d, v


def _incl2d(p, q):
    """inclination of p->q vs image horizontal, deg, y up, (-180, 180]"""
    return math.degrees(math.atan2(-(q[1] - p[1]), q[0] - p[0]))


def _angle3(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float("nan") if n < 1e-9 else math.degrees(math.acos(np.clip(np.dot(v1, v2) / n, -1, 1)))


def _vec_angle(u, v):
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return float("nan") if n < 1e-9 else math.degrees(math.acos(np.clip(np.dot(u, v) / n, -1, 1)))


def _r(x, nd=1):
    if x is None:
        return None
    try:
        if not np.isfinite(x):
            return None
    except TypeError:
        return x
    return round(float(x), nd)


def center_of_mass(d):
    tot, acc = 0.0, np.zeros_like(d["mid_hip"])
    for _, a, b, m, f in DE_LEVA:
        acc += m * (d[a] + f * (d[b] - d[a]))
        tot += m
    acc += HEAD_MASS * d["head_center"]
    tot += HEAD_MASS
    return acc / tot


def body_measures(pts, vis, z, world, ski, frame_shape, min_vis=0.2):
    """All per-frame body measurements (2-D image, 3-D MediaPipe world, relative to skis)."""
    H, W = frame_shape[:2]
    d, v = _named_points(pts, vis)
    L = float(np.linalg.norm(d["mid_shoulder"] - d["mid_ankle"])) or 1.0
    out = {}

    # --- raw keypoints --------------------------------------------------------------
    out["keypoints_2d"] = {n: [_r(pts[i][0]), _r(pts[i][1]), _r(z[i]) if z is not None else None,
                              _r(vis[i], 3)] for n, i in N.items()}
    if world is not None:
        out["keypoints_3d"] = {n: [_r(world[i][0], 3), _r(world[i][1], 3), _r(world[i][2], 3)]
                               for n, i in N.items()}
    out["derived_points_2d"] = {k: [_r(d[k][0]), _r(d[k][1])] for k in
                                ("mid_shoulder", "mid_hip", "mid_ankle", "head_center")}
    com = center_of_mass(d)
    out["derived_points_2d"]["center_of_mass"] = [_r(com[0]), _r(com[1])]
    good = vis > 0.3
    p = pts[good] if good.sum() >= 3 else pts
    out["bbox"] = [_r(p[:, 0].min()), _r(p[:, 1].min()), _r(p[:, 0].max()), _r(p[:, 1].max())]
    out["body_length_px"] = _r(L)
    out["body_height_frac"] = _r((p[:, 1].max() - p[:, 1].min()) / H, 3)

    # --- joint angles -----------------------------------------------------------------
    ja = {}
    for name, (a, b, c) in JOINTS.items():
        ok = min(v[a], v[b], v[c]) >= min_vis
        ja[name] = _r(_angle3(d[a], d[b], d[c])) if ok else None
    out["joint_angles_2d"] = ja
    if world is not None:
        w, _ = _named_points(world)
        out["joint_angles_3d"] = {name: _r(_angle3(w[a], w[b], w[c]))
                                  for name, (a, b, c) in JOINTS.items()}
        thl, thr = w["left_knee"] - w["left_hip"], w["right_knee"] - w["right_hip"]
        ual, uar = w["left_elbow"] - w["left_shoulder"], w["right_elbow"] - w["right_shoulder"]
        trunk = w["mid_shoulder"] - w["mid_hip"]
        sh_line, hip_line = w["right_shoulder"] - w["left_shoulder"], w["right_hip"] - w["left_hip"]
        # twist between shoulder and hip lines, measured perpendicular to the trunk axis
        tn = trunk / (np.linalg.norm(trunk) + 1e-9)
        s_p, h_p = sh_line - np.dot(sh_line, tn) * tn, hip_line - np.dot(hip_line, tn) * tn
        out["posture_3d"] = {
            "leg_spread": _r(_vec_angle(thl, thr)),                 # angle between thighs
            "arm_spread": _r(_vec_angle(ual, uar)),                 # angle between upper arms
            "trunk_twist": _r(_vec_angle(s_p, h_p)),
            "head_to_trunk": _r(_vec_angle(w["nose"] - w["mid_ear"], trunk)),
            "ankle_separation_m": _r(np.linalg.norm(w["left_ankle"] - w["right_ankle"]), 3),
            "knee_separation_m": _r(np.linalg.norm(w["left_knee"] - w["right_knee"]), 3),
            "wrist_separation_m": _r(np.linalg.norm(w["left_wrist"] - w["right_wrist"]), 3),
            "hand_to_hip_l_m": _r(np.linalg.norm(w["left_wrist"] - w["left_hip"]), 3),
            "hand_to_hip_r_m": _r(np.linalg.norm(w["right_wrist"] - w["right_hip"]), 3),
            "body_length_m": _r(np.linalg.norm(w["mid_shoulder"] - w["mid_ankle"]), 3),
            "center_of_mass_m": [_r(x, 3) for x in center_of_mass(w)],
        }

    # --- segment inclinations (image) & 2-D distances ------------------------------------
    out["segment_inclination_2d"] = {
        k: (_r(_incl2d(d[a], d[b])) if min(v[a], v[b]) >= min_vis else None)
        for k, (a, b) in SEGMENTS.items()}
    out["distances_rel_2d"] = {    # divided by body length (mid-ankle -> mid-shoulder)
        "shoulder_width": _r(np.linalg.norm(d["left_shoulder"] - d["right_shoulder"]) / L, 3),
        "hip_width": _r(np.linalg.norm(d["left_hip"] - d["right_hip"]) / L, 3),
        "ankle_separation": _r(np.linalg.norm(d["left_ankle"] - d["right_ankle"]) / L, 3),
        "knee_separation": _r(np.linalg.norm(d["left_knee"] - d["right_knee"]) / L, 3),
        "wrist_separation": _r(np.linalg.norm(d["left_wrist"] - d["right_wrist"]) / L, 3),
        "hand_to_hip_l": _r(np.linalg.norm(d["left_wrist"] - d["left_hip"]) / L, 3),
        "hand_to_hip_r": _r(np.linalg.norm(d["right_wrist"] - d["right_hip"]) / L, 3),
        "head_to_shoulder": _r(np.linalg.norm(d["head_center"] - d["mid_shoulder"]) / L, 3),
    }

    # --- everything relative to the skis ----------------------------------------------
    if ski is not None:
        u = np.asarray(ski["dir"], float)
        if np.dot(u, d["mid_shoulder"] - d["mid_ankle"]) < 0:
            u = -u
        nrm = np.array([-u[1], u[0]])
        if nrm[1] > 0:
            nrm = -nrm
        out["relative_to_ski_2d"] = {
            k: (_r(signed_angle_to_ski(d[b] - d[a], u)) if min(v[a], v[b]) >= min_vis else None)
            for k, (a, b) in SEGMENTS.items()}
        rel = com - d["mid_ankle"]
        out["center_of_mass_vs_ski"] = {
            "along_ski_from_ankles": _r(float(np.dot(rel, u)) / L, 3),   # + = towards tips
            "above_ski": _r(float(np.dot(rel, nrm)) / L, 3),
        }
    return out


def ski_geometry(ski):
    if ski is None:
        return None
    g = {"n_skis": ski["n_skis"], "v_angle_2d": _r(ski.get("v_angle")),
         "axis_inclination_2d": None, "visible_length_rel": _r(ski["support"], 3)}
    u = ski["dir"]
    g["axis_inclination_2d"] = _r(math.degrees(math.atan2(-u[1], u[0])) if u[0] >= 0
                                  else math.degrees(math.atan2(u[1], -u[0])))
    for foot, sk in ski["skis"].items():
        g[foot] = {"tail": [_r(x) for x in sk["tail_pt"]], "tip": [_r(x) for x in sk["tip_pt"]],
                   "length_px": _r(sk["length_px"]),
                   "inclination_2d": _r(_incl2d(sk["tail_pt"], sk["tip_pt"])),
                   "visible_length_rel": _r(sk["support"], 3),
                   "tip_ahead_of_boot_rel": _r(sk["tip"], 3),
                   "tail_behind_boot_rel": _r(sk["tail"], 3)}
    return g


# --------------------------------------------------------------------------- #
# broadcast overlay text (OCR) and video metadata
# --------------------------------------------------------------------------- #
FIS_NATIONS = set("""AUT BUL CAN CHN CZE EST FIN FRA GER GBR ITA JPN KAZ KOR NOR POL ROU RUS SLO
SUI SWE TUR UKR USA SVK BLR GEO LAT LTU HUN ISL DEN NED BEL ESP AND ARG AUS BIH CRO GRE ISR
MGL NZL SRB""".split())


class OverlayReader:
    """Reads on-screen graphics with RapidOCR (optional dependency)."""

    def __init__(self):
        self.ocr = None
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR()
        except Exception:
            try:
                from rapidocr import RapidOCR          # newer package name
                self.ocr = RapidOCR()
            except Exception:
                print("[ocr] rapidocr not installed -> overlay text disabled "
                      "(pip install rapidocr-onnxruntime)", file=sys.stderr)

    def read(self, frame, min_conf=0.75):
        if self.ocr is None:
            return []
        H, W = frame.shape[:2]
        try:
            res = self.ocr(frame)
            res = res[0] if isinstance(res, tuple) else res
            if hasattr(res, "txts"):                      # rapidocr >= 2 result object
                res = list(zip(res.boxes, res.txts, res.scores)) if res.txts else []
        except Exception:
            return []
        out = []
        for box, txt, conf in res or []:
            if conf < min_conf or len(txt.strip()) < 2:
                continue
            b = np.asarray(box, float)
            out.append(dict(text=txt.strip(), conf=float(conf),
                            box=[_r(b[:, 0].min() / W, 3), _r(b[:, 1].min() / H, 3),
                                 _r(b[:, 0].max() / W, 3), _r(b[:, 1].max() / H, 3)]))
        return out


def _norm_txt(t):
    return re.sub(r"\s+", "", t.upper())


def build_ocr_events(readings, ocr_step):
    """merge identical texts seen in consecutive OCR frames into events"""
    events, open_ = [], {}
    for t, items in readings:
        seen = set()
        for it in items:
            if re.fullmatch(r"[0-9:.\s]{1,6}", it["text"]) and "." not in it["text"]:
                continue                      # clocks / bare small numbers
            k = _norm_txt(it["text"])
            seen.add(k)
            e = open_.get(k)
            if e is not None and t - e["last"] <= 2.5 * ocr_step:
                e["last"] = t
                e["conf"] = max(e["conf"], it["conf"])
                if it["text"].count(" ") > e["text"].count(" "):
                    e["text"] = it["text"]    # keep the best-spaced variant
            else:
                e = dict(text=it["text"], first=t, last=t, conf=it["conf"], box=it["box"])
                open_[k] = e
                events.append(e)
    for e in events:
        e["first"], e["last"], e["conf"] = _r(e["first"], 2), _r(e["last"], 2), _r(e["conf"], 3)
    return events


def _num(s):
    return float(s.replace(",", "."))


def parse_broadcast(events):
    """extract structured facts from overlay texts"""
    info = {"athletes": [], "speed_kmh": [], "labelled_distances_m": [], "distances_m": [],
            "gate": [], "wind": [], "round": [], "records": []}
    nat_re = re.compile(r"^(?:\d{1,3}\s+)?((?:[A-ZÀ-Ž][A-Za-zÀ-ž'\-]+\s+){1,3}[A-ZÀ-Ž][A-Za-zÀ-ž'\-]+)"
                        r"\s+([A-Z]{3})$")
    for e in events:
        t = e["text"].strip()
        tu = t.upper()
        tn = _norm_txt(t)
        span = {"first": e["first"], "last": e["last"]}
        m = nat_re.match(t)
        if m and m.group(2) in FIS_NATIONS:
            info["athletes"].append({"name": m.group(1).title(), "nation": m.group(2), **span})
        else:                                  # OCR sometimes drops the spaces
            m = re.fullmatch(r"(?:\d{1,3})?([A-ZÀ-Ž]{5,30})([A-Z]{3})", tn)
            if m and m.group(2) in FIS_NATIONS and not re.search(r"KM|POINT|ROUND", tn):
                info["athletes"].append({"name": m.group(1).title(), "nation": m.group(2),
                                         "note": "spaces lost in OCR", **span})
        m = re.match(r"NEXT[:\s]+(.+)$", tu)
        if m:
            info.setdefault("next_athlete", []).append({"text": m.group(1).strip(), **span})
        for m in re.finditer(r"(\d{2,3}[.,]\d)\s*KM\s*/?\s*H", tu):
            info["speed_kmh"].append({"value": _num(m.group(1)), **span})
        for m in re.finditer(r"(TOBEAT|PB|SB|HILLRECORD|WORLDRECORD|WR|HR|HS|K|LEADER)[:\-]?"
                             r"(\d{2,3}[.,]\d)M", tn):
            info["labelled_distances_m"].append(
                {"label": m.group(1), "value": _num(m.group(2)), **span})
        m = re.fullmatch(r"(\d{2,3}[.,]\d)M", tn)
        if m:
            info["distances_m"].append({"value": _num(m.group(1)), **span})
        m = re.search(r"GATE[:\s]*([+-]?\d+(?:[.,]\d+)?)", tu)
        if m:
            info["gate"].append({"value": _num(m.group(1)), "raw": t, **span})
        m = re.search(r"WIND[:\s]*([+-]?\d+[.,]\d+)", tu)
        if m:
            info["wind"].append({"value": _num(m.group(1)), "raw": t, **span})
        m = re.search(r"(FINAL\s*ROUND|1ST\s*ROUND|2ND\s*ROUND|TRIAL\s*ROUND|QUALIFICATION|"
                      r"TEAM\s*EVENT|INDIVIDUAL)", tu)
        if m:
            info["round"].append({"value": t, **span})
        if re.search(r"(WORLD|HILL|NEW|PERSONAL|NATIONAL|TRACK)RECORD", tn):
            m = re.search(r"(\d{3}[.,]\d)", tu)
            info["records"].append({"text": t, "value_m": _num(m.group(1)) if m else None, **span})
    # collapse duplicates of the same athlete
    ath = {}
    for a in sorted(info["athletes"], key=lambda x: -x["name"].count(" ")):
        k = (a["name"].replace(" ", "").lower(), a["nation"])
        if k in ath:
            ath[k]["first"] = min(ath[k]["first"], a["first"])
            ath[k]["last"] = max(ath[k]["last"], a["last"])
            ath[k]["n_seen"] += 1
        else:
            ath[k] = {**a, "n_seen": 1}
            ath[k].pop("note", None) if " " in a["name"] else None
    info["athletes"] = sorted(ath.values(), key=lambda a: a["first"])
    return info


def video_metadata_from_info(info: dict | None, path: Path) -> dict:
    md = {"file": str(path)}
    try:
        md["file_size_bytes"] = path.stat().st_size
    except OSError:
        pass
    if info:
        for k in ("id", "title", "uploader", "channel", "upload_date", "duration",
                  "webpage_url", "view_count", "like_count", "categories", "tags",
                  "description"):
            if info.get(k) is not None:
                md[k] = info[k]
        if isinstance(md.get("description"), str) and len(md["description"]) > 3000:
            md["description"] = md["description"][:3000] + " ..."
        if info.get("chapters"):
            md["chapters"] = [{"start": c.get("start_time"), "end": c.get("end_time"),
                               "title": c.get("title")} for c in info["chapters"]]
    elif shutil.which("ffprobe"):
        try:
            j = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream=codec_name",
                 "-of", "json", str(path)], capture_output=True, text=True).stdout)
            md["container_tags"] = j.get("format", {}).get("tags", {})
            md["codecs"] = [s.get("codec_name") for s in j.get("streams", [])]
        except Exception:
            pass
    return md


def camera_shift(prev_gray, gray, scale):
    """global translation between two consecutive samples (phase correlation), full-res px"""
    if prev_gray is None:
        return None
    win = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(prev_gray, gray, win)
    if resp < 0.05:
        return None
    return [_r(dx * scale), _r(dy * scale), _r(resp, 3)]


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
    cam_motion: object = None            # global image shift vs previous sample [dx, dy] px
    _pts: object = field(default=None, repr=False)
    _vis: object = field(default=None, repr=False)
    _z: object = field(default=None, repr=False)
    _world: object = field(default=None, repr=False)
    _box: object = field(default=None, repr=False)
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
    pts, vis, box, method, z, world = r
    s.pose_ok, s.method, s._pts = True, method, pts
    s._vis, s._z, s._world, s._box = vis, z, world, box
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
                  progress=True, ocr_step=0.5):
    meta = probe_video(path)
    fps, nfr, dur = meta["fps"], meta["frames"], meta["duration"]
    end = dur if end is None else min(end, dur)

    pose_est = PoseEstimator(model_dir, heavy=heavy)
    ski_det = SkiDetector()
    ocr = OverlayReader() if ocr_step and ocr_step > 0 else None
    ocr_readings, next_ocr = [], start
    prev_cam = None
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
        # global camera motion (phase correlation on a downscaled frame)
        cam = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_cam is not None and s.shot == samples[-1][0].shot:
            s.cam_motion = camera_shift(prev_cam, cam, frame.shape[1] / 320)
        prev_cam = cam
        # broadcast graphics
        if ocr is not None and ocr.ocr is not None and t >= next_ocr - 1e-6:
            ocr_readings.append((float(t), ocr.read(frame)))
            next_ocr = t + ocr_step
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
    meta["ocr_readings"] = ocr_readings
    meta["ocr_step"] = ocr_step
    return S, segments, meta


def _stat(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "mean": _r(v.mean(), 2), "median": _r(np.median(v), 2),
            "std": _r(v.std(), 2), "min": _r(v.min(), 2), "max": _r(v.max(), 2)}


def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def _body(s, meta):
    if "_body" not in s.__dict__:
        s.__dict__["_body"] = body_measures(
            s._pts, s._vis, s._z, s._world, s._ski,
            (meta["height"], meta["width"])) if s.pose_ok else None
    return s.__dict__["_body"]


def _events_for_segment(S, a, b):
    """take-off / landing estimates just outside a flight segment (same camera shot)"""
    ev = {}
    sh0, sh1 = S[a].shot, S[b - 1].shot
    for j in range(a - 1, max(-1, a - 25), -1):
        if S[j].shot != sh0:
            break
        if S[j].pose_ok and S[j].knee is not None and S[j].knee < 130:
            ev["takeoff_est"] = {"time": S[j + 1].t, "note": "first sample after in-run crouch"}
            break
    for j in range(b, min(len(S), b + 25)):
        if S[j].shot != sh1:
            break
        if S[j].pose_ok and S[j].knee is not None and S[j].knee < 150:
            body = S[j].__dict__.get("_body") or {}
            sep = _get(body, "distances_rel_2d", "ankle_separation")
            ev["landing_est"] = {"time": S[j].t, "knee_angle": S[j].knee,
                                 "ankle_separation_rel": sep,
                                 "telemark_likely": bool(sep is not None and sep > 0.2)}
            break
    return ev


def build_json(src, path, S, segments, meta, step, all_frames=False, keypoints=True,
               video_info=None):
    # body measures for all samples that will be written / summarised
    for i, s in enumerate(S):
        if s.pose_ok and (s.airborne or all_frames or
                          any(b <= i < b + 25 or a - 25 <= i < a for a, b in segments)):
            _body(s, meta)

    # --- overlay text ----------------------------------------------------------------
    ocr_events = build_ocr_events(meta.get("ocr_readings", []), meta.get("ocr_step") or 0.5)
    broadcast = parse_broadcast(ocr_events)

    # --- flight segments ---------------------------------------------------------------
    segs = []
    for k, (a, b) in enumerate(segments):
        pts = [S[i] for i in range(a, b)]
        rel = [p for p in pts if p.angle is not None and p.view != "frontal"]
        bodies = [p.__dict__.get("_body") for p in pts]
        bodies = [x for x in bodies if x]
        t0, t1 = pts[0].t, round(pts[-1].t + step, 3)
        ctx = [e for e in ocr_events if e["last"] >= t0 - 15 and e["first"] <= t1 + 5]
        sp = [x["value"] for x in broadcast["speed_kmh"] if x["last"] >= t0 - 15 and x["first"] <= t1]
        ath = [x for x in broadcast["athletes"] if x["last"] >= t0 - 60 and x["first"] <= t1 + 30]
        segs.append(dict(
            id=k, start=t0, end=t1, duration_video_s=round(t1 - t0, 3),
            shots=sorted({p.shot for p in pts}),
            n_points=len(pts), n_angle_measured=len(rel),
            views={v: sum(p.view == v for p in pts) for v in ("side", "oblique", "frontal")},
            events=_events_for_segment(S, a, b),
            stats={
                "angle_body_ski": _stat([p.angle for p in rel]),
                "angle_leg_ski": _stat([p.angle_leg for p in rel]),
                "angle_trunk_ski": _stat([p.angle_torso for p in rel]),
                "knee_2d": _stat([p.knee for p in pts]),
                "hip_2d": _stat([p.hip for p in pts]),
                "ski_v_angle_2d_frontal_views": _stat(
                    [p.v_angle for p in pts if p.view == "frontal" and p.n_skis == 2]),
                "knee_3d": _stat([np.nanmean([_get(x, "joint_angles_3d", "left_knee") or np.nan,
                                              _get(x, "joint_angles_3d", "right_knee") or np.nan])
                                  for x in bodies]),
                "hip_3d": _stat([np.nanmean([_get(x, "joint_angles_3d", "left_hip") or np.nan,
                                             _get(x, "joint_angles_3d", "right_hip") or np.nan])
                                 for x in bodies]),
                "arm_trunk_3d": _stat([np.nanmean([
                    _get(x, "joint_angles_3d", "left_shoulder") or np.nan,
                    _get(x, "joint_angles_3d", "right_shoulder") or np.nan]) for x in bodies]),
                "leg_spread_3d": _stat([_get(x, "posture_3d", "leg_spread") for x in bodies]),
                "arm_spread_3d": _stat([_get(x, "posture_3d", "arm_spread") for x in bodies]),
                "head_to_trunk_3d": _stat([_get(x, "posture_3d", "head_to_trunk") for x in bodies]),
                "com_along_ski": _stat([_get(x, "center_of_mass_vs_ski", "along_ski_from_ankles")
                                        for x, p in zip(bodies, pts) if p.view != "frontal"]),
            },
            overlay={"speed_kmh": sp[-1] if sp else None,
                     "athletes": [{"name": x["name"], "nation": x["nation"]} for x in ath],
                     "texts": [e["text"] for e in ctx]},
        ))

    # --- shots -------------------------------------------------------------------------
    shots = []
    for sh in sorted({s.shot for s in S}):
        ss = [s for s in S if s.shot == sh]
        pose = [s for s in ss if s.pose_ok]
        hfrac = [(s._box[3] - s._box[1]) / meta["height"] for s in pose if s._box]
        cam = [math.hypot(s.cam_motion[0], s.cam_motion[1]) for s in ss if s.cam_motion]
        fa = sum(s.airborne for s in ss) / len(ss)
        fp = len(pose) / len(ss)
        mh = float(np.median(hfrac)) if hfrac else 0.0
        if fa > 0.3:
            typ = "flight"
        elif fp < 0.2:
            typ = "no_person"
        elif mh > 0.45:
            typ = "person_closeup"
        else:
            typ = "person_wide"
        shots.append(dict(id=sh, start=ss[0].t, end=round(ss[-1].t + step, 3), type=typ,
                          pose_fraction=_r(fp, 2), airborne_fraction=_r(fa, 2),
                          median_person_height_frac=_r(mh, 3),
                          mean_camera_motion_px=_r(np.mean(cam)) if cam else None))

    # --- samples -----------------------------------------------------------------------
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
                 segment=s.segment, airborne=s.airborne,
                 angle_leg=s.angle_leg if reliable else None,
                 angle_torso=s.angle_torso if reliable else None, n_skis=s.n_skis,
                 body_inclination=s.body_incl, ski_inclination=s.ski_incl,
                 knee_angle=s.knee, hip_angle=s.hip, view=s.view,
                 ski_v_angle_2d=s.v_angle, confidence=s.confidence,
                 shot=s.shot, frame=s.frame, pose_method=s.method,
                 camera_motion_px=s.cam_motion)
        body = s.__dict__.get("_body")
        if body is not None:
            if not keypoints:
                body = {k: v for k, v in body.items() if not k.startswith("keypoints")}
            d["body"] = body
            d["skis"] = ski_geometry(s._ski)
        data.append(d)

    return {
        "source": src,
        "video_file": str(path),
        "video": {"fps": meta["fps"], "frames": meta["frames"], "width": meta["width"],
                  "height": meta["height"], "duration_s": round(meta["duration"], 3)},
        "video_metadata": video_metadata_from_info(video_info, Path(path)),
        "sampling_step_s": step,
        "landmark_names": LANDMARK_NAMES,
        "definitions": DEFINITIONS,
        "broadcast_info": broadcast,
        "segments": segs,
        "shots": shots,
        "ocr_events": ocr_events,
        "data": data,
    }


DEFINITIONS = {
    "time": "video time in seconds (broadcast replays are often slow motion)",
    "angle": "2-D angle (deg) between the body line (mid-ankle -> mid-shoulder) and the ski "
             "axis. Positive = body above the skis. null for frontal/rear views, outliers or "
             "when no ski was found.",
    "angle_leg / angle_torso": "same with mid-ankle -> mid-hip / mid-hip -> mid-shoulder",
    "angle_raw": "angle before outlier rejection (reported even for frontal views)",
    "outlier": "angle_raw > 15 deg from the median of its neighbours (+-0.3 s, same shot) or "
               "with < 2 such neighbours",
    "angle_smooth": "median of 3 neighbouring samples in the same shot",
    "body_inclination / ski_inclination": "vs image horizontal, deg (camera dependent)",
    "knee_angle / hip_angle": "mean of left/right interior 2-D joint angles, 180 = straight",
    "view": "side | oblique | frontal (estimated from body width ratio and ski V)",
    "confidence": "0..1 heuristic (pose visibility x ski support x view)",
    "camera_motion_px": "[dx, dy, response] global image shift since previous sample",
    "body.keypoints_2d": "33 MediaPipe landmarks: [x_px, y_px, z_rel_px, visibility]; z is "
                         "relative depth (smaller = closer to camera), same scale as x",
    "body.keypoints_3d": "MediaPipe world landmarks, metres, origin = mid-hip, axes aligned "
                         "with the camera (x right, y down, z away); model estimate",
    "body.joint_angles_2d / _3d": "interior angles, deg, 180 = straight. shoulder = upper "
                                  "arm vs trunk side (arm-trunk angle); ankle = shank vs foot; "
                                  "neck = head centre - mid-shoulder - mid-hip. 2-D values are "
                                  "null when a keypoint is not visible",
    "body.posture_3d": "view-independent posture from 3-D landmarks: leg_spread (angle "
                       "between thighs), arm_spread, trunk_twist, head_to_trunk, separations "
                       "in metres (approximate scale), centre of mass (m)",
    "body.segment_inclination_2d": "each segment vs image horizontal, deg, y up",
    "body.relative_to_ski_2d": "each segment vs the ski axis, deg (signed, + = above skis)",
    "body.distances_rel_2d": "image distances divided by body length (ankle->shoulder)",
    "body.center_of_mass_vs_ski": "body CoM (de Leva segment masses, without skis) relative "
                                  "to mid-ankle, along the ski (+ = towards tips) and above "
                                  "it, in body lengths",
    "skis": "per ski tail/tip points (px), length, inclination; v_angle_2d = opening between "
            "the skis in the image (meaningful in frontal/rear views)",
    "segments": "continuous airborne periods with statistics, take-off/landing estimates and "
                "overlay context (speed, athletes, texts)",
    "shots": "camera shots between detected cuts with a coarse type",
    "broadcast_info": "facts parsed from on-screen graphics (OCR)",
    "ocr_events": "all on-screen texts with first/last time and normalised box",
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


def dump_json(obj, fp):
    """indented JSON, but short numeric lists kept on one line"""
    txt = json.dumps(obj, indent=1, ensure_ascii=False)
    txt = re.sub(r"\[\s*([-0-9.eE,\s]*?|null|true|false)\s*\]",
                 lambda m: "[" + re.sub(r"\s+", " ", m.group(1)).replace(" ,", ",") + "]"
                 if "\n" in m.group(0) else m.group(0), txt)
    txt = re.sub(r"\[((?:\s*(?:-?[0-9.eE]+|null),?)+)\s*\]",
                 lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", txt)
    fp.write(txt)


CSV_GROUPS = ("joint_angles_2d", "joint_angles_3d", "posture_3d", "relative_to_ski_2d",
              "segment_inclination_2d", "distances_rel_2d", "center_of_mass_vs_ski")


def write_csv(out, path):
    import csv
    rows = []
    for d in out["data"]:
        r = {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
        body = d.get("body") or {}
        for g in CSV_GROUPS:
            for k, v in (body.get(g) or {}).items():
                if not isinstance(v, list):
                    r[f"{g}.{k}"] = v
        sk = d.get("skis") or {}
        for k in ("v_angle_2d", "axis_inclination_2d", "visible_length_rel"):
            r[f"skis.{k}"] = sk.get(k)
        rows.append(r)
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--no-keypoints", action="store_true",
                    help="omit raw 2-D/3-D keypoints (smaller JSON; measures are kept)")
    ap.add_argument("--ocr-step", type=float, default=0.5,
                    help="read on-screen graphics every N seconds (0 = off)")
    ap.add_argument("--fast", action="store_true", help="use the lighter pose model")
    ap.add_argument("--models", default=str(Path(__file__).with_name("models")))
    ap.add_argument("--cache", default=str(Path(__file__).with_name("video_cache")))
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--debug-video", default=None, help="write annotated mp4 (1 frame per step)")
    ap.add_argument("--plot", default=None, help="write PNG plot of angle over time")
    ap.add_argument("--csv", default=None, help="also write a flat CSV of all scalar measures")
    ap.add_argument("--source-url", default=None,
                    help="for a local file: URL it came from (to add title/uploader/... metadata)")
    a = ap.parse_args()

    path, info = resolve_video(a.source, Path(a.cache), a.max_height)
    if info is None and a.source_url:
        info = fetch_info(a.source_url)
    print(f"[video] {path}", file=sys.stderr)
    S, segs, meta = analyse_video(path, a.step, a.start, a.end, Path(a.models),
                                  heavy=not a.fast, debug_video=a.debug_video,
                                  ocr_step=a.ocr_step)
    out = build_json(a.source, path, S, segs, meta, a.step, a.all_frames,
                     keypoints=not a.no_keypoints, video_info=info)
    op = a.output or str(Path(path).with_suffix(".angles.json"))
    with open(op, "w", encoding="utf-8") as f:
        dump_json(out, f)
    print(f"[done] {len(out['data'])} points, {len(segs)} airborne segments -> {op}",
          file=sys.stderr)
    for sg in out["segments"]:
        st = sg["stats"]["angle_body_ski"] or {}
        print(f"  segment {sg['id']}: {sg['start']:.1f}-{sg['end']:.1f}s  "
              f"angle measured {sg['n_angle_measured']}/{sg['n_points']}  "
              f"mean {st.get('mean')}  speed {sg['overlay']['speed_kmh']}",
              file=sys.stderr)
    bi = out["broadcast_info"]
    if bi["athletes"]:
        print("  athletes on screen: " + ", ".join(f"{x['name']} ({x['nation']})"
                                                   for x in bi["athletes"]), file=sys.stderr)
    if a.csv:
        write_csv(out, a.csv)
    if a.plot:
        save_plot(out, a.plot)


if __name__ == "__main__":
    main()
