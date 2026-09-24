# SkiJumpAnalyzer

Extracts the **body-to-ski angle** of a ski jumper every 0.1 s while he/she is airborne,
from a local video or a YouTube link, and writes it as JSON.

```
pip install -r requirements.txt
python skijump_angles.py "https://www.youtube.com/watch?v=Wij38rVUTEY" -o out.json --plot out.png
python skijump_angles.py sj.mp4 -o out.json --debug-video dbg.mp4      # annotated video
python skijump_angles.py sj.mp4 --start 55 --end 80 --all-frames       # window, all samples
```

MediaPipe models (~60 MB) are downloaded automatically into `models/` on first run;
YouTube videos are cached in `video_cache/` (H.264 preferred, ffmpeg is used as a
fallback decoder for AV1/VP9). A 2.5-min 720p broadcast takes ~6 min on 2 CPU cores
(`--fast` uses the lighter pose model).

## Method

| step | technique |
|---|---|
| shot cuts | HSV-histogram + pixel-difference between samples → resets tracking |
| jumper + pose | MediaPipe PoseLandmarker (heavy), on a tracked crop → full frame → EfficientDet person crops |
| skis | line segments (FastLineDetector) around the feet, jumper's limbs masked out; collinear segments grouped into lines; a line is a ski if it is 0.35–2.5 × body length, passes under a boot and extends ahead of it; bonus for parallel edge pairs and for agreement with the previous sample |
| body line | mid-ankle → mid-shoulder (also mid-ankle → mid-hip and hip → shoulder) |
| angle | image-plane angle between body line and ski axis (bisector of the V if both skis are found); positive = body above the skis |
| view | side / oblique / frontal from body width ratio and the 2-D V opening. **Frontal/rear views → angle = null** (not observable) |
| airborne | legs extended (knee > 140°, hip > 115°) + skis at the feet + plausible angle (or wide V in frontal views); gaps closed, blips removed, segments need ≥ 3 "strong" samples; pieces < 1.2 s apart are merged |
| clean-up | outliers (> 15° from neighbours or isolated) → null; `angle_smooth` = median of 3 |

## Output (`*.angles.json`)

```json
{
  "segments": [{"id": 0, "start": 22.0, "end": 27.2, "n_measured": 29, "angle_mean": 15.6, ...}],
  "data": [
    {"time": 22.3, "angle": 15.15, "angle_smooth": 15.15, "segment": 0, "airborne": true,
     "angle_leg": 22.28, "angle_torso": 3.57, "knee_angle": 173.4, "hip_angle": 158.9,
     "view": "side", "confidence": 1.0, "angle_raw": 15.15, "outlier": false, ...}
  ],
  "definitions": { ... }
}
```

`time` is **video time** – broadcast replays are often slow motion, so a replayed flight
lasts longer than the real one.

## Limitations

* 2-D measurement: exact only for a side view, increasingly distorted in oblique views
  (`view` and `confidence` tell you which points to trust). No angle for front/rear shots.
* Very distant shots (jumper < ~40 px tall) are skipped – pose can't be estimated.
* Generic pose models are not trained on ski jumpers; ankle keypoints on ski boots can be
  off by a few pixels, which shifts the angle by a few degrees.
* Broadcast graphics, dissolves and crowd shots can still create occasional false points.

Example results for the Planica video (Wij38rVUTEY): `sj.angles.json`, `sj_angles.png`,
`sj_debug.mp4` (annotated: yellow skeleton, red body line, blue ski segments, green ski axis).
