# Depth → colorized video (agent brief)

You are producing a colorized MP4 from the RealSense depth capture inside
a Caddy-Training-Data recording folder.

## Files in the recording folder

Every `Caddy-Training-Data-YYYY-MM-DD_HH-MM-SS/` folder that captured
RealSense contains:

- `realsense_color.mp4` — RGB stream (BGR mp4v), width×height @ fps
- `realsense_depth.bin` — concatenated raw `uint16` depth frames, aligned
  to color (so pixel (r, c) is the same physical ray as in the mp4)
- `realsense_depth_meta.json` — shape/intrinsics/scale:
  ```json
  {
    "width": 640, "height": 480, "fps": 30,
    "dtype": "uint16",
    "depth_scale_m": 0.0010000000474974513,
    "aligned_to": "color",
    "color_file": "realsense_color.mp4",
    "depth_file": "realsense_depth.bin",
    "timestamps_file": "realsense_depth_ts.jsonl",
    "intrinsics": { "fx": ..., "fy": ..., "ppx": ..., "ppy": ..., "model": "...", "coeffs": [...] }
  }
  ```
- `realsense_depth_ts.jsonl` — one line per depth frame:
  `{idx, wall_t, rel_t, rs_t_ms}`. Use for cross-stream sync with
  `ego.jsonl` / `gps.jsonl` / `control.jsonl`. Not needed for the video
  itself.

Each raw depth pixel in `.bin` is a `uint16`; metres = `value * depth_scale_m`
(scale is ~0.001, so raw units are millimetres on D435).

## Loading depth

```python
import json, numpy as np
meta = json.load(open("realsense_depth_meta.json"))
H, W = meta["height"], meta["width"]
depth_u16 = np.fromfile("realsense_depth.bin", dtype=np.uint16).reshape(-1, H, W)
# Optional: depth_m = depth_u16.astype(np.float32) * meta["depth_scale_m"]
```

Frame count comes from the file size — there's no header. Sanity:
`depth_u16.shape[0]` should equal `len(open(ts_file).readlines())`.

## Colorizing one frame

`cv2.applyColorMap` needs `uint8`. The live UI clips to ~6 m and rescales:

```python
import cv2
clip_m = 6.0
clip_u = int(clip_m / meta["depth_scale_m"])           # in raw units
d = np.clip(depth_u16[i], 0, clip_u)
d8 = (d.astype(np.float32) * (255.0 / max(1, clip_u))).astype(np.uint8)
bgr = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
# Optional: mask invalid (zero) pixels as black
bgr[depth_u16[i] == 0] = 0
```

Tradeoffs:
- `COLORMAP_JET` matches the live web UI. `COLORMAP_TURBO` is perceptually
  better. `COLORMAP_INFERNO` reads well on dark backgrounds.
- A fixed clip (6 m here) makes brightness comparable across frames. If you
  prefer per-frame autoscale use `clip = depth_u16[i].max()` — looks more
  contrasty but flickers.
- `np.where(depth_u16[i] == 0, …)` is worth keeping: zeros are
  "no return" pixels, not "very close." Coloring them as zero/black
  preserves that distinction.

## Producing the video

Match the source fps and frame size exactly so it sits next to
`realsense_color.mp4`:

```python
fps = meta["fps"]
W, H = meta["width"], meta["height"]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("realsense_depth_color.mp4", fourcc, fps, (W, H))
for i in range(depth_u16.shape[0]):
    d = np.clip(depth_u16[i], 0, clip_u)
    d8 = (d.astype(np.float32) * (255.0 / max(1, clip_u))).astype(np.uint8)
    bgr = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    bgr[depth_u16[i] == 0] = 0
    writer.write(bgr)
writer.release()
```

For a side-by-side video with color, `np.hstack((color_bgr, depth_bgr))`
each frame and double the writer's width. The color mp4 is the same length
+ fps as the depth stream by construction.

## Gotchas

- Depth is **already aligned to color** at capture time — do NOT re-run
  `rs.align`. Pixel (r, c) corresponds directly between the two streams.
- `.bin` has no header. If `reshape(-1, H, W)` raises a size mismatch,
  the file was truncated (recorder killed mid-frame). Trim with
  `// (H*W*2)` bytes before reshape.
- mp4v is widely supported but not great quality. For a viewer-friendly
  artifact you can swap to `*"avc1"` (H.264) if your OpenCV build has it.
- Depth uses 16-bit values up to ~65 m on a D435 (with `depth_scale_m`
  ~0.001). Pixels beyond `clip_u` get flattened to one colour by the
  `np.clip` — choose `clip_m` to match the scene.
