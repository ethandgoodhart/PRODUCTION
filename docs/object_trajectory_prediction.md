# Object Trajectory Prediction

The segmentation pipeline keeps object prediction modular:

```
YOLO detections -> ByteTrack IDs -> homography ground projection
-> object predictor provider -> collision/risk -> UI
```

The built-in fallback is constant velocity. A real model should run as a
separate provider and receive normalized agent histories instead of being
embedded in `segmentation_infer.py`.

## Why Not UniAD Directly

UniAD is an end-to-end autonomous-driving stack. It couples perception,
tracking, mapping, motion forecasting, occupancy, and planning around
nuScenes-style multi-camera BEV inputs. That is a poor fit for the current
front-camera YOLO/ByteTrack pipeline as a small object-futures module.

## Recommended Model Path

The current running provider is Social-NCE Social-STGCNN because it has public
pretrained weights in the repo and is lightweight enough for a live sidecar.
It predicts multi-agent futures from recent observed trajectories using a
spatio-temporal graph convolutional network.

Trajectron++ remains supported as a sidecar boundary, but it needs an actual
`model_registrar-<checkpoint>.pt` checkpoint. MTR or QCNet are better long-term
targets if we build dataset-style scene tensors with map/lane context.

## Provider Contract

Run `./start.sh` with:

```
--seg-object-predictor-url http://127.0.0.1:8765/predict
```

For the built-in Social-STGCNN sidecar:

```
--with-social-stgcnn
```

This uses:

```
.cache/third_party/social-nce-stgcnn/checkpoint-snce/snce-social-stgcnn-univ/val_best.pth
```

Override with:

```
--social-stgcnn-checkpoint-dir /path/to/checkpoint_dir
```

For the Trajectron++ sidecar:

```
--with-trajectronpp
```

The sidecar expects an official Trajectron++ model directory containing a
`model_registrar-<checkpoint>.pt` file. The StanfordASL repository includes
configs under `experiments/nuScenes/models/*`, but not the checkpoint weights
the notebooks reference. If no checkpoint is present, the sidecar stays up on
`127.0.0.1:8765` and `/health` reports unavailable; `start.sh` then keeps the
safe constant-velocity fallback instead of sending live prediction traffic to a
missing model.

Override the checkpoint location with:

```
--trajectronpp-model-dir /path/to/model_dir --trajectronpp-checkpoint 12
```

`segmentation_infer.py` sends:

```json
{
  "schema": "caddy.object_prediction.v1",
  "frame_index": 123,
  "fps": 30.0,
  "horizon_s": 4.0,
  "step_s": 0.5,
  "ego": {
    "speed_mps": 0.9,
    "planned_path_m": [[2.0, 0.1], [2.5, 0.2]]
  },
  "agents": [
    {
      "track_id": 42,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.82,
      "state": {
        "x_m": 8.4,
        "y_m": -1.2,
        "vx_mps": 0.1,
        "vy_mps": 0.0,
        "speed_mps": 0.1
      },
      "history_m": [[-1.0, 8.2, -1.2], [-0.5, 8.3, -1.2], [0.0, 8.4, -1.2]]
    }
  ]
}
```

The provider returns multimodal futures:

```json
{
  "provider": "trajectronpp",
  "agents": [
    {
      "track_id": 42,
      "modes": [
        { "prob": 0.65, "future_m": [[8.5, -1.2], [8.6, -1.1]] },
        { "prob": 0.25, "future_m": [[8.4, -1.3], [8.3, -1.5]] }
      ]
    }
  ]
}
```

If the provider fails, times out, or does not return a matching track, the
pipeline keeps the constant-velocity fallback for that object.
