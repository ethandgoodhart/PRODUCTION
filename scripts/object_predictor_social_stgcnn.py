#!/usr/bin/env python3
"""Social-STGCNN/Social-NCE object trajectory provider.

This is a real neural trajectory predictor sidecar using the pretrained
Social-NCE Social-STGCNN weights. It consumes the caddy.object_prediction.v1
JSON contract and returns multimodal futures in local ground-plane meters.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request


def normalized_laplacian(a: np.ndarray) -> np.ndarray:
    deg = a.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 1e-9
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def build_graph(obs_abs: np.ndarray, obs_rel: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, num_agents, _ = obs_abs.shape
    v = np.asarray(obs_rel, dtype=np.float32)
    a = np.zeros((seq_len, num_agents, num_agents), dtype=np.float32)
    for t in range(seq_len):
        step = obs_rel[t]
        for i in range(num_agents):
            a[t, i, i] = 1.0
            for j in range(i + 1, num_agents):
                dist = float(np.linalg.norm(step[i] - step[j]))
                weight = 0.0 if dist <= 1e-9 else 1.0 / dist
                a[t, i, j] = weight
                a[t, j, i] = weight
        a[t] = normalized_laplacian(a[t])
    return torch.from_numpy(v), torch.from_numpy(a)


def interp_history(history: list, target_times: np.ndarray) -> np.ndarray | None:
    pts = []
    for item in history:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            t, x, y = float(item[0]), float(item[1]), float(item[2])
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(x) and math.isfinite(y):
            pts.append((t, x, y))
    if len(pts) < 2:
        return None
    pts.sort(key=lambda r: r[0])
    ts = np.asarray([p[0] for p in pts], dtype=np.float64)
    xs = np.asarray([p[1] for p in pts], dtype=np.float64)
    ys = np.asarray([p[2] for p in pts], dtype=np.float64)
    start_t = max(float(target_times[0]), float(ts[0]))
    if start_t >= -0.2:
        return None
    target_times = np.linspace(start_t, 0.0, len(target_times))
    xq = np.interp(target_times, ts, xs)
    yq = np.interp(target_times, ts, ys)
    return np.stack([xq, yq], axis=-1).astype(np.float32)


class SocialSTGCNNProvider:
    def __init__(self, repo: Path, checkpoint_dir: Path, device: str = "cpu"):
        self.repo = repo
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)
        self.ready = False
        self.error = ""
        self.model = None
        self.args = None
        self.dt_s = 0.4
        self._load()

    def _load(self) -> None:
        try:
            sys.path.insert(0, str(self.repo))
            from model import social_stgcnn  # type: ignore

            args_path = self.checkpoint_dir / "args.pkl"
            ckpt_path = self.checkpoint_dir / "val_best.pth"
            if not args_path.exists():
                raise FileNotFoundError(args_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(ckpt_path)
            with args_path.open("rb") as f:
                self.args = pickle.load(f)
            model = social_stgcnn(
                n_stgcnn=self.args.n_stgcnn,
                n_txpcnn=self.args.n_txpcnn,
                output_feat=self.args.output_size,
                seq_len=self.args.obs_seq_len,
                kernel_size=self.args.kernel_size,
                pred_seq_len=self.args.pred_seq_len,
            ).to(self.device)
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.ready = True
        except Exception as exc:
            self.ready = False
            self.error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict:
        return {
            "provider": "social_stgcnn_snce",
            "ready": bool(self.ready),
            "error": self.error,
            "repo": str(self.repo),
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": str(self.device),
        }

    def _predict_arrays(self, obs_abs: np.ndarray, samples: int) -> np.ndarray:
        obs_rel = np.zeros_like(obs_abs, dtype=np.float32)
        obs_rel[1:] = obs_abs[1:] - obs_abs[:-1]
        v_obs, a_obs = build_graph(obs_abs, obs_rel)
        v_in = v_obs.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        a_in = a_obs.to(self.device)
        with torch.no_grad():
            v_pred, _ = self.model(v_in, a_in.squeeze())
            v_pred = v_pred.detach().permute(0, 2, 3, 1).squeeze(0)
            mean = v_pred[:, :, 0:2]
            sx = torch.exp(torch.clamp(v_pred[:, :, 2], -4.0, 3.0))
            sy = torch.exp(torch.clamp(v_pred[:, :, 3], -4.0, 3.0))
            corr = torch.tanh(v_pred[:, :, 4])
            cov = torch.zeros(v_pred.shape[0], v_pred.shape[1], 2, 2, device=self.device)
            cov[:, :, 0, 0] = sx * sx
            cov[:, :, 1, 1] = sy * sy
            cov[:, :, 0, 1] = corr * sx * sy
            cov[:, :, 1, 0] = corr * sx * sy
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            rel_samples = [mean]
            for _ in range(max(0, samples - 1)):
                rel_samples.append(dist.sample())
            rel = torch.stack(rel_samples, dim=0).cpu().numpy()
        last = obs_abs[-1][None, None, :, :]
        return np.cumsum(rel, axis=1) + last

    def predict(self, payload: dict) -> tuple[dict, int]:
        if not self.ready:
            return {"provider": "social_stgcnn_snce", "ready": False, "error": self.error, "agents": []}, 503
        agents = [a for a in payload.get("agents", []) if isinstance(a, dict) and a.get("track_id") is not None]
        if not agents:
            return {"provider": "social_stgcnn_snce", "agents": []}, 200
        obs_len = int(self.args.obs_seq_len)
        pred_len = int(self.args.pred_seq_len)
        target_times = np.linspace(-(obs_len - 1) * self.dt_s, 0.0, obs_len)
        valid_agents = []
        obs = []
        for agent in agents[:32]:
            arr = interp_history(agent.get("history_m", []), target_times)
            if arr is None:
                continue
            valid_agents.append(agent)
            obs.append(arr)
        if not obs:
            return {"provider": "social_stgcnn_snce", "agents": []}, 200
        obs_abs = np.stack(obs, axis=1).astype(np.float32)
        samples = int(np.clip(payload.get("samples", 3), 1, 6))
        pred_abs = self._predict_arrays(obs_abs, samples=samples)
        horizon_s = float(payload.get("horizon_s", 4.0))
        step_s = float(payload.get("step_s", 0.5))
        requested = np.arange(step_s, horizon_s + 1e-6, step_s)
        model_times = np.arange(1, pred_len + 1, dtype=np.float32) * self.dt_s
        out_agents = []
        for agent_i, agent in enumerate(valid_agents):
            modes = []
            for sample_i in range(pred_abs.shape[0]):
                xs = np.interp(requested, model_times, pred_abs[sample_i, :, agent_i, 0])
                ys = np.interp(requested, model_times, pred_abs[sample_i, :, agent_i, 1])
                future = [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]
                modes.append({
                    "prob": round(1.0 / pred_abs.shape[0], 4),
                    "future_m": future,
                })
            out_agents.append({"track_id": agent.get("track_id"), "modes": modes})
        return {"provider": "social_stgcnn_snce", "agents": out_agents}, 200


def build_app(provider: SocialSTGCNNProvider) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        status = provider.status()
        return jsonify(status), (200 if status["ready"] else 503)

    @app.post("/predict")
    def predict():
        t0 = time.perf_counter()
        try:
            payload = request.get_json(force=True)
        except Exception as exc:
            return jsonify({"provider": "social_stgcnn_snce", "error": f"bad JSON: {exc}", "agents": []}), 400
        body, code = provider.predict(payload or {})
        body["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return jsonify(body), code

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Social-STGCNN object future provider")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--repo", type=Path, default=Path(".cache/third_party/social-nce-stgcnn"))
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path(".cache/third_party/social-nce-stgcnn/checkpoint-snce/snce-social-stgcnn-univ"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    provider = SocialSTGCNNProvider(args.repo.resolve(), args.checkpoint_dir.resolve(), args.device)
    print(json.dumps(provider.status()), flush=True)
    build_app(provider).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
