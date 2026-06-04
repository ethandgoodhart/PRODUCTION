#!/usr/bin/env python3
"""HTTP provider boundary for Trajectron++ object futures.

This process is intentionally separate from the driving loop. The official
Trajectron++ codebase is Python/dependency sensitive and checkpoint-driven; if
the configured checkpoint is not present, this server reports unavailable
instead of silently returning heuristic trajectories as "Trajectron++".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flask import Flask, jsonify, request


def find_checkpoint(model_dir: Path, checkpoint: int) -> Path | None:
    candidate = model_dir / f"model_registrar-{checkpoint}.pt"
    if candidate.exists():
        return candidate
    matches = sorted(model_dir.glob("model_registrar-*.pt"))
    return matches[-1] if matches else None


class TrajectronPPProvider:
    def __init__(self, repo: Path, model_dir: Path, checkpoint: int):
        self.repo = repo
        self.model_dir = model_dir
        self.checkpoint = int(checkpoint)
        self.config_file = model_dir / "config.json"
        self.checkpoint_file = find_checkpoint(model_dir, self.checkpoint)
        self.ready = False
        self.error = ""
        self._validate()

    def _validate(self) -> None:
        if not self.repo.exists():
            self.error = f"Trajectron++ repo not found: {self.repo}"
            return
        if not self.config_file.exists():
            self.error = f"Trajectron++ config not found: {self.config_file}"
            return
        if self.checkpoint_file is None:
            self.error = (
                f"Trajectron++ checkpoint model_registrar-{self.checkpoint}.pt "
                f"not found in {self.model_dir}"
            )
            return

        # The official project pins Python 3.6-era packages. Do the import
        # checks here so startup tells us what is missing before the driving
        # loop routes prediction traffic to this provider.
        missing = []
        for module in ("ncls", "sklearn", "pyquaternion", "dill"):
            try:
                __import__(module)
            except Exception as exc:  # pragma: no cover - diagnostic path
                missing.append(f"{module}: {exc}")
        if missing:
            self.error = "Trajectron++ dependencies missing: " + "; ".join(missing)
            return

        # A real checkpoint is available and dependency imports are present.
        # Loading/inference is intentionally gated here until the target
        # checkpoint is verified against our caddy.object_prediction.v1 scene
        # adapter. Failing closed avoids mislabeled behavior predictions.
        self.error = (
            "Trajectron++ checkpoint found, but online scene adapter is not "
            "enabled for this checkpoint yet"
        )

    def status(self) -> dict:
        return {
            "provider": "trajectronpp",
            "ready": self.ready,
            "error": self.error,
            "repo": str(self.repo),
            "model_dir": str(self.model_dir),
            "checkpoint": self.checkpoint,
            "checkpoint_file": str(self.checkpoint_file) if self.checkpoint_file else None,
        }

    def predict(self, payload: dict) -> tuple[dict, int]:
        if not self.ready:
            return {
                "provider": "trajectronpp",
                "ready": False,
                "error": self.error,
                "agents": [],
            }, 503
        return {
            "provider": "trajectronpp",
            "agents": [],
        }, 501


def build_app(provider: TrajectronPPProvider) -> Flask:
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
            return jsonify({
                "provider": "trajectronpp",
                "ready": False,
                "error": f"bad JSON: {exc}",
                "agents": [],
            }), 400
        body, code = provider.predict(payload or {})
        body["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return jsonify(body), code

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Trajectron++ object future provider")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--repo", type=Path,
                   default=Path(".cache/third_party/Trajectron-plus-plus"))
    p.add_argument("--model-dir", type=Path,
                   default=Path(".cache/third_party/Trajectron-plus-plus/experiments/nuScenes/models/int_ee_me"))
    p.add_argument("--checkpoint", type=int, default=12)
    args = p.parse_args()

    provider = TrajectronPPProvider(
        repo=args.repo.resolve(),
        model_dir=args.model_dir.resolve(),
        checkpoint=args.checkpoint,
    )
    print(json.dumps(provider.status()), flush=True)
    app = build_app(provider)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
