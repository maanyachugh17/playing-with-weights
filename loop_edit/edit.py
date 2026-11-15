"""Rank-one column edit of MLP down_proj: W' = W - α u (s ⊙ m)^T."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .activations import get_down_proj, get_layers


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def closure_direction(h_loop: np.ndarray, h_close: np.ndarray) -> np.ndarray:
    """Unit vector from mean finishing/pre-close state toward mean loop state."""
    loop = np.asarray(h_loop, dtype=np.float64)
    close = np.asarray(h_close, dtype=np.float64)
    if loop.ndim == 2:
        loop = loop.mean(0)
    if close.ndim == 2:
        close = close.mean(0)
    return unit(loop - close)


def select_top_writers(s: np.ndarray, k: int) -> np.ndarray:
    """Columns with largest |write| along u (signed write magnitude)."""
    s = np.asarray(s, dtype=np.float64)
    # Prefer positive writers that reinforce the loop direction.
    idx = np.argsort(-s)[:k]
    return np.sort(idx.astype(np.int64))


def apply_edit_numpy(W: np.ndarray, u: np.ndarray, cols: np.ndarray, alpha: float) -> np.ndarray:
    """W is (d, n); edit only columns in cols."""
    W64 = np.asarray(W, dtype=np.float64)
    u64 = unit(u)
    s = W64.T @ u64
    m = np.zeros_like(s)
    m[np.asarray(cols, dtype=np.int64)] = 1.0
    return W64 - alpha * np.outer(u64, m * s)


def localize_layers(
    h_loop: dict[int, np.ndarray],
    h_finish: dict[int, np.ndarray],
    top_n: int = 4,
) -> list[dict[str, Any]]:
    """Rank layers by ||mean_loop - mean_finish|| (our attribution signal)."""
    scores = []
    for idx in sorted(set(h_loop) & set(h_finish)):
        u = closure_direction(h_loop[idx], h_finish[idx])
        gap = float(np.linalg.norm(h_loop[idx].mean(0) - h_finish[idx].mean(0)))
        scores.append({"layer": int(idx), "gap": gap, "u": u})
    scores.sort(key=lambda r: -r["gap"])
    return scores[:top_n]


@torch.no_grad()
def apply_edit_to_model(
    model,
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply a list of {layer, u, cols, alpha} edits in place. Returns backup tensors."""
    layer_mods = get_layers(model)
    backups = {}
    for ed in edits:
        layer = int(ed["layer"])
        W_mod = get_down_proj(layer_mods[layer])
        key = f"layers.{layer}.mlp.down_proj.weight"
        backups[key] = W_mod.weight.detach().cpu().clone()
        W_np = W_mod.weight.detach().float().cpu().numpy()
        W_new = apply_edit_numpy(W_np, np.asarray(ed["u"]), np.asarray(ed["cols"]), float(ed["alpha"]))
        W_mod.weight.copy_(torch.from_numpy(W_new.astype(np.float32)).to(W_mod.weight.device).to(W_mod.weight.dtype))
    return backups


@torch.no_grad()
def restore_model(model, backups: dict[str, torch.Tensor]) -> None:
    layer_mods = get_layers(model)
    for key, tensor in backups.items():
        # key like layers.N.mlp.down_proj.weight
        parts = key.split(".")
        layer = int(parts[1])
        get_down_proj(layer_mods[layer]).weight.copy_(tensor.to(get_down_proj(layer_mods[layer]).weight.device))


def save_edit_spec(path: Path, spec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy to lists
    serial = {
        "model_id": spec.get("model_id"),
        "method": spec.get("method"),
        "edits": [],
        "meta": spec.get("meta", {}),
    }
    for ed in spec["edits"]:
        serial["edits"].append(
            {
                "layer": int(ed["layer"]),
                "alpha": float(ed["alpha"]),
                "cols": [int(c) for c in ed["cols"]],
                "u": [float(x) for x in np.asarray(ed["u"]).tolist()],
            }
        )
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def load_edit_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
