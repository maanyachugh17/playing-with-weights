"""Collect residual-stream and MLP-write activations for contrast sets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


def get_layers(model) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise AttributeError("Could not find transformer layers on model")


def get_down_proj(layer) -> torch.nn.Linear:
    return layer.mlp.down_proj


@torch.no_grad()
def forward_hidden_windows(
    model,
    tokenizer,
    prompt_and_completion: str,
    layers: list[int],
    early_window: int = 64,
    late_window: int = 64,
    max_length: int = 1024,
    device: str = "cpu",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Teacher-force full text; return per-layer (early_mean, late_mean) over residual stream."""
    layer_mods = get_layers(model)
    caches: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            caches[idx] = h.detach()

        return hook

    for idx in layers:
        hooks.append(layer_mods[idx].register_forward_hook(make_hook(idx)))
    try:
        enc = tokenizer(
            prompt_and_completion,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc, use_cache=False)
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for idx in layers:
            h = caches[idx][0].float().cpu()  # (seq, d)
            seq = h.shape[0]
            if seq < early_window + late_window:
                # Split halves
                mid = max(seq // 2, 1)
                early = h[:mid].mean(0).numpy()
                late = h[mid:].mean(0).numpy()
            else:
                early = h[:early_window].mean(0).numpy()
                late = h[-late_window:].mean(0).numpy()
            out[idx] = (early, late)
        return out
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def collect_early_late_from_generations(
    model,
    tokenizer,
    rows: list[dict],
    format_prompt_fn,
    device: str = "cpu",
    max_length: int = 1024,
    early_window: int = 64,
    late_window: int = 64,
) -> dict[str, Any]:
    """Build contrast from early vs late windows inside our own generations.

    Early ≈ productive reasoning; late ≈ looping tail. This does not use external
    edited tensors and works when few completions finish cleanly.
    """
    layer_mods = get_layers(model)
    all_layers = list(range(len(layer_mods)))
    early_acc: dict[int, list[np.ndarray]] = defaultdict(list)
    late_acc: dict[int, list[np.ndarray]] = defaultdict(list)

    for row in rows:
        prompt = format_prompt_fn(tokenizer, row["prompt"], enable_thinking=True)
        full = prompt + (row.get("text") or "")
        windows = forward_hidden_windows(
            model,
            tokenizer,
            full,
            layers=all_layers,
            early_window=early_window,
            late_window=late_window,
            max_length=max_length,
            device=device,
        )
        for idx, (e, l) in windows.items():
            early_acc[idx].append(e)
            late_acc[idx].append(l)

    h_early = {i: np.stack(v, 0) for i, v in early_acc.items()}
    h_late = {i: np.stack(v, 0) for i, v in late_acc.items()}
    return {"h_early": h_early, "h_late": h_late, "n_layers": len(layer_mods), "n": len(rows)}


@torch.no_grad()
def mlp_write_along_u(
    model,
    u_by_layer: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """For each layer, s = W.T @ u where W is down_proj weight (d, n)."""
    layer_mods = get_layers(model)
    out = {}
    for idx, u in u_by_layer.items():
        W = get_down_proj(layer_mods[idx]).weight.detach().float().cpu().numpy()
        u64 = u.astype(np.float64)
        u64 = u64 / (np.linalg.norm(u64) + 1e-12)
        out[idx] = W.T @ u64
    return out


@torch.no_grad()
def attribute_writer_mass(model, u_by_layer: dict[int, np.ndarray]) -> list[dict]:
    """Rank layers by positive write mass along u (sum of relu(s))."""
    s_by = mlp_write_along_u(model, u_by_layer)
    rows = []
    for idx, s in s_by.items():
        pos = np.maximum(s, 0.0)
        rows.append(
            {
                "layer": int(idx),
                "pos_mass": float(pos.sum()),
                "max_s": float(s.max()),
                "top64_mass": float(np.sort(pos)[-64:].sum()),
            }
        )
    rows.sort(key=lambda r: -r["pos_mass"])
    return rows
