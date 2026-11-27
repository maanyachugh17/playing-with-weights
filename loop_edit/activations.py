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


def _slice_mean(h: torch.Tensor, start: int, end: int) -> np.ndarray:
    start = max(0, min(start, h.shape[0]))
    end = max(start + 1, min(end, h.shape[0]))
    return h[start:end].mean(0).numpy()


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
    prompt_len: int | None = None,
    mode: str = "full_seq",
    first_repeat_at: int | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Teacher-force full text; return per-layer (early_mean, late_mean) over residual stream.

    Modes:
      - full_seq: first/last windows of the entire teacher-forced sequence (legacy).
      - gen_early_late: first/last windows of the *generation* span only (excludes prompt).
      - preloop_loop: window just before first 8-gram repeat vs window in the looping tail.
      - think_close: if ``</think>`` appears in generation, late=window before close token,
        early=earlier productive span (CLC-flavored when finishes exist).
    """
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
        seq_len = int(enc["input_ids"].shape[1])
        if prompt_len is None:
            prompt_len = 0
        prompt_len = max(0, min(int(prompt_len), seq_len - 1))
        gen_len = max(seq_len - prompt_len, 1)

        # Optional </think> index within generation (token ids).
        close_rel: int | None = None
        try:
            close_ids = tokenizer.encode("</think>", add_special_tokens=False)
            ids = enc["input_ids"][0].tolist()
            gen_ids = ids[prompt_len:]
            if close_ids and len(gen_ids) >= len(close_ids):
                n = len(close_ids)
                for i in range(len(gen_ids) - n + 1):
                    if gen_ids[i : i + n] == close_ids:
                        close_rel = i
                        break
        except Exception:
            close_rel = None

        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for idx in layers:
            h = caches[idx][0].float().cpu()  # (seq, d)
            seq = h.shape[0]
            pl = min(prompt_len, seq - 1)

            if mode == "full_seq":
                if seq < early_window + late_window:
                    mid = max(seq // 2, 1)
                    early = _slice_mean(h, 0, mid)
                    late = _slice_mean(h, mid, seq)
                else:
                    early = _slice_mean(h, 0, early_window)
                    late = _slice_mean(h, seq - late_window, seq)
            elif mode == "gen_early_late":
                if gen_len < early_window + late_window:
                    mid = pl + max(gen_len // 2, 1)
                    early = _slice_mean(h, pl, mid)
                    late = _slice_mean(h, mid, seq)
                else:
                    early = _slice_mean(h, pl, pl + early_window)
                    late = _slice_mean(h, seq - late_window, seq)
            elif mode == "preloop_loop":
                # first_repeat_at is relative to generation tokens.
                fr = first_repeat_at
                if fr is None or fr < max(early_window // 2, 16):
                    # Fall back to gen early/late when loop onset is unknown/too early.
                    if gen_len < early_window + late_window:
                        mid = pl + max(gen_len // 2, 1)
                        early = _slice_mean(h, pl, mid)
                        late = _slice_mean(h, mid, seq)
                    else:
                        early = _slice_mean(h, pl, pl + early_window)
                        late = _slice_mean(h, seq - late_window, seq)
                else:
                    abs_fr = pl + int(fr)
                    early_end = abs_fr
                    early_start = max(pl, early_end - early_window)
                    late_start = min(abs_fr, seq - 1)
                    late_end = min(seq, late_start + late_window)
                    # Prefer the true looping tail when long enough.
                    if seq - abs_fr >= late_window:
                        late_start = seq - late_window
                        late_end = seq
                    early = _slice_mean(h, early_start, early_end)
                    late = _slice_mean(h, late_start, late_end)
            elif mode == "think_close":
                if close_rel is not None and close_rel > early_window:
                    abs_close = pl + close_rel
                    # CLC-flavored: late ≈ states approaching </think>; early ≈ earlier gen.
                    late = _slice_mean(h, max(pl, abs_close - late_window), abs_close)
                    early = _slice_mean(h, pl, pl + early_window)
                    # Flip polarity later: for think_close we want u toward *loop*, so if
                    # this row finished, treat close-window as the "good" state (early here)
                    # and use a mid/late looping proxy when available. For finished rows
                    # alone this mode builds finish-ward; caller may invert.
                else:
                    # No close tag — use gen early/late (loopward).
                    if gen_len < early_window + late_window:
                        mid = pl + max(gen_len // 2, 1)
                        early = _slice_mean(h, pl, mid)
                        late = _slice_mean(h, mid, seq)
                    else:
                        early = _slice_mean(h, pl, pl + early_window)
                        late = _slice_mean(h, seq - late_window, seq)
            else:
                raise ValueError(f"Unknown activation window mode: {mode}")
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
    mode: str = "gen_early_late",
) -> dict[str, Any]:
    """Build contrast from early vs late windows inside our own generations.

    Default mode ``gen_early_late`` excludes the prompt (unlike legacy ``full_seq``).
    Early ≈ productive reasoning; late ≈ looping tail. Does not use external edited tensors.
    """
    layer_mods = get_layers(model)
    all_layers = list(range(len(layer_mods)))
    early_acc: dict[int, list[np.ndarray]] = defaultdict(list)
    late_acc: dict[int, list[np.ndarray]] = defaultdict(list)

    for row in rows:
        prompt = format_prompt_fn(tokenizer, row["prompt"], enable_thinking=True)
        completion = row.get("text") or ""
        full = prompt + completion
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        metrics = row.get("metrics") or {}
        first_rep = metrics.get("first_repeat_at")
        windows = forward_hidden_windows(
            model,
            tokenizer,
            full,
            layers=all_layers,
            early_window=early_window,
            late_window=late_window,
            max_length=max_length,
            device=device,
            prompt_len=len(prompt_ids),
            mode=mode,
            first_repeat_at=first_rep if isinstance(first_rep, int) else None,
        )
        for idx, (e, l) in windows.items():
            early_acc[idx].append(e)
            late_acc[idx].append(l)

    h_early = {i: np.stack(v, 0) for i, v in early_acc.items()}
    h_late = {i: np.stack(v, 0) for i, v in late_acc.items()}
    return {
        "h_early": h_early,
        "h_late": h_late,
        "n_layers": len(layer_mods),
        "n": len(rows),
        "mode": mode,
        "early_window": early_window,
        "late_window": late_window,
    }


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
