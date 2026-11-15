#!/usr/bin/env python3
"""Build our sparse gate_proj mask via |W|*activation scores (Wanda-style).

Produces packbits_zstd masks compatible with deleting-weights apply_mask.py.
This is the Phase-2 method that produced results/our_mask/ in the CPU smoke run.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import zstandard as zstd


RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target-delete", type=float, default=0.001)
    ap.add_argument("--calib-n", type=int, default=12)
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # Import train helpers for eval only
    from train_mask import eval_base_vs_masked, load_gsm8k, reward_fn, extract_boxed, generate_completion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, trust_remote_code=True).to(device)
    data = load_gsm8k(args.calib_n + args.eval_n, seed=args.seed)
    calib, held = data[: args.calib_n], data[args.calib_n :]

    print("=== Base eval ===", flush=True)
    base = eval_base_vs_masked(model, tok, held, device=device, max_new=args.max_new)
    print(base, flush=True)

    gates = [(n, m) for n, m in model.named_modules() if n.endswith("mlp.gate_proj") and isinstance(m, nn.Linear)]
    acts = {n: None for n, _ in gates}
    hooks = []

    def make_hook(n):
        def hook(mod, inp, out):
            x = inp[0].detach()
            v = x.abs().mean(dim=tuple(range(x.ndim - 1))).cpu()
            acts[n] = v if acts[n] is None else acts[n] + v
        return hook

    for n, m in gates:
        hooks.append(m.register_forward_hook(make_hook(n)))
    model.eval()
    for ex in calib:
        prompt = f"Question: {ex['q']}\nAnswer: The answer is \\boxed{{{ex['a']}}}."
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            model(**enc)
    for h in hooks:
        h.remove()

    scores = []
    for n, m in gates:
        W = m.weight.detach().float().cpu()
        a = acts[n] if acts[n] is not None else torch.ones(W.shape[1])
        a = a / (a.mean() + 1e-8)
        scores.append((W.abs() * a.unsqueeze(0)).reshape(-1))
    cat = torch.cat(scores)
    n_del = max(1, int(args.target_delete * cat.numel()))
    thresh = torch.kthvalue(cat, n_del).values.item()

    keep_maps = {}
    deleted = total = 0
    for (n, m), score in zip(gates, scores):
        keep = score.view_as(m.weight.cpu()) > thresh
        keep_maps[f"{n}.weight"] = keep
        deleted += int((~keep).sum())
        total += keep.numel()
        with torch.no_grad():
            m.weight.mul_(keep.to(device=m.weight.device, dtype=m.weight.dtype))

    print("=== Masked eval ===", flush=True)
    masked = eval_base_vs_masked(model, tok, held, device=device, max_new=args.max_new)
    print(masked, flush=True)

    out_dir = RESULTS / "our_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    cctx = zstd.ZstdCompressor(level=3)
    for name, keep in keep_maps.items():
        k = keep.numpy().astype(np.uint8)
        packed = np.packbits(k.reshape(-1))
        compressed = cctx.compress(packed.tobytes())
        tensors[f"{name}.shape"] = torch.tensor(list(keep.shape), dtype=torch.int64)
        tensors[f"{name}.packed"] = torch.from_numpy(np.frombuffer(compressed, dtype=np.uint8).copy())
    save_file(tensors, str(out_dir / "model.safetensors"))
    meta = {
        "format": "packbits_zstd",
        "deleted": deleted,
        "total_masked_params": total,
        "deleted_fraction_of_masked": deleted / max(total, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "model": args.model,
        "device": device,
        "method": "activation_magnitude_sparse_delete_on_gate_proj",
        "target_delete": args.target_delete,
        "calibration_n": args.calib_n,
        "elapsed_sec": time.time() - t0,
        "mask": meta,
        "deleted_fraction_of_model": deleted / max(n_params, 1),
        "base_eval": base,
        "masked_eval": masked,
        "mask_dir": str(out_dir),
    }
    (RESULTS / "phase2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
