#!/usr/bin/env python3
"""Focused layer-set sweep using our frozen direction (no Astra tensors).

Evaluates without early-stop so hit-cap / max-8gram / first_repeat_at differentiate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .activations import mlp_write_along_u
from .data import load_jsonl
from .edit import apply_edit_to_model, restore_model, select_top_writers
from .evaluate import summarize_rows
from .generate import generate_dataset, load_model_and_tokenizer


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def score(summary: dict) -> float:
    # Prefer finishing / boxing; then delay first repeat; then lower hit-cap & 8gram severity.
    first = summary.get("mean_first_repeat_at") or 0.0
    return (
        2.0 * summary["finished_clean_rate"]
        + 1.5 * summary["boxed_rate"]
        + 1.0 * summary.get("no_repeat_rate", 0.0)
        + 0.002 * first
        - 1.0 * summary["hit_cap_rate"]
        - 0.05 * summary["mean_max_8gram"]
        - 1.0 * summary["early_repeat_rate"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")

    loc = json.loads((RESULTS / "localization.json").read_text())
    combined = loc["combined"]
    # Our candidates only (from localization), not Astra's assumed 15/19.
    top = [r["layer"] for r in combined[:6]]
    layer_sets = [
        top[:2],                 # 23,22
        top[2:4],                # next pair
        top[4:6],                # next pair
        top[:4],                 # top-4
        [r["layer"] for r in loc["by_gap"] if r["layer"] in (18, 19, 20, 21)][:2] or top[2:4],
    ]
    # Dedup while preserving order
    seen = set()
    uniq = []
    for ls in layer_sets:
        key = tuple(ls)
        if key and key not in seen:
            seen.add(key)
            uniq.append(list(ls))
    layer_sets = uniq
    print("Layer sets to sweep:", layer_sets, flush=True)

    # Load directions from npz
    dirs = np.load(RESULTS / "directions.npz")
    u_all = {}
    for k in dirs.files:
        if k.startswith("u_layer_"):
            u_all[int(k.split("_")[-1])] = dirs[k]

    sweep_problems = load_jsonl(RESULTS / "split_sweep.jsonl")
    model, tok = load_model_and_tokenizer(args.model, device=device)

    # Base baseline (no edit), no early-stop
    base_path = RESULTS / "sweep_layers_base.jsonl"
    if base_path.exists():
        base_path.unlink()
    print("=== BASE (no early-stop) ===", flush=True)
    base_rows = generate_dataset(
        model, tok, sweep_problems, base_path,
        max_new_tokens=args.max_new, seed=args.seed, device=device,
        early_repeat_threshold=3, stop_on_repeat=False,
    )
    base_sum = summarize_rows(base_rows)
    print("base", base_sum, "score", score(base_sum), flush=True)

    results = [{"tag": "base", "layers": [], "alpha": None, "k": None, "summary": base_sum, "score": score(base_sum)}]
    best = None

    for layers in layer_sets:
        missing = [L for L in layers if L not in u_all]
        if missing:
            print(f"skip {layers}: missing u for {missing}", flush=True)
            continue
        s = mlp_write_along_u(model, {L: u_all[L] for L in layers})
        edits = [
            {"layer": L, "u": u_all[L], "cols": select_top_writers(s[L], k=args.k), "alpha": args.alpha}
            for L in layers
        ]
        tag = f"a{args.alpha}_k{args.k}_L{'-'.join(map(str, layers))}"
        print(f"=== EDIT {tag} ===", flush=True)
        backups = apply_edit_to_model(model, edits)
        out = RESULTS / f"sweep_layers_{tag}.jsonl"
        if out.exists():
            out.unlink()
        rows = generate_dataset(
            model, tok, sweep_problems, out,
            max_new_tokens=args.max_new, seed=args.seed, device=device,
            early_repeat_threshold=3, stop_on_repeat=False,
        )
        summary = summarize_rows(rows)
        sc = score(summary)
        entry = {
            "tag": tag,
            "layers": layers,
            "alpha": args.alpha,
            "k": args.k,
            "summary": summary,
            "score": sc,
            "edits": [
                {"layer": e["layer"], "alpha": e["alpha"], "cols": e["cols"].tolist(), "u": e["u"].tolist()}
                for e in edits
            ],
        }
        results.append({k: entry[k] for k in entry if k != "edits"})
        print(f"{tag}: score={sc:.3f} {summary}", flush=True)
        restore_model(model, backups)
        if best is None or sc > best["score"]:
            best = entry

    assert best is not None
    # Also try a second α on the winning layer set if it beats base
    if best["tag"] != "base":
        for alpha in (6.0, 12.0):
            if alpha == args.alpha:
                continue
            layers = best["layers"]
            s = mlp_write_along_u(model, {L: u_all[L] for L in layers})
            edits = [
                {"layer": L, "u": u_all[L], "cols": select_top_writers(s[L], k=args.k), "alpha": alpha}
                for L in layers
            ]
            tag = f"a{alpha}_k{args.k}_L{'-'.join(map(str, layers))}"
            print(f"=== REFINE {tag} ===", flush=True)
            backups = apply_edit_to_model(model, edits)
            out = RESULTS / f"sweep_layers_{tag}.jsonl"
            if out.exists():
                out.unlink()
            rows = generate_dataset(
                model, tok, sweep_problems, out,
                max_new_tokens=args.max_new, seed=args.seed, device=device,
                early_repeat_threshold=3, stop_on_repeat=False,
            )
            summary = summarize_rows(rows)
            sc = score(summary)
            entry = {
                "tag": tag,
                "layers": layers,
                "alpha": alpha,
                "k": args.k,
                "summary": summary,
                "score": sc,
                "edits": [
                    {"layer": e["layer"], "alpha": e["alpha"], "cols": e["cols"].tolist(), "u": e["u"].tolist()}
                    for e in edits
                ],
            }
            results.append({k: entry[k] for k in entry if k != "edits"})
            print(f"{tag}: score={sc:.3f} {summary}", flush=True)
            restore_model(model, backups)
            if sc > best["score"]:
                best = entry

    report = {
        "base": base_sum,
        "results": results,
        "best": {k: best[k] for k in best if k != "edits"},
        "beats_base": best["score"] > score(base_sum),
        "protocol": {
            "stop_on_repeat": False,
            "max_new_tokens": args.max_new,
            "alpha_default": args.alpha,
            "k": args.k,
            "n_sweep": len(sweep_problems),
            "device": device,
        },
    }
    (RESULTS / "layer_sweep_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    if "edits" in best:
        from .edit import save_edit_spec

        save_edit_spec(
            RESULTS / "frozen_edit.json",
            {
                "model_id": args.model,
                "method": "early_late_diff_of_means_top_writers",
                "edits": best["edits"],
                "meta": report["best"],
            },
        )
    print("BEST", report["best"], "beats_base", report["beats_base"], flush=True)


if __name__ == "__main__":
    main()
