#!/usr/bin/env python3
"""Phase 1 end-to-end: diagnose → contrast → direction → localize → sweep → freeze → smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .activations import (
    attribute_writer_mass,
    collect_early_late_from_generations,
    mlp_write_along_u,
)
from .data import build_splits, save_jsonl
from .edit import (
    apply_edit_to_model,
    closure_direction,
    restore_model,
    save_edit_spec,
    select_top_writers,
)
from .evaluate import summarize_rows
from .generate import format_prompt, generate_dataset, load_model_and_tokenizer
from .metrics import GenMetrics


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def candidate_score(summary: dict) -> float:
    """Higher is better. Emphasize exiting loops and producing boxed answers."""
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


def run(args: argparse.Namespace) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = build_splits(
        n_diagnose=args.n_diagnose,
        n_sweep=args.n_sweep,
        n_smoke=args.n_smoke,
        seed=args.seed,
    )
    for name, rows in splits.items():
        save_jsonl(RESULTS / f"split_{name}.jsonl", rows)

    print(f"Loading {args.model} on {device}...", flush=True)
    model, tok = load_model_and_tokenizer(args.model, device=device)

    # --- 1. Diagnose ---
    print("=== Diagnose on practice prompts ===", flush=True)
    diag_path = RESULTS / "diagnose_generations.jsonl"
    if args.reuse_diagnose and diag_path.exists():
        from .data import load_jsonl

        diag_rows = load_jsonl(diag_path)
        print(f"Reusing {len(diag_rows)} diagnose generations", flush=True)
    else:
        if diag_path.exists():
            diag_path.unlink()
        diag_rows = generate_dataset(
            model,
            tok,
            splits["diagnose"],
            out_path=diag_path,
            max_new_tokens=args.diag_max_tokens,
            seed=args.seed,
            device=device,
            early_repeat_threshold=args.repeat_threshold,
        )
    diag_summary = summarize_rows(diag_rows)
    (RESULTS / "diagnose_summary.json").write_text(json.dumps(diag_summary, indent=2), encoding="utf-8")
    print("Diagnose:", json.dumps(diag_summary), flush=True)

    # Prefer looping rows for early/late contrast; fall back to all.
    loop_rows = [r for r in diag_rows if GenMetrics(**r["metrics"]).early_repeat or GenMetrics(**r["metrics"]).hit_cap]
    contrast_rows = (loop_rows or diag_rows)[: args.max_contrast]
    print(f"Contrast generations: {len(contrast_rows)}", flush=True)

    # --- 2/3. Early vs late activations + direction ---
    print("=== Collecting early/late activations (our generations) ===", flush=True)
    acts = collect_early_late_from_generations(
        model,
        tok,
        contrast_rows,
        format_prompt_fn=format_prompt,
        device=device,
        max_length=args.act_max_length,
        early_window=args.window,
        late_window=args.window,
    )
    # Direction u = normalize(late - early): points toward the loop state.
    u_all = {}
    gap_rows = []
    for L in acts["h_late"]:
        u = closure_direction(acts["h_late"][L], acts["h_early"][L])
        gap = float(np.linalg.norm(acts["h_late"][L].mean(0) - acts["h_early"][L].mean(0)))
        u_all[L] = u
        gap_rows.append({"layer": int(L), "gap": gap})
    gap_rows.sort(key=lambda r: -r["gap"])

    writer_rows = attribute_writer_mass(model, u_all)
    # Combine: prefer layers with both large residual gap and large positive write mass.
    gap_map = {r["layer"]: r["gap"] for r in gap_rows}
    combined = []
    for r in writer_rows:
        combined.append(
            {
                "layer": r["layer"],
                "gap": gap_map.get(r["layer"], 0.0),
                "pos_mass": r["pos_mass"],
                "top64_mass": r["top64_mass"],
                "score": 0.5 * gap_map.get(r["layer"], 0.0) + 0.5 * r["pos_mass"] / (writer_rows[0]["pos_mass"] + 1e-9),
            }
        )
    combined.sort(key=lambda r: -r["score"])
    localization = combined[: max(args.top_layers, 8)]
    (RESULTS / "localization.json").write_text(
        json.dumps(
            {
                "by_gap": gap_rows[:10],
                "by_writer_mass": writer_rows[:10],
                "combined": [{k: v for k, v in r.items()} for r in localization],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Top combined layers:", [{k: localization[i][k] for k in ('layer','gap','pos_mass','score')} for i in range(min(6, len(localization)))], flush=True)

    edit_layer_candidates = [r["layer"] for r in localization[: args.edit_layers]]
    # Also try a mid-layer pair from writer mass if combined is all near the end.
    if args.include_mid_layers:
        mid = [r["layer"] for r in writer_rows if 10 <= r["layer"] <= 19][:2]
        for L in mid:
            if L not in edit_layer_candidates:
                edit_layer_candidates.append(L)
        edit_layer_candidates = edit_layer_candidates[: args.edit_layers]

    u_by_layer = {L: u_all[L] for L in edit_layer_candidates}
    s_by_layer = mlp_write_along_u(model, u_by_layer)
    np.savez(
        RESULTS / "directions.npz",
        **{f"u_layer_{L}": u_all[L] for L in sorted(u_all)},
        **{f"s_layer_{L}": s_by_layer[L] for L in s_by_layer},
    )

    # --- 5. Sweep ---
    print("=== Sweep α / k / layer-sets ===", flush=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    layer_sets = [edit_layer_candidates]
    # Optional alternate: top-2 by writer mass alone
    alt = [r["layer"] for r in writer_rows[: args.edit_layers]]
    if alt != edit_layer_candidates:
        layer_sets.append(alt)

    sweep_results = []
    best = None
    for layers in layer_sets:
        u_use = {L: u_all[L] for L in layers}
        s_use = mlp_write_along_u(model, u_use)
        for alpha in alphas:
            for k in ks:
                edits = []
                for L in layers:
                    cols = select_top_writers(s_use[L], k=k)
                    edits.append({"layer": L, "u": u_use[L], "cols": cols, "alpha": alpha})
                backups = apply_edit_to_model(model, edits)
                out_path = RESULTS / f"sweep_a{alpha}_k{k}_L{'-'.join(map(str, layers))}.jsonl"
                if out_path.exists():
                    out_path.unlink()
                rows = generate_dataset(
                    model,
                    tok,
                    splits["sweep"],
                    out_path=out_path,
                    max_new_tokens=args.sweep_max_tokens,
                    seed=args.seed + 100,
                    device=device,
                    early_repeat_threshold=args.repeat_threshold,
                    stop_on_repeat=False,
                )
                summary = summarize_rows(rows)
                score = candidate_score(summary)
                entry = {"alpha": alpha, "k": k, "layers": layers, "summary": summary, "score": score}
                sweep_results.append(entry)
                print(f"sweep α={alpha} k={k} L={layers}: score={score:.3f} {summary}", flush=True)
                restore_model(model, backups)
                if best is None or score > best["score"]:
                    best = entry
                    best["edits"] = [
                        {
                            "layer": e["layer"],
                            "u": e["u"],
                            "cols": e["cols"].tolist() if hasattr(e["cols"], "tolist") else list(e["cols"]),
                            "alpha": e["alpha"],
                        }
                        for e in edits
                    ]

    (RESULTS / "sweep_results.json").write_text(json.dumps(sweep_results, indent=2, default=float), encoding="utf-8")
    assert best is not None
    print("Best candidate:", {k: best[k] for k in ("alpha", "k", "layers", "score")}, flush=True)

    # --- Baseline sweep score (no edit) for comparison ---
    base_sweep_path = RESULTS / "sweep_base.jsonl"
    if base_sweep_path.exists():
        base_sweep_path.unlink()
    base_sweep_rows = generate_dataset(
        model,
        tok,
        splits["sweep"],
        out_path=base_sweep_path,
        max_new_tokens=args.sweep_max_tokens,
        seed=args.seed + 100,
        device=device,
        early_repeat_threshold=args.repeat_threshold,
        stop_on_repeat=False,
    )
    base_sweep_summary = summarize_rows(base_sweep_rows)
    print("Base sweep:", base_sweep_summary, flush=True)

    frozen_edits = best["edits"]
    spec = {
        "model_id": args.model,
        "method": "early_late_diff_of_means_top_writers",
        "edits": frozen_edits,
        "meta": {
            "diagnose": diag_summary,
            "localization_top": localization[:6],
            "best_score": best["score"],
            "best_summary": best["summary"],
            "base_sweep_summary": base_sweep_summary,
            "seed": args.seed,
            "device": device,
        },
    }
    save_edit_spec(RESULTS / "frozen_edit.json", spec)

    # --- 6. Smoke ---
    print("=== Smoke eval: base ===", flush=True)
    base_path = RESULTS / "smoke_base.jsonl"
    if base_path.exists():
        base_path.unlink()
    base_rows = generate_dataset(
        model,
        tok,
        splits["smoke"],
        out_path=base_path,
        max_new_tokens=args.smoke_max_tokens,
        seed=args.seed + 200,
        device=device,
        early_repeat_threshold=args.repeat_threshold,
        stop_on_repeat=False,
    )
    base_summary = summarize_rows(base_rows)

    print("=== Smoke eval: edited ===", flush=True)
    backups = apply_edit_to_model(model, frozen_edits)
    edit_path = RESULTS / "smoke_edited.jsonl"
    if edit_path.exists():
        edit_path.unlink()
    edit_rows = generate_dataset(
        model,
        tok,
        splits["smoke"],
        out_path=edit_path,
        max_new_tokens=args.smoke_max_tokens,
        seed=args.seed + 200,
        device=device,
        early_repeat_threshold=args.repeat_threshold,
        stop_on_repeat=False,
    )
    edit_summary = summarize_rows(edit_rows)
    restore_model(model, backups)

    beats = (
        (edit_summary["early_repeat_rate"] + edit_summary["hit_cap_rate"])
        < (base_summary["early_repeat_rate"] + base_summary["hit_cap_rate"])
    ) or (edit_summary["finished_clean_rate"] > base_summary["finished_clean_rate"]) or (
        edit_summary["boxed_rate"] > base_summary["boxed_rate"]
    ) or (edit_summary["mean_tokens"] < base_summary["mean_tokens"] * 0.85)

    report = {
        "diagnose": diag_summary,
        "localization_top": localization[:6],
        "best": {
            "alpha": best["alpha"],
            "k": best["k"],
            "layers": best["layers"],
            "score": best["score"],
            "sweep_summary": best["summary"],
        },
        "base_sweep": base_sweep_summary,
        "smoke_base": base_summary,
        "smoke_edited": edit_summary,
        "elapsed_sec": time.time() - t_start,
        "beats_base": beats,
    }
    (RESULTS / "phase1_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("=== Phase 1 report ===", flush=True)
    print(json.dumps(report, indent=2, default=float), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description="Derive our own Qwen thinking-loop weight edit")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-diagnose", type=int, default=20)
    p.add_argument("--n-sweep", type=int, default=8)
    p.add_argument("--n-smoke", type=int, default=25)
    p.add_argument("--diag-max-tokens", type=int, default=640)
    p.add_argument("--sweep-max-tokens", type=int, default=640)
    p.add_argument("--smoke-max-tokens", type=int, default=2048)
    p.add_argument("--act-max-length", type=int, default=768)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--max-contrast", type=int, default=8)
    p.add_argument("--top-layers", type=int, default=8)
    p.add_argument("--edit-layers", type=int, default=2)
    p.add_argument("--include-mid-layers", action="store_true", default=True)
    p.add_argument("--alphas", default="4,6,8,12")
    p.add_argument("--ks", default="64,128")
    p.add_argument("--repeat-threshold", type=int, default=3)
    p.add_argument("--reuse-diagnose", action="store_true", default=True)
    p.add_argument("--no-reuse-diagnose", action="store_false", dest="reuse_diagnose")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
