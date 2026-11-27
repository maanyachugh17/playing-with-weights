#!/usr/bin/env python3
"""Second-pass loop-edit discovery: better contrast windows + broader sweep.

Does **not** copy Astra tensors. Reuses existing practice splits. Freezes a new
edit only if it beats both base and the prior frozen L21–20 edit on the held-out
sweep score (MATH-500 never used for selection).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .activations import (
    attribute_writer_mass,
    collect_early_late_from_generations,
    mlp_write_along_u,
)
from .data import load_jsonl
from .edit import (
    apply_edit_to_model,
    closure_direction,
    load_edit_spec,
    restore_model,
    save_edit_spec,
    select_top_writers,
    unit,
)
from .evaluate import summarize_rows
from .generate import format_prompt, generate_dataset, load_model_and_tokenizer, resolve_device
from .metrics import GenMetrics


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def candidate_score(summary: dict) -> float:
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


def build_u_all(acts: dict) -> tuple[dict[int, np.ndarray], list[dict]]:
    u_all: dict[int, np.ndarray] = {}
    gap_rows = []
    for L in acts["h_late"]:
        u = closure_direction(acts["h_late"][L], acts["h_early"][L])
        gap = float(np.linalg.norm(acts["h_late"][L].mean(0) - acts["h_early"][L].mean(0)))
        u_all[L] = u
        gap_rows.append({"layer": int(L), "gap": gap})
    gap_rows.sort(key=lambda r: -r["gap"])
    return u_all, gap_rows


def shared_direction(u_all: dict[int, np.ndarray], layers: list[int]) -> np.ndarray:
    """Average unit directions over selected layers, then re-normalize."""
    mats = [unit(u_all[L]) for L in layers if L in u_all]
    if not mats:
        raise ValueError("no layers for shared direction")
    return unit(np.mean(np.stack(mats, 0), axis=0))


def make_edits(
    model,
    u_by_layer: dict[int, np.ndarray],
    layers: list[int],
    alpha: float,
    k: int,
    shared_u: np.ndarray | None = None,
) -> list[dict]:
    if shared_u is not None:
        u_use = {L: shared_u for L in layers}
    else:
        u_use = {L: u_by_layer[L] for L in layers}
    s_use = mlp_write_along_u(model, u_use)
    edits = []
    for L in layers:
        cols = select_top_writers(s_use[L], k=k)
        edits.append({"layer": L, "u": u_use[L], "cols": cols, "alpha": alpha})
    return edits


def eval_edits(model, tok, problems, edits, out_path: Path, max_new: int, seed: int, device: str) -> dict:
    if out_path.exists():
        out_path.unlink()
    backups = apply_edit_to_model(model, edits)
    try:
        rows = generate_dataset(
            model,
            tok,
            problems,
            out_path=out_path,
            max_new_tokens=max_new,
            seed=seed,
            device=device,
            early_repeat_threshold=3,
            stop_on_repeat=False,
        )
        return summarize_rows(rows)
    finally:
        restore_model(model, backups)


def run(args: argparse.Namespace) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = resolve_device(args.device)

    # Preserve prior freeze as v1 reference (never overwrite without a winner).
    old_frozen_path = RESULTS / "frozen_edit.json"
    v1_backup = RESULTS / "frozen_edit_v1_L21-20.json"
    if old_frozen_path.exists() and not v1_backup.exists():
        shutil.copy2(old_frozen_path, v1_backup)
        print(f"Backed up prior freeze → {v1_backup}", flush=True)

    diagnose_split = load_jsonl(RESULTS / "split_diagnose.jsonl")
    sweep_split = load_jsonl(RESULTS / "split_sweep.jsonl")
    smoke_split = load_jsonl(RESULTS / "split_smoke.jsonl")
    if args.n_smoke_cap and args.n_smoke_cap < len(smoke_split):
        smoke_split = smoke_split[: args.n_smoke_cap]

    print(f"Loading {args.model} on {device}...", flush=True)
    model, tok = load_model_and_tokenizer(args.model, device=device)

    # --- Diagnose (refresh) ---
    diag_path = RESULTS / "diagnose_generations_v2.jsonl"
    if args.reuse_diagnose and diag_path.exists():
        diag_rows = load_jsonl(diag_path)
        print(f"Reusing {len(diag_rows)} diagnose_v2 generations", flush=True)
    else:
        if diag_path.exists():
            diag_path.unlink()
        print("=== Diagnose v2 ===", flush=True)
        diag_rows = generate_dataset(
            model,
            tok,
            diagnose_split,
            out_path=diag_path,
            max_new_tokens=args.diag_max_tokens,
            seed=args.seed,
            device=device,
            early_repeat_threshold=3,
            stop_on_repeat=True,
        )
    diag_summary = summarize_rows(diag_rows)
    (RESULTS / "diagnose_summary_v2.json").write_text(json.dumps(diag_summary, indent=2), encoding="utf-8")
    print("Diagnose v2:", json.dumps(diag_summary), flush=True)

    loop_rows = [
        r
        for r in diag_rows
        if GenMetrics(**r["metrics"]).early_repeat or GenMetrics(**r["metrics"]).hit_cap
    ]
    contrast_rows = (loop_rows or diag_rows)[: args.max_contrast]
    print(f"Contrast rows: {len(contrast_rows)} (looping={len(loop_rows)})", flush=True)

    # --- Direction variants ---
    direction_bank: dict[str, dict[int, np.ndarray]] = {}
    localization_bank: dict[str, dict] = {}
    for mode, ewin, lwin in [
        ("gen_early_late", args.window, args.window),
        ("preloop_loop", args.window, args.late_window),
        ("gen_early_late", 32, 96),  # shorter productive / longer loop tail
    ]:
        tag = f"{mode}_e{ewin}_l{lwin}"
        print(f"=== Activations {tag} ===", flush=True)
        acts = collect_early_late_from_generations(
            model,
            tok,
            contrast_rows,
            format_prompt_fn=format_prompt,
            device=device,
            max_length=args.act_max_length,
            early_window=ewin,
            late_window=lwin,
            mode=mode,
        )
        u_all, gap_rows = build_u_all(acts)
        writer_rows = attribute_writer_mass(model, u_all)
        gap_map = {r["layer"]: r["gap"] for r in gap_rows}
        combined = []
        for r in writer_rows:
            combined.append(
                {
                    "layer": r["layer"],
                    "gap": gap_map.get(r["layer"], 0.0),
                    "pos_mass": r["pos_mass"],
                    "top64_mass": r["top64_mass"],
                    "score": 0.5 * gap_map.get(r["layer"], 0.0)
                    + 0.5 * r["pos_mass"] / (writer_rows[0]["pos_mass"] + 1e-9),
                }
            )
        combined.sort(key=lambda r: -r["score"])
        direction_bank[tag] = u_all
        localization_bank[tag] = {
            "by_gap": gap_rows[:10],
            "by_writer_mass": writer_rows[:10],
            "combined": combined[:10],
            "mode": mode,
            "early_window": ewin,
            "late_window": lwin,
        }
        np.savez(
            RESULTS / f"directions_v2_{tag}.npz",
            **{f"u_layer_{L}": u_all[L] for L in sorted(u_all)},
        )
        print(
            f"{tag} top layers:",
            [r["layer"] for r in combined[:6]],
            flush=True,
        )

    (RESULTS / "localization_v2.json").write_text(
        json.dumps(localization_bank, indent=2, default=float), encoding="utf-8"
    )

    # Prefer gen_early_late as primary; keep preloop as alternate.
    primary_tag = f"gen_early_late_e{args.window}_l{args.window}"
    alt_tag = f"preloop_loop_e{args.window}_l{args.late_window}"
    wide_tag = "gen_early_late_e32_l96"
    primary_u = direction_bank[primary_tag]
    loc = localization_bank[primary_tag]["combined"]
    top_layers = [r["layer"] for r in loc[:6]]

    # Candidate layer sets (beyond 21/20).
    layer_sets: list[list[int]] = []
    for ls in [
        top_layers[:2],
        top_layers[2:4],
        top_layers[4:6],
        [21, 20],
        [21],
        [20],
        [19],
        [18, 19],
        [17, 16],
        [15, 14],
        [21, 19],
        [20, 19],
        [21, 20, 19],
        [19, 18],
        [22, 21],
        [10, 18],  # mid+writer-mass style oddball from prior localization
    ]:
        key = tuple(ls)
        if key and key not in {tuple(x) for x in layer_sets}:
            # Keep only layers that exist on this model.
            n_layers = len(model.model.layers) if hasattr(model, "model") else 24
            cleaned = [L for L in ls if 0 <= L < n_layers]
            if cleaned:
                layer_sets.append(cleaned)

    alphas = [float(x) for x in args.alphas.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    # Build sweep schedule: primary grid is sparse; expand around promising hits.
    schedule: list[dict] = []
    # Always include old-layer α/k with new u variants.
    for tag in (primary_tag, alt_tag, wide_tag):
        for layers in ([21, 20], [21], [20], [19], [18, 19], [21, 19], [17, 16], top_layers[2:4]):
            if not layers:
                continue
            for alpha in alphas:
                for k in ks:
                    schedule.append(
                        {
                            "dir_tag": tag,
                            "layers": list(layers),
                            "alpha": alpha,
                            "k": k,
                            "shared": False,
                        }
                    )
    # Shared-u variants on mid-late layers (average of 18–21).
    for tag in (primary_tag, alt_tag):
        for layers in ([21, 20], [21, 19], [20, 19], [18, 19]):
            for alpha in alphas:
                for k in ks:
                    schedule.append(
                        {
                            "dir_tag": tag,
                            "layers": list(layers),
                            "alpha": alpha,
                            "k": k,
                            "shared": True,
                            "shared_from": [18, 19, 20, 21],
                        }
                    )

    # Dedup schedule
    seen = set()
    uniq_sched = []
    for item in schedule:
        key = (
            item["dir_tag"],
            tuple(item["layers"]),
            item["alpha"],
            item["k"],
            item.get("shared", False),
            tuple(item.get("shared_from") or []),
        )
        if key not in seen:
            seen.add(key)
            uniq_sched.append(item)
    schedule = uniq_sched
    if args.max_candidates and len(schedule) > args.max_candidates:
        # Prefer diverse layer sets at α=8 k=64 first, then others.
        def prio(it: dict) -> tuple:
            return (
                0 if it["alpha"] == 8.0 and it["k"] == 64 else 1,
                0 if it["dir_tag"] == primary_tag else 1,
                0 if not it.get("shared") else 1,
                -len(it["layers"]),
            )

        schedule = sorted(schedule, key=prio)[: args.max_candidates]
    print(f"Sweep schedule: {len(schedule)} candidates", flush=True)

    # --- Base sweep ---
    base_path = RESULTS / "sweep_v2_base.jsonl"
    if base_path.exists():
        base_path.unlink()
    print("=== BASE sweep ===", flush=True)
    base_rows = generate_dataset(
        model,
        tok,
        sweep_split,
        out_path=base_path,
        max_new_tokens=args.sweep_max_tokens,
        seed=args.seed + 100,
        device=device,
        early_repeat_threshold=3,
        stop_on_repeat=False,
    )
    base_sum = summarize_rows(base_rows)
    base_score = candidate_score(base_sum)
    print(f"base score={base_score:.3f} {base_sum}", flush=True)

    # --- Prior frozen edit on same protocol ---
    old_score = None
    old_sum = None
    if v1_backup.exists() or old_frozen_path.exists():
        src = v1_backup if v1_backup.exists() else old_frozen_path
        old_spec = load_edit_spec(src)
        old_edits = [
            {
                "layer": e["layer"],
                "u": np.asarray(e["u"], dtype=np.float64),
                "cols": np.asarray(e["cols"], dtype=np.int64),
                "alpha": float(e["alpha"]),
            }
            for e in old_spec["edits"]
        ]
        print("=== OLD frozen edit on sweep protocol ===", flush=True)
        old_sum = eval_edits(
            model,
            tok,
            sweep_split,
            old_edits,
            RESULTS / "sweep_v2_old_frozen.jsonl",
            args.sweep_max_tokens,
            args.seed + 100,
            device,
        )
        old_score = candidate_score(old_sum)
        print(f"old_frozen score={old_score:.3f} {old_sum}", flush=True)

    results = [
        {"tag": "base", "layers": [], "alpha": None, "k": None, "summary": base_sum, "score": base_score}
    ]
    if old_sum is not None:
        results.append(
            {
                "tag": "old_frozen_L21-20",
                "layers": [e["layer"] for e in old_spec["edits"]],
                "alpha": old_spec["edits"][0]["alpha"],
                "k": len(old_spec["edits"][0]["cols"]),
                "summary": old_sum,
                "score": old_score,
            }
        )

    best = None
    best_edits = None
    for i, item in enumerate(schedule):
        tag = item["dir_tag"]
        layers = item["layers"]
        alpha = item["alpha"]
        k = item["k"]
        u_all = direction_bank[tag]
        missing = [L for L in layers if L not in u_all]
        if missing:
            print(f"skip {item}: missing {missing}", flush=True)
            continue
        shared_u = None
        if item.get("shared"):
            shared_u = shared_direction(u_all, item.get("shared_from") or layers)
        edits = make_edits(model, u_all, layers, alpha, k, shared_u=shared_u)
        run_tag = (
            f"{tag}_a{alpha}_k{k}_L{'-'.join(map(str, layers))}"
            + ("_shared" if item.get("shared") else "")
        )
        print(f"=== [{i+1}/{len(schedule)}] {run_tag} ===", flush=True)
        summary = eval_edits(
            model,
            tok,
            sweep_split,
            edits,
            RESULTS / f"sweep_v2_{run_tag}.jsonl",
            args.sweep_max_tokens,
            args.seed + 100,
            device,
        )
        sc = candidate_score(summary)
        entry = {
            "tag": run_tag,
            "dir_tag": tag,
            "layers": layers,
            "alpha": alpha,
            "k": k,
            "shared": bool(item.get("shared")),
            "summary": summary,
            "score": sc,
        }
        results.append(entry)
        print(f"{run_tag}: score={sc:.3f} {summary}", flush=True)
        if best is None or sc > best["score"]:
            best = entry
            best_edits = [
                {
                    "layer": e["layer"],
                    "u": e["u"],
                    "cols": e["cols"].tolist() if hasattr(e["cols"], "tolist") else list(e["cols"]),
                    "alpha": e["alpha"],
                }
                for e in edits
            ]
            # Persist interim best edits (not yet declared frozen).
            save_edit_spec(
                RESULTS / "candidate_best_v2.json",
                {
                    "model_id": args.model,
                    "method": f"v2_{tag}" + ("_shared_u" if item.get("shared") else ""),
                    "edits": best_edits,
                    "meta": {**entry, "beats_base": sc > base_score, "beats_old": old_score is None or sc > old_score},
                },
            )

    assert best is not None and best_edits is not None
    beats_base = best["score"] > base_score
    beats_old = old_score is None or best["score"] > old_score
    freeze = beats_base and beats_old and best["tag"] != "base"

    report = {
        "diagnose": diag_summary,
        "localization": {k: localization_bank[k] for k in localization_bank},
        "base_sweep": base_sum,
        "base_score": base_score,
        "old_frozen_sweep": old_sum,
        "old_frozen_score": old_score,
        "results": results,
        "best": best,
        "beats_base": beats_base,
        "beats_old": beats_old,
        "froze_new": freeze,
        "protocol": {
            "diag_max_tokens": args.diag_max_tokens,
            "sweep_max_tokens": args.sweep_max_tokens,
            "stop_on_repeat_sweep": False,
            "n_diagnose": len(diagnose_split),
            "n_sweep": len(sweep_split),
            "n_schedule": len(schedule),
            "device": device,
            "seed": args.seed,
        },
        "elapsed_sec": time.time() - t0,
    }

    if freeze:
        # Keep v1 backup; write new freeze.
        save_edit_spec(
            RESULTS / "frozen_edit_v2.json",
            {
                "model_id": args.model,
                "method": f"v2_{best['dir_tag']}" + ("_shared_u" if best.get("shared") else ""),
                "edits": best_edits,
                "meta": {
                    **best,
                    "compare_to_v1": {
                        "v1_path": str(v1_backup),
                        "v1_score": old_score,
                        "v2_score": best["score"],
                    },
                },
            },
        )
        save_edit_spec(
            RESULTS / "frozen_edit.json",
            {
                "model_id": args.model,
                "method": f"v2_{best['dir_tag']}" + ("_shared_u" if best.get("shared") else ""),
                "edits": best_edits,
                "meta": {
                    **best,
                    "compare_to_v1": {
                        "v1_path": str(v1_backup),
                        "v1_score": old_score,
                        "v2_score": best["score"],
                    },
                },
            },
        )
        print(f"FROZE new edit → frozen_edit.json / frozen_edit_v2.json: {best['tag']}", flush=True)
    else:
        print(
            f"No freeze (beats_base={beats_base}, beats_old={beats_old}). "
            f"Best={best['tag']} score={best['score']:.3f}",
            flush=True,
        )
        # Leave frozen_edit.json as v1; still save candidate for inspection.
        save_edit_spec(
            RESULTS / "frozen_edit_v2_candidate.json",
            {
                "model_id": args.model,
                "method": f"v2_{best['dir_tag']}_NOT_FROZEN",
                "edits": best_edits,
                "meta": {**best, "beats_base": beats_base, "beats_old": beats_old},
            },
        )

    # --- Smoke (new candidate vs base; optionally vs old) ---
    edit_for_smoke = best_edits
    smoke_edit_path = RESULTS / "smoke_v2_edited.jsonl"
    smoke_base_path = RESULTS / "smoke_v2_base.jsonl"
    if smoke_base_path.exists():
        smoke_base_path.unlink()
    print(f"=== Smoke base n={len(smoke_split)} max_new={args.smoke_max_tokens} ===", flush=True)
    smoke_base_rows = generate_dataset(
        model,
        tok,
        smoke_split,
        out_path=smoke_base_path,
        max_new_tokens=args.smoke_max_tokens,
        seed=args.seed + 200,
        device=device,
        early_repeat_threshold=3,
        stop_on_repeat=False,
    )
    smoke_base_sum = summarize_rows(smoke_base_rows)

    print("=== Smoke edited (best v2 candidate) ===", flush=True)
    smoke_edit_sum = eval_edits(
        model,
        tok,
        smoke_split,
        edit_for_smoke,
        smoke_edit_path,
        args.smoke_max_tokens,
        args.seed + 200,
        device,
    )

    smoke_old_sum = None
    if (v1_backup.exists() or old_frozen_path.exists()) and args.smoke_old:
        src = v1_backup if v1_backup.exists() else old_frozen_path
        old_spec = load_edit_spec(src)
        old_edits = [
            {
                "layer": e["layer"],
                "u": np.asarray(e["u"], dtype=np.float64),
                "cols": np.asarray(e["cols"], dtype=np.int64),
                "alpha": float(e["alpha"]),
            }
            for e in old_spec["edits"]
        ]
        print("=== Smoke old frozen ===", flush=True)
        smoke_old_sum = eval_edits(
            model,
            tok,
            smoke_split,
            old_edits,
            RESULTS / "smoke_v2_old.jsonl",
            args.smoke_max_tokens,
            args.seed + 200,
            device,
        )

    def smoke_beats(edited: dict, base: dict) -> bool:
        return (
            (edited["early_repeat_rate"] + edited["hit_cap_rate"])
            < (base["early_repeat_rate"] + base["hit_cap_rate"])
        ) or (edited["finished_clean_rate"] > base["finished_clean_rate"]) or (
            edited["boxed_rate"] > base["boxed_rate"]
        ) or (edited["mean_tokens"] < base["mean_tokens"] * 0.85) or (
            edited.get("no_repeat_rate", 0) > base.get("no_repeat_rate", 0)
        )

    smoke_report = {
        "smoke_base": smoke_base_sum,
        "smoke_v2": smoke_edit_sum,
        "smoke_old": smoke_old_sum,
        "beats_base": smoke_beats(smoke_edit_sum, smoke_base_sum),
        "selection_froze_new": freeze,
        "best_tag": best["tag"],
        "protocol": {
            "max_new": args.smoke_max_tokens,
            "n": len(smoke_split),
            "device": device,
            "stop_on_repeat": False,
        },
    }
    (RESULTS / "smoke_report_v2.json").write_text(json.dumps(smoke_report, indent=2, default=float), encoding="utf-8")
    report["smoke"] = smoke_report
    report["elapsed_sec"] = time.time() - t0
    (RESULTS / "discover_v2_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("=== discover_v2 report (summary) ===", flush=True)
    print(
        json.dumps(
            {
                "best": best,
                "beats_base": beats_base,
                "beats_old": beats_old,
                "froze_new": freeze,
                "smoke_beats_base": smoke_report["beats_base"],
                "elapsed_sec": report["elapsed_sec"],
            },
            indent=2,
            default=float,
        ),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description="v2 loop-edit discovery (own generations only)")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diag-max-tokens", type=int, default=512)
    p.add_argument("--sweep-max-tokens", type=int, default=512)
    p.add_argument("--smoke-max-tokens", type=int, default=768)
    p.add_argument("--n-smoke-cap", type=int, default=12, help="Cap smoke n for CPU budget (0=all)")
    p.add_argument("--act-max-length", type=int, default=768)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--late-window", type=int, default=96)
    p.add_argument("--max-contrast", type=int, default=10)
    p.add_argument("--alphas", default="6,8,10")
    p.add_argument("--ks", default="48,64")
    p.add_argument("--max-candidates", type=int, default=18)
    p.add_argument("--reuse-diagnose", action="store_true", default=False)
    p.add_argument("--smoke-old", action="store_true", default=True)
    p.add_argument("--no-smoke-old", action="store_false", dest="smoke_old")
    args = p.parse_args()
    if args.n_smoke_cap == 0:
        args.n_smoke_cap = None
    run(args)


if __name__ == "__main__":
    main()
