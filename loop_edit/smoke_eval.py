#!/usr/bin/env python3
"""Smoke-eval frozen edit vs base on the held-out ~25-problem set (no early-stop)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_jsonl
from .edit import apply_edit_to_model, load_edit_spec, restore_model
from .evaluate import summarize_rows
from .generate import generate_dataset, load_model_and_tokenizer


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--edit", default=str(RESULTS / "frozen_edit.json"))
    args = ap.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")

    spec = load_edit_spec(Path(args.edit))
    edits = []
    for e in spec["edits"]:
        edits.append(
            {
                "layer": int(e["layer"]),
                "alpha": float(e["alpha"]),
                "cols": np.asarray(e["cols"], dtype=np.int64),
                "u": np.asarray(e["u"], dtype=np.float64),
            }
        )
    smoke = load_jsonl(RESULTS / "split_smoke.jsonl")
    model, tok = load_model_and_tokenizer(args.model, device=device)

    base_path = RESULTS / "smoke_base.jsonl"
    if base_path.exists():
        base_path.unlink()
    print("=== smoke base ===", flush=True)
    base_rows = generate_dataset(
        model, tok, smoke, base_path, max_new_tokens=args.max_new, seed=args.seed,
        device=device, stop_on_repeat=False,
    )
    base_sum = summarize_rows(base_rows)

    backups = apply_edit_to_model(model, edits)
    edit_path = RESULTS / "smoke_edited.jsonl"
    if edit_path.exists():
        edit_path.unlink()
    print("=== smoke edited ===", flush=True)
    edit_rows = generate_dataset(
        model, tok, smoke, edit_path, max_new_tokens=args.max_new, seed=args.seed,
        device=device, stop_on_repeat=False,
    )
    edit_sum = summarize_rows(edit_rows)
    restore_model(model, backups)

    report = {
        "edit_spec": args.edit,
        "layers": [e["layer"] for e in edits],
        "alpha": edits[0]["alpha"] if edits else None,
        "k": len(edits[0]["cols"]) if edits else None,
        "smoke_base": base_sum,
        "smoke_edited": edit_sum,
        "beats_base": (
            (edit_sum["early_repeat_rate"] + edit_sum["hit_cap_rate"])
            < (base_sum["early_repeat_rate"] + base_sum["hit_cap_rate"])
        )
        or edit_sum["boxed_rate"] > base_sum["boxed_rate"]
        or edit_sum.get("no_repeat_rate", 0) > base_sum.get("no_repeat_rate", 0)
        or (
            (edit_sum.get("mean_first_repeat_at") or 0)
            > (base_sum.get("mean_first_repeat_at") or 0) * 1.1
        ),
        "protocol": {"max_new": args.max_new, "stop_on_repeat": False, "device": device, "n": len(smoke)},
    }
    (RESULTS / "smoke_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    # merge into phase1_report if present
    phase1 = RESULTS / "phase1_report.json"
    out = {"smoke": report}
    if phase1.exists():
        prev = json.loads(phase1.read_text())
        prev.update({"smoke_base": base_sum, "smoke_edited": edit_sum, "beats_base": report["beats_base"]})
        phase1.write_text(json.dumps(prev, indent=2, default=float), encoding="utf-8")
        out = prev
    else:
        phase1.write_text(json.dumps({"smoke_base": base_sum, "smoke_edited": edit_sum, "beats_base": report["beats_base"]}, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
