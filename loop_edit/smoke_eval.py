#!/usr/bin/env python3
"""Smoke-eval frozen edit vs base on the held-out ~25-problem set (no early-stop)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import load_jsonl
from .edit import apply_edit_to_model, load_edit_spec, restore_model
from .evaluate import summarize_rows
from .generate import generate_dataset, load_model_and_tokenizer, resolve_device


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _device_name(device: str) -> str | None:
    try:
        if device.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_name(0)
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--edit", default=str(RESULTS / "frozen_edit.json"))
    ap.add_argument(
        "--report-name",
        default="auto",
        help="smoke_report.json basename, or 'auto' → smoke_report_gpu.json on accelerator else smoke_report.json",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing smoke_*_{gpu,cpu}.jsonl rows and only generate missing problem ids (same seed indices).",
    )
    args = ap.parse_args()
    device = resolve_device(args.device)

    if args.report_name == "auto":
        report_name = "smoke_report_gpu.json" if device in ("xpu", "cuda") else "smoke_report.json"
    else:
        report_name = args.report_name

    # Preserve the original CPU smoke report under an explicit name.
    cpu_report = RESULTS / "smoke_report.json"
    cpu_backup = RESULTS / "smoke_report_cpu.json"
    if cpu_report.exists() and not cpu_backup.exists():
        shutil.copy2(cpu_report, cpu_backup)

    tag = "gpu" if device in ("xpu", "cuda") else "cpu"
    base_path = RESULTS / f"smoke_base_{tag}.jsonl"
    edit_path = RESULTS / f"smoke_edited_{tag}.jsonl"

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
    print(f"Loading {args.model} on {device} ({_device_name(device)})...", flush=True)
    model, tok = load_model_and_tokenizer(args.model, device=device)

    if args.resume:
        base_rows = _load_existing_rows(base_path)
        done_base = {r["id"] for r in base_rows}
        todo_base = [p for p in smoke if p["id"] not in done_base]
        print(f"=== smoke base (resume: {len(base_rows)} done, {len(todo_base)} remaining) ===", flush=True)
        if todo_base:
            # Preserve original seed offsets by using each problem's index in the full split.
            index_by_id = {p["id"]: i for i, p in enumerate(smoke)}
            new_rows = []
            for prob in todo_base:
                i = index_by_id[prob["id"]]
                from .generate import generate_one
                from .metrics import GenMetrics

                row = generate_one(
                    model, tok, prompt=prob["prompt"], max_new_tokens=args.max_new,
                    seed=args.seed + i, device=device, stop_on_repeat=False,
                )
                row.update({"id": prob["id"], "source": prob.get("source"), "answer": prob.get("answer")})
                new_rows.append(row)
                with base_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({k: v for k, v in row.items()}, ensure_ascii=False) + "\n")
                m = GenMetrics(**row["metrics"])
                print(
                    f"[{len(base_rows)+len(new_rows)}/{len(smoke)}] {prob['id']} tokens={m.n_tokens} "
                    f"hit_cap={m.hit_cap} early={m.early_repeat} first_rep={m.first_repeat_at} "
                    f"m8={m.max_8gram_count} boxed={m.has_boxed}",
                    flush=True,
                )
                if device.startswith("xpu") and hasattr(torch, "xpu"):
                    try:
                        torch.xpu.empty_cache()
                    except Exception:
                        pass
            base_rows.extend(new_rows)
    else:
        if base_path.exists():
            base_path.unlink()
        print("=== smoke base ===", flush=True)
        base_rows = generate_dataset(
            model, tok, smoke, base_path, max_new_tokens=args.max_new, seed=args.seed,
            device=device, stop_on_repeat=False,
        )
    base_sum = summarize_rows(base_rows)

    backups = apply_edit_to_model(model, edits)
    if args.resume:
        edit_rows = _load_existing_rows(edit_path)
        done_edit = {r["id"] for r in edit_rows}
        todo_edit = [p for p in smoke if p["id"] not in done_edit]
        print(f"=== smoke edited (resume: {len(edit_rows)} done, {len(todo_edit)} remaining) ===", flush=True)
        if todo_edit:
            index_by_id = {p["id"]: i for i, p in enumerate(smoke)}
            from .generate import generate_one
            from .metrics import GenMetrics

            new_rows = []
            for prob in todo_edit:
                i = index_by_id[prob["id"]]
                row = generate_one(
                    model, tok, prompt=prob["prompt"], max_new_tokens=args.max_new,
                    seed=args.seed + i, device=device, stop_on_repeat=False,
                )
                row.update({"id": prob["id"], "source": prob.get("source"), "answer": prob.get("answer")})
                new_rows.append(row)
                with edit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({k: v for k, v in row.items()}, ensure_ascii=False) + "\n")
                m = GenMetrics(**row["metrics"])
                print(
                    f"[{len(edit_rows)+len(new_rows)}/{len(smoke)}] {prob['id']} tokens={m.n_tokens} "
                    f"hit_cap={m.hit_cap} early={m.early_repeat} first_rep={m.first_repeat_at} "
                    f"m8={m.max_8gram_count} boxed={m.has_boxed}",
                    flush=True,
                )
                if device.startswith("xpu") and hasattr(torch, "xpu"):
                    try:
                        torch.xpu.empty_cache()
                    except Exception:
                        pass
            edit_rows.extend(new_rows)
    else:
        if edit_path.exists():
            edit_path.unlink()
        print("=== smoke edited ===", flush=True)
        edit_rows = generate_dataset(
            model, tok, smoke, edit_path, max_new_tokens=args.max_new, seed=args.seed,
            device=device, stop_on_repeat=False,
        )
    edit_sum = summarize_rows(edit_rows)
    restore_model(model, backups)

    incomplete = len(base_rows) < len(smoke) or len(edit_rows) < len(smoke)
    report = {
        "incomplete": incomplete,
        "edit_spec": args.edit,
        "layers": [e["layer"] for e in edits],
        "alpha": edits[0]["alpha"] if edits else None,
        "k": len(edits[0]["cols"]) if edits else None,
        "smoke_base": base_sum,
        "smoke_edited": edit_sum,
        "beats_base": (
            None
            if incomplete
            else (
                (
                    (edit_sum["early_repeat_rate"] + edit_sum["hit_cap_rate"])
                    < (base_sum["early_repeat_rate"] + base_sum["hit_cap_rate"])
                )
                or edit_sum["boxed_rate"] > base_sum["boxed_rate"]
                or edit_sum.get("no_repeat_rate", 0) > base_sum.get("no_repeat_rate", 0)
                or (
                    (edit_sum.get("mean_first_repeat_at") or 0)
                    > (base_sum.get("mean_first_repeat_at") or 0) * 1.1
                )
            )
        ),
        "protocol": {
            "max_new": args.max_new,
            "stop_on_repeat": False,
            "device": device,
            "device_name": _device_name(device),
            "dtype": str(next(model.parameters()).dtype),
            "n": len(smoke),
            "n_base_completed": len(base_rows),
            "n_edited_completed": len(edit_rows),
            "backend": "intel_xpu" if device.startswith("xpu") else ("cuda" if device.startswith("cuda") else "cpu"),
            "resume": bool(args.resume),
        },
        "artifacts": {
            "base_generations": str(base_path),
            "edited_generations": str(edit_path),
            "report": str(RESULTS / report_name),
        },
    }
    out_name = report_name
    if incomplete:
        # Never pretend a partial run is the full GPU smoke report.
        out_name = report_name.replace(".json", ".partial.json")
        if out_name == report_name:
            out_name = "smoke_report_gpu.partial.json"
        report["do_not_treat_as_full_smoke"] = True
        report["artifacts"]["report"] = str(RESULTS / out_name)
    (RESULTS / out_name).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    # Only merge complete GPU smoke into phase1_report.
    if not incomplete:
        phase1 = RESULTS / "phase1_report.json"
        prev = json.loads(phase1.read_text()) if phase1.exists() else {}
        if device in ("xpu", "cuda"):
            prev["smoke_gpu"] = report
            prev["smoke_gpu_base"] = base_sum
            prev["smoke_gpu_edited"] = edit_sum
            prev["smoke_gpu_beats_base"] = report["beats_base"]
        else:
            prev.update({"smoke_base": base_sum, "smoke_edited": edit_sum, "beats_base": report["beats_base"]})
        phase1.write_text(json.dumps(prev, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
