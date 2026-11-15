"""Practice and smoke problem sets from GSM8K / MATH train (never MATH-500 for selection)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_gsm8k_train(limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for i, ex in enumerate(ds):
        q = ex["question"].strip()
        a = ex["answer"].strip()
        rows.append(
            {
                "id": f"gsm8k-{i}",
                "source": "gsm8k_train",
                "question": q,
                "answer": a,
                "prompt": q + PROMPT_SUFFIX,
            }
        )
        if limit is not None and len(rows) >= limit * 4:
            break
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit is not None else rows


def load_math_train(limit: int | None = None, seed: int = 1) -> list[dict[str, Any]]:
    """Prefer hendrycks MATH train via HF; fall back to GSM8K-only if unavailable."""
    from datasets import load_dataset

    candidates = [
        ("EleutherAI/hendrycks_math", None),
        ("lighteval/MATH", "all"),
        ("DigitalLearningGmbH/MATH-lighteval", None),
    ]
    last_err: Exception | None = None
    for name, config in candidates:
        try:
            if config is None:
                ds = load_dataset(name, split="train")
            else:
                ds = load_dataset(name, config, split="train")
            rows = []
            for i, ex in enumerate(ds):
                q = (ex.get("problem") or ex.get("question") or "").strip()
                a = (ex.get("solution") or ex.get("answer") or "").strip()
                if not q:
                    continue
                rows.append(
                    {
                        "id": f"math-{i}-{_stable_id(q)}",
                        "source": "math_train",
                        "question": q,
                        "answer": a,
                        "prompt": q + PROMPT_SUFFIX,
                    }
                )
                if limit is not None and len(rows) >= limit * 4:
                    break
            rng = random.Random(seed)
            rng.shuffle(rows)
            return rows[:limit] if limit is not None else rows
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not load MATH train; last error: {last_err}")


def build_splits(
    n_diagnose: int = 32,
    n_sweep: int = 16,
    n_smoke: int = 25,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Build disjoint practice / held-out / smoke sets from GSM8K (+ MATH if available)."""
    gsm = load_gsm8k_train(limit=n_diagnose + n_sweep + n_smoke + 20, seed=seed)
    try:
        math = load_math_train(limit=n_diagnose + n_sweep + 20, seed=seed + 7)
    except Exception:
        math = []

    # Mix diagnose from both sources; keep smoke GSM8K-only for grading ease.
    diagnose = (gsm[: n_diagnose // 2] + math[: n_diagnose - n_diagnose // 2])[:n_diagnose]
    used = {r["id"] for r in diagnose}
    rest_gsm = [r for r in gsm if r["id"] not in used]
    rest_math = [r for r in math if r["id"] not in used]
    sweep = (rest_gsm[: n_sweep // 2] + rest_math[: n_sweep - n_sweep // 2])[:n_sweep]
    used |= {r["id"] for r in sweep}
    smoke = [r for r in gsm if r["id"] not in used][:n_smoke]
    return {"diagnose": diagnose, "sweep": sweep, "smoke": smoke}


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
