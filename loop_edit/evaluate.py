"""Smoke evaluation helpers (truncation / repetition / boxed; optional math-verify)."""

from __future__ import annotations

import re
from typing import Any

from .metrics import GenMetrics, aggregate


def extract_gsm8k_gold(answer_field: str) -> str | None:
    # GSM8K answers end with #### <number>
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return None


def extract_boxed(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\s*\{([^{}]*)\}", text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def try_math_verify(pred: str | None, gold: str | None) -> bool | None:
    if pred is None or gold is None:
        return None
    try:
        from math_verify import parse, verify  # type: ignore

        return bool(verify(parse(pred), parse(gold)))
    except Exception:
        # Fallback: normalized string compare
        def norm(x: str) -> str:
            return re.sub(r"\s+", "", x).lower()

        return norm(pred) == norm(gold)


def summarize_rows(rows: list[dict]) -> dict[str, Any]:
    metrics = [GenMetrics(**r["metrics"]) for r in rows]
    summary = aggregate(metrics)
    # Optional correctness when gold available
    graded = []
    for r in rows:
        gold = extract_gsm8k_gold(r.get("answer") or "")
        pred = extract_boxed(r.get("text") or "")
        ok = try_math_verify(pred, gold)
        if ok is not None:
            graded.append(ok)
    if graded:
        summary["math_verify_rate"] = sum(graded) / len(graded)
        summary["n_graded"] = len(graded)
    return summary
