"""Loop / completion metrics for thinking-mode generations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable


BOXED_RE = re.compile(r"\\boxed\s*\{")


@dataclass
class GenMetrics:
    n_tokens: int
    hit_cap: bool
    has_boxed: bool
    max_8gram_count: int
    early_repeat: bool
    finished_clean: bool
    first_repeat_at: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def ngram_counts(token_ids: list[int], n: int = 8) -> Counter:
    if len(token_ids) < n:
        return Counter()
    grams = (tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1))
    return Counter(grams)


def max_ngram_count(token_ids: list[int], n: int = 8) -> int:
    counts = ngram_counts(token_ids, n=n)
    return max(counts.values()) if counts else 0


def first_ngram_repeat_at(token_ids: list[int], n: int = 8, threshold: int = 3) -> int | None:
    """Index (end-exclusive) where some n-gram first reaches `threshold` occurrences."""
    if len(token_ids) < n:
        return None
    counts: Counter = Counter()
    for i in range(len(token_ids) - n + 1):
        gram = tuple(token_ids[i : i + n])
        counts[gram] += 1
        if counts[gram] >= threshold:
            return i + n
    return None


def has_boxed(text: str) -> bool:
    return bool(BOXED_RE.search(text))


def score_generation(
    text: str,
    token_ids: list[int],
    max_new_tokens: int,
    early_repeat_threshold: int = 3,
) -> GenMetrics:
    hit_cap = len(token_ids) >= max_new_tokens
    m8 = max_ngram_count(token_ids, n=8)
    first_at = first_ngram_repeat_at(token_ids, n=8, threshold=early_repeat_threshold)
    early = first_at is not None
    boxed = has_boxed(text)
    finished_clean = (not hit_cap) and (not early) and boxed
    return GenMetrics(
        n_tokens=len(token_ids),
        hit_cap=hit_cap,
        has_boxed=boxed,
        max_8gram_count=m8,
        early_repeat=early,
        finished_clean=finished_clean,
        first_repeat_at=first_at,
    )


def aggregate(metrics: Iterable[GenMetrics]) -> dict:
    rows = list(metrics)
    n = max(len(rows), 1)
    firsts = [m.first_repeat_at for m in rows if m.first_repeat_at is not None]
    return {
        "n": len(rows),
        "hit_cap_rate": sum(m.hit_cap for m in rows) / n,
        "early_repeat_rate": sum(m.early_repeat for m in rows) / n,
        "boxed_rate": sum(m.has_boxed for m in rows) / n,
        "finished_clean_rate": sum(m.finished_clean for m in rows) / n,
        "mean_tokens": sum(m.n_tokens for m in rows) / n,
        "mean_max_8gram": sum(m.max_8gram_count for m in rows) / n,
        "mean_first_repeat_at": (sum(firsts) / len(firsts)) if firsts else None,
        "no_repeat_rate": sum(m.first_repeat_at is None for m in rows) / n,
    }
