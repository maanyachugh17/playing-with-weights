# Vendored slice

Pinned commit: see `../PINNED_COMMIT.txt`.

This directory keeps the upstream `src/`, `scripts/`, root README/LICENSE, and requirements files needed to understand and apply Deletion Tuning masks.

The full `verl/` and `sdft/` trees from upstream are omitted here (large, GPU-only). Restore the complete repo with:

```bash
git clone https://github.com/agentic-learning-ai-lab/deleting-weights
git -C deleting-weights checkout 94f0340ebb574f95bb41d844165253a6785e215e
```

Our reduced-steps trainer lives in `../train_mask.py` and writes masks in the same `packbits_zstd` format as `src/apply_mask.py`.
