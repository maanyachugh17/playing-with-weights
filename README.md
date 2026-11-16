# Playing with weights

Discover and measure **our own** model weight edits — not copies of published checkpoints.

This repo holds the experiment code formerly under `maanyachugh17/Me` → `weight-edits/`. Paper draft lives in [`docs/weight-editing-paper.md`](docs/weight-editing-paper.md).

| Phase | Path | Backbone | Goal |
| --- | --- | --- | --- |
| 1 Thinking-loop edit | [`loop_edit/`](loop_edit/) | `Qwen/Qwen3.5-0.8B` | Diff-of-means direction + localized `down_proj` column edit |
| 2 Deletion Tuning | [`deletion_tuning/`](deletion_tuning/) | `Qwen/Qwen2.5-0.5B` | Short GRPO-style train of **our** `gate_proj` mask |

Published checkpoints (e.g. `RikiyaT/Qwen3.5-0.8B-weight-space-programming`, `Jacklu0831/deletion-tuning-masks`) are optional comparisons **after** our artifact is frozen.

## Setup

```bash
pip install torch transformers datasets accelerate safetensors sentencepiece zstandard math-verify
```

Accelerator order for `--device auto`: Intel **XPU** (Arc), then CUDA, else CPU. On CPU, generation lengths are capped and loops are stopped early via 8-gram repetition.

```bash
# Full XPU smoke (Arc), resume-safe after soft-stop
python -m loop_edit.smoke_eval --device xpu --max-new 4096 --resume
```

## Run loop edit (Phase 1)

From the repo root:

```bash
python -m loop_edit.run_phase1 \
  --n-diagnose 24 --n-sweep 8 --n-smoke 25 \
  --diag-max-tokens 768 --sweep-max-tokens 768 --smoke-max-tokens 2048 \
  --alphas 2,4,6 --ks 32,64
```

Pipeline: diagnose practice prompts → contrast residual means → localize writers → edit `W' = W − α u (m ⊙ s)ᵀ` → sweep on held-out practice (not MATH-500) → freeze → smoke-eval vs base.

Artifacts: `loop_edit/results/` (`frozen_edit.json`, `phase1_report.json`, smoke jsonl).

## Run deletion tuning (Phase 2)

Upstream pin: `deletion_tuning/deleting-weights` @ see [`deletion_tuning/PINNED_COMMIT.txt`](deletion_tuning/PINNED_COMMIT.txt)  
([agentic-learning-ai-lab/deleting-weights](https://github.com/agentic-learning-ai-lab/deleting-weights)).

Reduced-steps trainer (same packbits+zstd mask format):

```bash
python deletion_tuning/train_mask.py \
  --model Qwen/Qwen2.5-0.5B --steps 20 --batch-size 2 --group-size 2 \
  --max-new 128 --train-n 64 --eval-n 20
```

Outputs: `deletion_tuning/results/our_mask/`, `phase2_report.json`.

Apply a saved mask:

```bash
python deletion_tuning/apply_our_mask.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --mask_dir deletion_tuning/results/our_mask \
  --output_dir deletion_tuning/results/masked_model
```

## Docs

- [`docs/weight-editing-paper.md`](docs/weight-editing-paper.md) — main draft (Abstract → Conclusion)
- [`docs/loop-edit-appendix.md`](docs/loop-edit-appendix.md) — loop-edit metrics pointer
- [`docs/deletion-tuning-appendix.md`](docs/deletion-tuning-appendix.md) — deletion-tuning metrics pointer

## Principle

**Discover and measure our own edits.** External edited tensors are comparison-only after freeze.
