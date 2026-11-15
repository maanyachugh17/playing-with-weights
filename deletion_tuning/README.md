# Deletion Tuning (our mask)

- **Pinned upstream:** `deleting-weights/` @ commit in `../PINNED_COMMIT.txt`
- **Our trainer:** `train_mask.py` — short GRPO-style run on GSM8K, mask over every `mlp.gate_proj`
- **Mask format:** `packbits_zstd` (compatible with upstream `src/apply_mask.py`)
- **Backbone default:** `Qwen/Qwen2.5-0.5B` (fallback: `meta-llama/Llama-3.2-1B-Instruct`)

Released `Jacklu0831/deletion-tuning-masks` are **not** used to choose the train; optional secondary comparison only.
