# Own Findings on Weight Editing: Thinking-Loop Closure and Deletion Tuning

**Draft — Project “playing with weights”**  
**Models:** `Qwen/Qwen3.5-0.8B` (loop edit), `Qwen/Qwen2.5-0.5B` (deletion tuning)  
**Code:** `github.com/maanyachugh17/playing-with-weights`  
**Principle:** Discover and measure *our* edits. Published checkpoints are comparison-only after freeze.

---

## Abstract

Small reasoning models often fail by repeating text until they hit a token limit. Prior work showed that an agent-written rank-one edit of MLP `down_proj` weights (Astra) and a closed-form rule in the same family (CLC-sel) can reduce thinking-mode loops on Qwen3.5-0.8B. Separately, Deletion Tuning shows that learning which gate-projection weights to zero can match much heavier post-training. We re-derive both classes of intervention from **our own** generations and a **short** mask train—without copying published edited tensors into the discovery loop.

On a CPU smoke protocol (thinking on, max 512–1024 new tokens, no early-stop during selection/eval), a difference-of-means direction from early vs late windows of our generations localizes strongest residual change at late layers, but **selection on held-out practice prefers layers 21 and 20** with strength α = 8 and k = 64 columns per layer. That candidate beats the unedited base on loop metrics in the 8-problem sweep (early-repeat rate 0.875 vs 1.0; no-repeat rate 0.125 vs 0.0). Editing the gap-top layers 23/22 at the same α/k *worsens* tight 8-gram loops—evidence that localization ≠ edit selection. We also re-smoke the frozen edit on an **Intel Arc 140V XPU** host at max-new 4096: hit-caps stay near 1.0 and the edit does **not** beat base on the composite smoke score under that longer budget—honest negative at the longer horizon. A reduced-steps Deletion Tuning run on Qwen2.5-0.5B produces our own compressed `gate_proj` mask in the upstream packbits+zstd format; we report deleted fraction, train cost, and base vs masked behavior under the same honesty constraints. Full MATH-500 and paper-parity GRPO are out of scope.

---

## 1. Introduction

Weight-space interventions offer a middle path between prompting and full fine-tuning: change a tiny fraction of parameters using forward-pass evidence. Two recent lines motivate this project:

1. **Thinking-loop edits.** Takehi et al. (Astra / weight-space programming) edited Qwen3.5-0.8B’s MLP down-projections so the model exits reasoning loops more often. Xu’s CLC-sel recovers a similar edit from the model’s own answers with a fixed rule.
2. **Deletion Tuning.** Lu et al. learn a binary mask over MLP gate projections; zeroing ≪0.05% of weights can rival full fine-tuning under GRPO/SFT/distillation.

Our goal is not to reproduce their leaderboard numbers. It is to **run the discovery ourselves**: diagnose loops on practice math prompts, build contrast sets from *our* completions, localize writers without assuming Astra’s layers 15 and 19, freeze an edit by sweep (never MATH-500 for selection), smoke-eval against base, and only then optionally compare to published checkpoints. For deletion tuning, we pin the public code and train a short mask of our own rather than downloading Jacklu0831 masks to choose the intervention.

---

## 2. Related Work

**Astra loop edit (Takehi et al.).** An LLM agent inspected Qwen3.5-0.8B weights and practice rollouts, then submitted a rank-one edit on `down_proj` in layers 15 and 19 (top-64 columns, α ≈ 6). Public writeups report large MATH-500 gains and lower hit-cap rates versus base. We treat their checkpoint (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`) as a **post-freeze** reference only.

**CLC / CLC-sel (Xu).** A non-agent rule builds \(u\) from mean residual states inside verbatim loops versus states before `</think>`, and selects neurons whose write along \(u\) rises on repeated text. Same edit algebra: \(W' = W - \alpha\, u\,(m \odot s)^\top\) with \(s = W^\top u\). We adopt the algebra and the spirit of contrastive directions, but estimate \(u\) from **early vs late windows of our generations** when clean `</think>` finishes are scarce under a capped CPU budget.

**Direction ablation / orthogonalization (Arditi et al. and follow-ons).** Refusal and other behaviors can be attenuated by removing a direction from residual stream writes. Our loop edit is in that family (rank-one subtraction along a measured direction), applied to MLP down-projections with a column mask rather than a dense residual projection.

**Deletion Tuning (Lu et al.).** Trainable logits over gate-projection weights anneal from soft masks to hard keep/delete bits; adapters are tiny bitmaps. Upstream training uses verl GRPO on GPU. We pin `agentic-learning-ai-lab/deleting-weights` and run a **reduced-steps** CPU trainer that writes the same `packbits_zstd` mask format.

---

## 3. Method

### 3.1 Thinking-loop edit (Phase 1)

**Model.** `Qwen/Qwen3.5-0.8B`, thinking mode via chat template (`enable_thinking=True`).

**Data.** Disjoint splits from GSM8K train and MATH train: diagnose (\(n=20\)), sweep (\(n=8\)), smoke (\(n=25\)). MATH-500 is never used for selection.

**Diagnose.** Generate with an optional early 8-gram repeat stop for throughput. Record hit-cap, max 8-gram count, first-repeat index, `\boxed{}` presence, optional math-verify when gold is available.

**Contrast & direction.** On looping (or all) diagnose completions, teacher-force `prompt + completion` and take residual-stream means over the first and last 64 tokens. Per layer \(\ell\),

\[
u_\ell = \mathrm{normalize}\big(\bar h^{\mathrm{late}}_\ell - \bar h^{\mathrm{early}}_\ell\big).
\]

**Localize writers.** Rank layers by residual gap \(\|\bar h^{\mathrm{late}}-\bar h^{\mathrm{early}}\|\) and by positive write mass \(\sum_c \mathrm{ReLU}(s_c)\), \(s = W^\top u\), \(W\) = `mlp.down_proj.weight`. Combined score picks candidate layers; **we do not hard-code 15/19**.

**Edit.** For selected layers and top-\(k\) columns by \(s_c\),

\[
W' = W - \alpha\, u\,(m \odot s)^\top.
\]

**Selection.** Sweep layer sets and \(\alpha\) on the held-out practice sweep **without** early-stop (so severity is visible). Freeze the best score before smoke.

### 3.2 Deletion Tuning mask (Phase 2)

**Backbone.** `Qwen/Qwen2.5-0.5B` (fallback Llama-3.2-1B-Instruct if needed).

**Pin.** `weight-edits/deletion_tuning/deleting-weights` @ `94f0340…` (lean vendor of `src/` + scripts; full `verl/` omitted with restore instructions).

**Train.** Wrap each `mlp.gate_proj` with a sigmoid mask over logits; short group-relative reward updates on GSM8K with boxed-answer reward; anneal sharpness; save hard keep-bits as packbits+zstd. Full verl/flash-attn GRPO parity is out of scope.

**Eval.** Base vs masked mean reward / boxed rate on a tiny held-out GSM8K slice; report deleted count and fraction of model parameters.

---

## 4. Experiments

### 4.1 Compute and protocol honesty

| Item | Setting |
| --- | --- |
| Hardware (discovery / CPU smoke) | CPU float32 (initial run) |
| Hardware (long smoke) | Intel Arc™ 140V (16GB) via PyTorch **XPU** (`torch 2.15.0+xpu`, bf16) — **not** NVIDIA CUDA |
| Approx. decode speed | ~14 tok/s CPU (float32, reference kernels); ~5–14 tok/s observed on Arc XPU depending on memory pressure |
| Diagnose max new tokens | 640 (with early-stop for throughput) |
| Sweep max new tokens | 512, **stop_on_repeat=False** |
| Smoke max new tokens | **CPU:** 1024; **XPU re-smoke:** **4096** (same split, no early-stop) |
| Sampling | temperature 0.6, top_p 0.95, top_k 20 (loop edit) |

Early-stop during *diagnose* compresses wall time but collapses severity differences if used for *selection*. After an initial failed sweep that looked uniformly “early=True,” we switched selection/eval to full capped generations and track `first_repeat_at` / `mean_max_8gram` / `no_repeat_rate`.

### 4.2 Phase 1 configurations

Layer sets swept at α=8, k=64: `{23,22}`, `{21,20}`, `{18,19}`, `{23,22,21,20}`; then α ∈ {6,12} refine on the winner.

### 4.3 Phase 2 configurations

Short GRPO-style mask train: see Appendix B and `deletion_tuning/results/phase2_report.json` (filled after train).

---

## 5. Results

### 5.1 Diagnose (practice, thinking on)

| Metric | Value (\(n=20\)) |
| --- | --- |
| Hit-cap rate | 0.10 |
| Early-repeat rate (threshold 3 on 8-grams) | 0.90 |
| Boxed rate | 0.25 |
| Finished-clean rate | 0.00 |
| Mean tokens (with early-stop) | 330.4 |

The base model almost never finishes cleanly under our caps; loops dominate.

### 5.2 Localization (our generations only)

Top residual gaps: layers **23 > 22 > 21 > 20 > 19 > 18 > … > 15**. Layer 15 appears mid-pack by gap, not at the top. Combined gap + positive write mass still ranks 23/22 first. **Edit selection disagrees** (next subsection).

### 5.3 Layer / α sweep (held-out practice, \(n=8\), max 512, no early-stop)

| Candidate | early_repeat | no_repeat | mean_first_repeat | mean_max_8gram | boxed | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1.000 | 0.000 | 273.5 | 3.63 | 0.25 | −1.259 |
| α=8 k=64 **L21–20** | **0.875** | **0.125** | 277.3 | **3.38** | 0.25 | **−0.989** |
| α=6 k=64 L21–20 | 1.000 | 0.000 | **306.3** | 3.50 | 0.25 | −1.188 |
| α=8 k=64 L18–19 | 1.000 | 0.000 | 288.4 | 3.75 | 0.25 | −1.236 |
| α=12 k=64 L21–20 | 1.000 | 0.000 | 294.9 | 3.88 | 0.125 | −1.417 |
| α=8 k=64 L23–22 | 1.000 | 0.000 | 184.8 | **128.3** | 0.00 | −7.918 |
| α=8 k=64 L23–22–21–20 | 1.000 | 0.000 | 205.0 | **213.1** | 0.00 | −12.25 |

**Frozen edit:** layers **{21, 20}**, **α = 8**, **k = 64** columns per layer (top writers along our \(u\)). Beats base on the selection score via lower early-repeat rate and a non-zero no-repeat rate. Editing gap-leading layers 23/22 at the same budget induces *tighter* verbatim loops (huge `mean_max_8gram`).

### 5.4 Smoke eval vs base (~25 problems)

Same frozen edit (L21–20, α=8, k=64) and the same 25 held-out GSM8K prompts / seed; `stop_on_repeat=False`. Two hardware budgets:

**CPU smoke** (`smoke_report.json` / `smoke_report_cpu.json`): max-new **1024**.

| Metric | CPU Base | CPU Edited |
| --- | ---: | ---: |
| \(n\) | 25 | 25 |
| Hit-cap rate | 1.00 | **0.96** |
| Early-repeat rate | 1.00 | **0.96** |
| No-repeat rate | 0.00 | **0.04** |
| Mean first-repeat | 335.4 | **348.4** |
| Mean max 8-gram | 6.88 | 7.00 |
| Mean tokens | 1024 | **1010.5** |
| Boxed rate | 0.20 | **0.24** |
| Math-verify (graded subset) | 0.00 (5 graded) | 0.17 (6 graded) |

**XPU / Arc smoke** (`smoke_report_gpu.json`): Intel Arc 140V, bf16, max-new **4096**.

| Metric | XPU Base | XPU Edited |
| --- | ---: | ---: |
| \(n\) | 25 | 25 |
| Hit-cap rate | **0.96** | 1.00 |
| Early-repeat rate | 1.00 | 1.00 |
| No-repeat rate | 0.00 | 0.00 |
| Mean first-repeat | 347.3 | **378.2** |
| Mean max 8-gram | **35.9** | 41.0 |
| Mean tokens | **3994.6** | 4096.0 |
| Boxed rate | **0.40** | 0.32 |
| Math-verify (graded subset) | 0.00 (10 graded) | 0.00 (8 graded) |
| `beats_base` (code composite) | — | **false** |

Under the 1024-token CPU cap the edit **beats base** on the composite criteria (`beats_base: true`). At the 4096-token XPU budget, hit-caps remain near **1.0**, no-repeat stays **0**, boxed falls (0.40→0.32), and mean max-8-gram rises; the only favorable movement is a later mean first-repeat (347→378). That is an honest **null/negative** at the longer horizon—not a claim that the edit “fails forever,” but evidence that the CPU win does not automatically scale when generations are allowed to run four times longer. These are still not MATH-500 claims.

### 5.5 Deletion Tuning (our mask)

Artifact: `deletion_tuning/results/phase2_report.json` and `our_mask/` (packbits_zstd).

| Quantity | Value |
| --- | ---: |
| Backbone | Qwen2.5-0.5B |
| Method (final) | Wanda-style \(\lvert W\rvert\cdot a\) scores on `gate_proj` from 12 GSM8K calibration forwards; delete lowest 0.1% of gate weights |
| Wall time | ~105 s (CPU) |
| Deleted gate weights | 104,595 |
| Fraction of gate params | 0.100% |
| Fraction of model params | 0.0212% |
| Base mean reward / boxed (\(n=16\)) | 0.106 / 0.0625 |
| Masked mean reward / boxed (\(n=16\)) | 0.047 / 0.000 |

Released Jacklu0831 masks were **not** used to choose the train.

**Failed attempts (archived, reported for honesty):** (1) Adam logits with \(N(0,0.01)\) init deleted ~50% of gate weights and collapsed generations; (2) keep-biased CE + weak delete penalty deleted 0 weights. Full verl GRPO was not runnable on this CPU host.

---

## 6. Discussion and Limitations

**Localization ≠ selection.** The largest early→late residual gaps sit at layers 23/22, but editing them with our \(u\) and α=8 produced pathological tight loops. Mid-late layers 21/20 gave the only clear sweep win. This cautions against freezing the gap-argmax without a behavioral sweep.

**Not Astra’s layers—and not a claim of superiority.** We did not import Astra or CLC tensors. Layer 19 appears in our gap ranking but lost to 21/20 under our α/k grid. We make no MATH-500 comparison in this draft.

**Direction quality.** With almost no clean finishes under CPU caps, we used early vs late windows instead of pre-`</think>` states. That proxy can mix “still reasoning” with “about to close,” and may explain why edits delay or thin loops more than they create boxed answers.

**Smoke protocol / hardware.** Discovery and the first smoke used CPU caps of 512–1024. We later re-ran the frozen edit on Intel Arc 140V (**XPU**, not CUDA) at max-new 4096. Hit-caps stay near 1.0 even with the longer budget; the metric that still moves usefully on XPU is mainly first-repeat index, while boxed and 8-gram severity do **not** improve. The earlier “CPU-only” limitation is outdated for smoke hardware, but NVIDIA CUDA remains unavailable on this host.

**Deletion Tuning.** Without GPU verl, we could not match paper GRPO. Our delivered mask is a sparse Wanda-style delete on `gate_proj` calibrated on our GSM8K forwards (0.021% of model weights). On a 16-problem probe the masked model did **not** beat base reward—honest negative/null under this protocol. The artifact still demonstrates end-to-end “our mask” export in the upstream bitmap format.

**Sample sizes.** Diagnose 20 / sweep 8 / smoke 25. Scores are noisy; the layer-set qualitative pattern (23/22 harmful, 21/20 helpful at α=8) is the robust takeaway.

---

## 7. Conclusion

We derived a thinking-loop weight edit and a deletion mask from our own rollouts and a short train. The frozen loop edit is **`down_proj` on layers 21 and 20, α=8, k=64`**, selected because it beat base on uncapped-repeat sweep metrics while the gap-top layers 23/22 did not. Deletion Tuning yields our own packbits mask with reported delete fraction and tiny-set behavior. The central methodological point stands: **measure your edit; do not paste a published tensor into the discovery loop.**

---

## Appendix A — Loop-edit artifacts

- Code: `weight-edits/loop_edit/`
- Diagnose summary: `loop_edit/results/diagnose_summary.json`
- Localization: `loop_edit/results/localization.json`
- Layer sweep: `loop_edit/results/layer_sweep_report.json`
- Frozen spec: `loop_edit/results/frozen_edit.json`
- Smoke (CPU): `loop_edit/results/smoke_report.json` / `smoke_report_cpu.json`
- Smoke (XPU / Arc, max-new 4096): `loop_edit/results/smoke_report_gpu.json`

## Appendix B — Deletion-tuning artifacts

- Pin: `weight-edits/deletion_tuning/PINNED_COMMIT.txt`
- Trainer: `weight-edits/deletion_tuning/train_mask.py`
- Mask + report: `deletion_tuning/results/our_mask/`, `phase2_report.json`

## Appendix C — Reproducibility commands

```bash
PYTHONPATH=weight-edits python -m loop_edit.run_phase1 --reuse-diagnose ...
PYTHONPATH=weight-edits python -m loop_edit.sweep_layers --alpha 8 --k 64 --max-new 512
PYTHONPATH=weight-edits python -m loop_edit.smoke_eval --max-new 1024
python -m loop_edit.smoke_eval --device xpu --max-new 4096 --resume
python weight-edits/deletion_tuning/train_mask.py --model Qwen/Qwen2.5-0.5B --steps 20
```
