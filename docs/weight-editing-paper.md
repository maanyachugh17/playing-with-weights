# Own Findings on Weight Editing: Thinking-Loop Closure and Deletion Tuning

**Draft — Project “playing with weights”**  
**Models:** `Qwen/Qwen3.5-0.8B` (loop edit), `Qwen/Qwen2.5-0.5B` (deletion tuning)  
**Code:** `github.com/maanyachugh17/playing-with-weights`  
**Principle:** Discover and measure *our* edits. Published checkpoints are comparison-only after freeze.

---

## Abstract

Small reasoning models often fail by repeating text until they hit a token limit. Prior work showed that an agent-written rank-one edit of MLP `down_proj` weights (Astra) and a closed-form rule in the same family (CLC-sel) can reduce thinking-mode loops on Qwen3.5-0.8B. Separately, Deletion Tuning shows that learning which gate-projection weights to zero can match much heavier post-training. We re-derive both classes of intervention from **our own** generations and a **short** mask train—without copying published edited tensors into the discovery loop.

On a CPU protocol (thinking on, max 512–1024 new tokens, no early-stop during selection/eval), a difference-of-means direction from early vs late windows of our generations localizes strongest residual change at late layers. An initial freeze used layers **21/20**, α = 8, k = 64 and beat base on an 8-problem sweep, but failed the composite smoke score on **Intel Arc 140V XPU** at max-new 4096 (`beats_base: false`). A second discovery pass rebuilt \(u\) with **generation-relative** and **pre-loop vs loop-tail** windows (still our generations only; no Astra tensors) and swept beyond 21/20. Selection then prefers **layers 21 and 19** with the same α/k under the pre-loop/loop direction (sweep score −0.601 vs base −1.259 vs old freeze −0.989). A capped CPU smoke (n=12, max-new 768) shows only soft severity gains for the new freeze (later first-repeat, lower mean max-8-gram) and does **not** flip the composite `beats_base` flag—honest mixed transfer. Editing gap-top layers 23/22 at α=8 still worsens tight 8-gram loops—localization ≠ selection. A reduced-steps Deletion Tuning run on Qwen2.5-0.5B produces our own compressed `gate_proj` mask; we report deleted fraction, train cost, and base vs masked behavior under the same honesty constraints. Full MATH-500 and paper-parity GRPO are out of scope.

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

**Contrast & direction.** On looping diagnose completions, teacher-force `prompt + completion` and estimate residual means. v1 used first/last windows of the full sequence. v2 improves the contrast (still without Astra tensors):

- `gen_early_late`: first/last windows of the *generation* span only (excludes the prompt).
- `preloop_loop`: window just before the first 8-gram repeat vs a looping-tail window (falls back to gen early/late when onset is unknown).

Per layer \(\ell\),

\[
u_\ell = \mathrm{normalize}\big(\bar h^{\mathrm{late}}_\ell - \bar h^{\mathrm{early}}_\ell\big).
\]

**Localize writers.** Rank layers by residual gap \(\|\bar h^{\mathrm{late}}-\bar h^{\mathrm{early}}\|\) and by positive write mass \(\sum_c \mathrm{ReLU}(s_c)\), \(s = W^\top u\), \(W\) = `mlp.down_proj.weight`. Combined score picks candidates; **we do not hard-code 15/19**.

**Edit.** For selected layers and top-\(k\) columns by \(s_c\),

\[
W' = W - \alpha\, u\,(m \odot s)^\top.
\]

**Selection.** Sweep layer sets / \(\alpha\) / \(k\) / direction variants on held-out practice **without** early-stop. Freeze only if the winner beats both base and the prior freeze on that sweep score; then smoke.

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
| Diagnose max new tokens | v1: 640; v2: 512 (early-stop for throughput) |
| Sweep max new tokens | 512, **stop_on_repeat=False** |
| Smoke max new tokens | **CPU v1:** 1024; **XPU v1:** **4096**; **CPU v2:** 768 (n=12 cap) |
| Sampling | temperature 0.6, top_p 0.95, top_k 20 (loop edit) |

Early-stop during *diagnose* compresses wall time but collapses severity differences if used for *selection*. After an initial failed sweep that looked uniformly “early=True,” we switched selection/eval to full capped generations and track `first_repeat_at` / `mean_max_8gram` / `no_repeat_rate`.

### 4.2 Phase 1 configurations

**v1:** layer sets at α=8, k=64: `{23,22}`, `{21,20}`, `{18,19}`, `{23,22,21,20}`; then α ∈ {6,12} refine on the winner.

**v2** (`loop_edit/discover_v2.py`, CPU): refreshed diagnose (n=20, max-new 512, early-stop on); three direction banks (`gen_early_late` 64/64, `preloop_loop` 64/96, `gen_early_late` 32/96); 16 scheduled candidates spanning single layers, pairs including `{21,19}`, `{17,16}`, `{18,19}`, shared-\(u\) ablations, and the prior `{21,20}`; α=8 k=64 primary grid. MATH-500 unused.

### 4.3 Phase 2 configurations

Short GRPO-style mask train: see Appendix B and `deletion_tuning/results/phase2_report.json` (filled after train).

---

## 5. Results

### 5.1 Diagnose (practice, thinking on)

**v1** (early-stop diagnose, prior run):

| Metric | Value (\(n=20\)) |
| --- | ---: |
| Hit-cap rate | 0.10 |
| Early-repeat rate (threshold 3 on 8-grams) | 0.90 |
| Boxed rate | 0.25 |
| Finished-clean rate | 0.00 |
| Mean tokens (with early-stop) | 330.4 |

**v2 refresh** (`diagnose_summary_v2.json`, max-new 512, early-stop on): hit-cap 0.20; early-repeat 0.85; boxed 0.25; finished-clean 0.00; mean tokens 311.7; no-repeat 0.15. Loops still dominate.

### 5.2 Localization (our generations only)

**v1:** top residual gaps **23 > 22 > 21 > 20 > 19 > 18 > … > 15**. Combined gap + positive write mass still ranks 23/22 first.

**v2** (`localization_v2.json`): with `gen_early_late` / `preloop_loop`, gap leaders remain **23 > 22 > 21**, then 19/20/18 depending on window. Combined still ranks 23/22 first. **Edit selection continues to disagree** (next subsection).

### 5.3 Layer / α sweep — v1 (held-out practice, \(n=8\), max 512, no early-stop)

| Candidate | early_repeat | no_repeat | mean_first_repeat | mean_max_8gram | boxed | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1.000 | 0.000 | 273.5 | 3.63 | 0.25 | −1.259 |
| α=8 k=64 **L21–20** (v1 freeze) | **0.875** | **0.125** | 277.3 | **3.38** | 0.25 | **−0.989** |
| α=6 k=64 L21–20 | 1.000 | 0.000 | **306.3** | 3.50 | 0.25 | −1.188 |
| α=8 k=64 L18–19 | 1.000 | 0.000 | 288.4 | 3.75 | 0.25 | −1.236 |
| α=12 k=64 L21–20 | 1.000 | 0.000 | 294.9 | 3.88 | 0.125 | −1.417 |
| α=8 k=64 L23–22 | 1.000 | 0.000 | 184.8 | **128.3** | 0.00 | −7.918 |
| α=8 k=64 L23–22–21–20 | 1.000 | 0.000 | 205.0 | **213.1** | 0.00 | −12.25 |

**v1 frozen edit** (archived `frozen_edit_v1_L21-20.json`): layers **{21, 20}**, **α = 8**, **k = 64**. Beats base on selection; gap-top 23/22 induces tighter verbatim loops.

### 5.3b v2 discovery sweep (same split, max 512, no early-stop)

Same base/old scores re-measured under the v2 runner. Selected rows from `discover_v2_report.json`:

| Candidate | early_repeat | no_repeat | mean_first_repeat | mean_max_8gram | boxed | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1.000 | 0.000 | 273.5 | 3.63 | 0.25 | −1.259 |
| Old freeze L21–20 (v1 \(u\)) | 0.875 | 0.125 | 277.3 | 3.38 | 0.25 | −0.989 |
| `gen_early_late` α=8 k=64 L21–20 | 1.000 | 0.000 | 300.4 | 4.38 | 0.00 | −1.618 |
| `gen_early_late` α=8 k=64 L17–16 | 0.750 | 0.250 | 286.5 | 3.38 | 0.125 | −0.908 |
| `gen_early_late` α=8 k=64 L21 | 0.750 | 0.250 | 275.7 | 3.25 | 0.125 | −0.924 |
| **`preloop_loop` α=8 k=64 L21–19** | **0.750** | **0.250** | **336.8** | **3.00** | **0.25** | **−0.601** |
| `preloop_loop` α=8 k=64 L18–19 | 0.875 | 0.125 | 283.0 | 3.50 | 0.25 | −0.984 |

**v2 frozen edit** (`frozen_edit.json` / `frozen_edit_v2.json`): layers **{21, 19}**, **α = 8**, **k = 64**, method `v2_preloop_loop_e64_l96`. Clearly beats base **and** the v1 freeze on the selection score. Rebuilding \(u\) with gen-relative windows on the *old* layer pair L21–20 *hurts*—direction quality and layer choice interact. Shared-\(u\) ablations underperformed per-layer \(u\).

### 5.4 Smoke eval vs base

#### v1 freeze (L21–20) — prior CPU / XPU smokes

**CPU smoke** (`smoke_report_cpu.json`): max-new **1024**, n=25. Edited beats base on composite (`beats_base: true`): hit-cap 0.96 vs 1.0; early-repeat 0.96 vs 1.0; no-repeat 0.04 vs 0.0; boxed 0.24 vs 0.20.

**XPU / Arc smoke** (`smoke_report_gpu.json`): Intel Arc 140V, bf16, max-new **4096**, n=25. Hit-caps stay near 1.0; boxed falls 0.40→0.32; mean max-8-gram rises; `beats_base: false`—honest null/negative at the longer horizon (motivated the v2 pass).

#### v2 freeze (L21–19 preloop/loop) — CPU smoke n=12, max-new 768

Same seed offsets; compare base / v1 freeze / v2 freeze (`smoke_report_v2.json`):

| Metric | Base | Old L21–20 | **New L21–19** |
| --- | ---: | ---: | ---: |
| \(n\) | 12 | 12 | 12 |
| Hit-cap rate | 1.00 | 1.00 | 1.00 |
| Early-repeat rate | 1.00 | **0.917** | 1.00 |
| No-repeat rate | 0.00 | **0.083** | 0.00 |
| Mean first-repeat | 317.9 | 294.8 | **333.8** |
| Mean max 8-gram | 5.00 | 5.67 | **4.50** |
| Boxed rate | 0.167 | **0.25** | 0.167 |
| `beats_base` (code composite) | — | (not reflagged) | **false** |

Selection win for L21–19 does **not** automatically yield a composite smoke win at this budget: hit-cap and early-repeat stay saturated at 1.0 for base and new, while soft severity (first-repeat, max-8-gram) moves in the right direction. On this small smoke slice the *old* freeze still looks better on early-repeat / boxed—another reminder that n=8 selection and n=12 smoke can disagree. Not a MATH-500 claim.

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

**Localization ≠ selection.** The largest early→late residual gaps sit at layers 23/22, but editing them with our \(u\) and α=8 produced pathological tight loops. v1 selection preferred 21/20; v2 selection preferred **21/19** under a pre-loop/loop direction. Gap-argmax remains a poor freeze rule without a behavioral sweep.

**Direction quality matters as much as layers.** Applying a new gen-relative \(u\) on the old L21–20 pair *worsened* the sweep score versus base. The winning v2 edit couples a better contrast (`preloop_loop`) with layers **21/19**—not a drop-in replacement of tensors into the v1 recipe.

**Selection ≠ smoke.** v2 clearly beats base and v1 on the n=8 selection score, but the n=12 / max-768 CPU smoke does not flip the composite `beats_base` flag. Soft metrics move (later first-repeat, lower mean max-8-gram); early-repeat stays saturated. On that smoke slice v1 still looks stronger on early-repeat/boxed. We report both.

**Not Astra’s layers—and not a claim of superiority.** We did not import Astra or CLC tensors. We make no MATH-500 comparison in this draft.

**Smoke protocol / hardware.** v1 long smoke used Intel Arc 140V (**XPU**) at max-new 4096 and was null/negative—prompting v2. v2 discovery/smoke here is CPU-only. NVIDIA CUDA remains unavailable on this host.

**Deletion Tuning.** Without GPU verl, we could not match paper GRPO. Our delivered mask is a sparse Wanda-style delete on `gate_proj` (0.021% of model weights) with an honest tiny-set null vs base reward.

**Sample sizes.** Diagnose 20 / sweep 8 / smoke 12–25. Scores are noisy; the robust qualitative takeaways are (i) 23/22 harmful at α=8, (ii) contrast window choice changes which mid-late layers win, (iii) selection wins need separate smoke confirmation.

---

## 7. Conclusion

We derived thinking-loop weight edits and a deletion mask from our own rollouts. The **current** frozen loop edit is **`down_proj` on layers 21 and 19, α=8, k=64`**, with \(u\) from a **pre-loop vs loop-tail** contrast on our generations (`v2_preloop_loop_e64_l96`). It beats both base and the prior L21–20 freeze on the held-out selection score; a small CPU smoke shows only soft severity gains (`beats_base: false` on the composite). The prior L21–20 freeze remains archived for comparison. Deletion Tuning yields our own packbits mask with reported delete fraction and tiny-set behavior. The central methodological point stands: **measure your edit; do not paste a published tensor into the discovery loop.**

---

## Appendix A — Loop-edit artifacts

- Code: `loop_edit/` (`run_phase1.py`, `discover_v2.py`, `sweep_layers.py`, `smoke_eval.py`)
- Diagnose: `loop_edit/results/diagnose_summary.json`, `diagnose_summary_v2.json`
- Localization: `loop_edit/results/localization.json`, `localization_v2.json`
- Directions: `loop_edit/results/directions_v2_preloop_loop_e64_l96.npz` (and sibling banks)
- Layer sweep (v1): `loop_edit/results/layer_sweep_report.json`
- Discovery report (v2): `loop_edit/results/discover_v2_report.json`
- Frozen specs: `loop_edit/results/frozen_edit.json` (current=v2), `frozen_edit_v2.json`, `frozen_edit_v1_L21-20.json`
- Smoke (v1 CPU / XPU): `smoke_report.json` / `smoke_report_cpu.json` / `smoke_report_gpu.json`
- Smoke (v2 CPU n=12): `loop_edit/results/smoke_report_v2.json`

## Appendix B — Deletion-tuning artifacts

- Pin: `deletion_tuning/PINNED_COMMIT.txt`
- Trainer: `deletion_tuning/train_mask.py`
- Mask + report: `deletion_tuning/results/our_mask/`, `phase2_report.json`

## Appendix C — Reproducibility commands

```bash
PYTHONPATH=. python -m loop_edit.run_phase1 --reuse-diagnose
PYTHONPATH=. python -m loop_edit.sweep_layers --alpha 8 --k 64 --max-new 512
PYTHONPATH=. python -m loop_edit.discover_v2 --device cpu --max-candidates 16 --smoke-max-tokens 768 --n-smoke-cap 12
PYTHONPATH=. python -m loop_edit.smoke_eval --max-new 1024
PYTHONPATH=. python -m loop_edit.smoke_eval --device xpu --max-new 4096 --resume
python deletion_tuning/train_mask.py --model Qwen/Qwen2.5-0.5B --steps 20
```
