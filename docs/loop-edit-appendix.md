# Appendix — Loop-edit metrics (pointer)

Primary narrative: [`weight-editing-paper.md`](weight-editing-paper.md).

**Frozen edit:** layers `{21, 20}`, α=`8`, k=`64` (our early/late \(u\); not Astra tensors).

| Stage | Signal |
| --- | --- |
| Diagnose \(n=20\) | early_repeat 0.90, boxed 0.25, finished_clean 0.00 |
| Sweep winner vs base \(n=8\) | early_repeat 0.875 vs 1.0; no_repeat 0.125 vs 0.0 |
| Smoke \(n=25\), max 1024 | hit_cap 0.96 vs 1.0; boxed 0.24 vs 0.20; beats_base true |

Repo: `loop_edit/results/{localization,layer_sweep_report,frozen_edit,smoke_report}.json`
