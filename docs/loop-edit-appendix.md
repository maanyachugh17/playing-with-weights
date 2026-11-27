# Appendix — Loop-edit metrics (pointer)

Primary narrative: [`weight-editing-paper.md`](weight-editing-paper.md).

**Current frozen edit (v2):** layers `{21, 19}`, α=`8`, k=`64`, method `v2_preloop_loop_e64_l96` (our pre-loop/loop \(u\); not Astra tensors).

**Archived v1 freeze:** layers `{21, 20}`, α=`8`, k=`64` → `loop_edit/results/frozen_edit_v1_L21-20.json`.

| Stage | Signal |
| --- | --- |
| Diagnose v2 \(n=20\) | early_repeat 0.85, boxed 0.25, finished_clean 0.00 |
| Sweep winner vs base \(n=8\) | score −0.601 vs −1.259; early_repeat 0.75 vs 1.0; no_repeat 0.25 vs 0.0 |
| Sweep winner vs v1 freeze | −0.601 vs −0.989 (same protocol) |
| Smoke v2 \(n=12\), max 768 | soft severity gains; composite `beats_base: false` |

Repo: `loop_edit/results/{localization_v2,discover_v2_report,frozen_edit,frozen_edit_v2,smoke_report_v2}.json`
