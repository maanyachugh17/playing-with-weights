# Appendix — Deletion Tuning metrics (pointer)

Primary narrative: [`weight-editing-paper.md`](weight-editing-paper.md).

**Our mask:** delete 104,595 `gate_proj` weights (0.1% of gate params; 0.021% of model) via \(\lvert W\rvert\cdot a\) on our calibration set. Format `packbits_zstd`.

| Eval \(n=16\) | Base | Masked |
| --- | ---: | ---: |
| Mean reward | 0.106 | 0.047 |
| Boxed rate | 0.0625 | 0.000 |

Pin: `deletion_tuning/PINNED_COMMIT.txt`. Report: `deletion_tuning/results/phase2_report.json`.
