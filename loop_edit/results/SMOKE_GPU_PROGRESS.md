# XPU smoke — complete

**Status:** finished. Full report: `smoke_report_gpu.json` (`incomplete: false`).

## Protocol

- Device: Intel Arc 140V via PyTorch XPU (`torch 2.15.0+xpu`), bf16
- Frozen edit: layers `{21,20}`, α=8, k=64
- `max-new=4096`, `stop_on_repeat=False`, full `split_smoke.jsonl` n=25
- Soft-resumed after laptop sleep / one XPU OOM mid-base

## Verdict

`beats_base: false` at 4096. Hit-caps stay near 1.0; later first-repeat is the main movement; boxed and 8-gram severity do not improve vs base.

See `docs/weight-editing-paper.md` §5.4 for CPU vs XPU tables.
