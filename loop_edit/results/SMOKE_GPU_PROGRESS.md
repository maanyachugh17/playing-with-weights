# XPU smoke checkpoint (incomplete)

**Status:** soft-stopped before laptop sleep. **Not** a finished GPU smoke.

## Protocol (unchanged target)

- Device: Intel Arc 140V via PyTorch XPU (`torch 2.15.0+xpu`), bf16
- Frozen edit: layers `{21,20}`, α=8, k=64 (`frozen_edit.json`)
- `max-new=4096`, `stop_on_repeat=False`, full `split_smoke.jsonl` **n=25**
- Compare base vs edited

## Done

| Phase | Progress | Artifact |
| --- | --- | --- |
| Base | **7 / 25** | `smoke_base_gpu.jsonl` |
| Edited | **0 / 25** | (not started) |

Base ids completed:

`gsm8k-291`, `gsm8k-227`, `gsm8k-242`, `gsm8k-108`, `gsm8k-90`, `gsm8k-262`, `gsm8k-287`

Partial metrics file: `smoke_report_gpu.partial.json` (`incomplete: true`).  
Do **not** treat as `smoke_report_gpu.json`.

## Remaining

- Base: 18 problems
- Edited: all 25 problems
- Then write final `smoke_report_gpu.json` + paper update + PR

## Resume (from repo root, after activating the XPU venv)

```powershell
cd C:\Users\maany\playing-with-weights-gpu-smoke\repo
C:\Users\maany\playing-with-weights-gpu-smoke\.venv\Scripts\python.exe -m loop_edit.smoke_eval --device xpu --max-new 4096 --resume
```

`--resume` keeps existing `smoke_base_gpu.jsonl` / `smoke_edited_gpu.jsonl` rows and only generates missing ids (same seed offsets).
