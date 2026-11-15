#!/usr/bin/env python3
"""Short Deletion Tuning train: sparse binary mask over MLP gate_proj.

Writes packbits_zstd masks compatible with deleting-weights apply_mask.py.
CPU-friendly / reduced-steps; not verl paper parity.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import zstandard as zstd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_gsm8k(n: int, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = [{"q": ex["question"], "a": ex["answer"].split("####")[-1].strip()} for ex in ds]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def extract_boxed(text: str) -> str | None:
    import re

    ms = list(re.finditer(r"\\boxed\s*\{([^{}]*)\}", text))
    return ms[-1].group(1).strip() if ms else None


def reward_fn(text: str, gold: str) -> float:
    pred = extract_boxed(text)
    def norm(x: str) -> str:
        return "".join(x.split()).lower().replace(",", "")

    r = 0.0
    if pred is not None:
        r += 0.3
        if norm(pred) == norm(gold):
            r += 0.7
    elif any(c.isdigit() for c in text):
        r += 0.05
    # Mild length shaping: prefer non-empty, non-degenerate short answers under cap
    if len(text) < 8:
        r -= 0.05
    if len(set(text)) < 4 and len(text) > 20:
        r -= 0.1  # garbage loops of one char
    return r


class GateMask(nn.Module):
    """Soft→hard mask over one frozen gate_proj weight (Deletion Tuning style)."""

    def __init__(self, weight: nn.Parameter, sharpness=(1.0, 25.0), total_steps: int = 50, init_keep: float = 2.0):
        super().__init__()
        self.register_buffer("weight_buf", weight.detach().clone())
        # Keep Parameter reference for saving shapes; frozen.
        self.weight = weight
        self.weight.requires_grad_(False)
        # Keep-biased init so hard masks start near-dense (paper deletes ≪1%).
        self.gate = nn.Parameter(torch.full((weight.numel(),), float(init_keep), dtype=torch.float32))
        self.gate.data += 0.01 * torch.randn_like(self.gate)
        self.sharpness = sharpness
        self.total_steps = total_steps
        self.current_step = 0

    def progress(self) -> float:
        return min(self.current_step / max(self.total_steps, 1), 1.0)

    def mask(self) -> torch.Tensor:
        if self.training:
            s = self.sharpness[0] + (self.sharpness[1] - self.sharpness[0]) * self.progress()
            m = torch.sigmoid(self.gate * s)
        else:
            m = (self.gate > 0).float()
        return m.view_as(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask().to(dtype=self.weight.dtype, device=self.weight.device)
        return F.linear(x, w)

    def step(self) -> None:
        self.current_step += 1


def convert_gate_projs(model: nn.Module, total_steps: int, init_keep: float) -> list[GateMask]:
    masks: list[GateMask] = []
    for name, module in list(model.named_modules()):
        if name.endswith("mlp.gate_proj") and isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0]
            parent = model.get_submodule(parent_name)
            gm = GateMask(module.weight, total_steps=total_steps, init_keep=init_keep)
            setattr(parent, "gate_proj", gm)
            masks.append(gm)
    return masks


def save_compressed_mask(masks: list[tuple[str, GateMask]], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    deleted = 0
    total = 0
    cctx = zstd.ZstdCompressor(level=3)
    for name, gm in masks:
        keep = (gm.gate.detach() > 0).cpu().numpy().astype(np.uint8).reshape(gm.weight.shape)
        deleted += int((1 - keep).sum())
        total += keep.size
        packed = np.packbits(keep.reshape(-1))
        compressed = cctx.compress(packed.tobytes())
        tensors[f"{name}.shape"] = torch.tensor(list(keep.shape), dtype=torch.int64)
        tensors[f"{name}.packed"] = torch.from_numpy(np.frombuffer(compressed, dtype=np.uint8).copy())
    save_file(tensors, str(out_dir / "model.safetensors"))
    meta = {
        "format": "packbits_zstd",
        "deleted": deleted,
        "total_masked_params": total,
        "deleted_fraction_of_masked": deleted / max(total, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@torch.no_grad()
def generate_completion(model, tokenizer, question: str, max_new: int, device: str) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + " Put the final answer in \\boxed{}."}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = (
            f"<|im_start|>user\n{question} Put the final answer in \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def teacher_force_loss(model, tokenizer, question: str, gold: str, device: str) -> torch.Tensor:
    """Differentiable proxy: CE on a short supervised target with soft masks active."""
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + " Put the final answer in \\boxed{}."}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"Question: {question}\nAnswer: "
    target = f"The answer is \\boxed{{{gold}}}."
    full = prompt + target
    enc = tokenizer(full, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    # Labels: mask prompt tokens
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    out = model(input_ids=input_ids, labels=labels)
    return out.loss


def eval_base_vs_masked(model, tokenizer, problems, device, max_new) -> dict:
    model.eval()
    rows = []
    for ex in problems:
        with torch.no_grad():
            text = generate_completion(model, tokenizer, ex["q"], max_new=max_new, device=device)
        rows.append({"reward": reward_fn(text, ex["a"]), "boxed": extract_boxed(text) is not None, "n_chars": len(text)})
    return {
        "n": len(rows),
        "mean_reward": sum(r["reward"] for r in rows) / max(len(rows), 1),
        "boxed_rate": sum(r["boxed"] for r in rows) / max(len(rows), 1),
        "mean_chars": sum(r["n_chars"] for r in rows) / max(len(rows), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--train-n", type=int, default=48)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--target-delete", type=float, default=0.001, help="Target fraction of gate weights to delete")
    ap.add_argument("--delete-coef", type=float, default=2.0)
    ap.add_argument("--ce-coef", type=float, default=1.0)
    ap.add_argument("--init-keep", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, trust_remote_code=True)
    model.to(device)

    data = load_gsm8k(args.train_n + args.eval_n, seed=args.seed)
    train, held = data[: args.train_n], data[args.train_n :]

    print("=== Base eval ===", flush=True)
    base_stats = eval_base_vs_masked(model, tok, held, device=device, max_new=args.max_new)
    print(base_stats, flush=True)

    masks = convert_gate_projs(model, total_steps=args.steps, init_keep=args.init_keep)
    print(f"Wrapped {len(masks)} gate_proj layers (init_keep={args.init_keep})", flush=True)
    named = []
    for name, module in model.named_modules():
        if isinstance(module, GateMask):
            named.append((f"{name}.weight", module))

    opt = torch.optim.Adam([m.gate for m in masks], lr=args.lr)
    metrics = []
    for step in range(args.steps):
        model.train()
        for m in masks:
            m.train()
        batch = random.sample(train, k=min(args.batch_size, len(train)))
        opt.zero_grad()
        ce = torch.zeros((), device=device)
        for ex in batch:
            ce = ce + teacher_force_loss(model, tok, ex["q"], ex["a"], device=device)
        ce = ce / max(len(batch), 1)

        soft_keeps = torch.stack([m.mask().mean() for m in masks])
        mean_keep = soft_keeps.mean()
        # Encourage keep rate ≈ 1 - target_delete
        target_keep = 1.0 - args.target_delete
        delete_pen = (mean_keep - target_keep).pow(2)

        loss = args.ce_coef * ce + args.delete_coef * delete_pen
        loss.backward()
        opt.step()
        for m in masks:
            m.step()

        # Rollout reward probe (no grad) for logging
        with torch.no_grad():
            model.eval()
            for m in masks:
                m.eval()
            probe = generate_completion(model, tok, batch[0]["q"], max_new=args.max_new, device=device)
            probe_r = reward_fn(probe, batch[0]["a"])
            hard_keep = float(torch.stack([(m.gate > 0).float().mean() for m in masks]).mean().cpu())

        stats = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "ce": float(ce.detach().cpu()),
            "delete_pen": float(delete_pen.detach().cpu()),
            "mean_soft_keep": float(mean_keep.detach().cpu()),
            "mean_hard_keep": hard_keep,
            "probe_reward": probe_r,
        }
        metrics.append(stats)
        print(
            f"step {step}: ce={stats['ce']:.3f} keep_soft={stats['mean_soft_keep']:.4f} "
            f"keep_hard={hard_keep:.4f} probe_r={probe_r:.3f}",
            flush=True,
        )

    for m in masks:
        m.eval()
    model.eval()
    print("=== Masked eval ===", flush=True)
    masked_stats = eval_base_vs_masked(model, tok, held, device=device, max_new=args.max_new)
    print(masked_stats, flush=True)

    mask_dir = RESULTS / "our_mask"
    meta = save_compressed_mask(named, mask_dir)
    n_params = sum(p.numel() for p in model.parameters())
    # Count only unique stored weights roughly via state dict sizes
    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "model": args.model,
        "device": device,
        "steps": args.steps,
        "train_n": args.train_n,
        "batch_size": args.batch_size,
        "max_new": args.max_new,
        "target_delete": args.target_delete,
        "init_keep": args.init_keep,
        "elapsed_sec": time.time() - t0,
        "mask": meta,
        "deleted_fraction_of_model": meta["deleted"] / max(n_params, 1),
        "base_eval": base_stats,
        "masked_eval": masked_stats,
        "metrics": metrics,
        "mask_dir": str(mask_dir),
        "note": "Reduced-steps CPU train with keep-biased init + CE + target-delete penalty; not verl GRPO parity.",
    }
    (RESULTS / "phase2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
