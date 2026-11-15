#!/usr/bin/env python3
"""Apply a packbits_zstd Deletion Tuning mask (ours or upstream) to a base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
import zstandard
from safetensors.torch import load_file


def load_mask(mask_dir: str) -> dict[str, torch.Tensor]:
    with open(Path(mask_dir) / "config.json") as f:
        fmt = json.load(f)["format"]
    assert fmt == "packbits_zstd", fmt
    tensors = load_file(str(Path(mask_dir) / "model.safetensors"), device="cpu")
    dctx = zstandard.ZstdDecompressor()
    masks = {}
    for key in tensors:
        if not key.endswith(".shape"):
            continue
        name = key[: -len(".shape")]
        shape = tuple(tensors[key].tolist())
        numel = int(np.prod(shape))
        packed = np.frombuffer(dctx.decompress(tensors[f"{name}.packed"].numpy().tobytes()), dtype=np.uint8)
        masks[name] = torch.from_numpy(np.unpackbits(packed)[:numel].astype(bool)).reshape(shape)
    return masks


@torch.no_grad()
def apply_mask(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> int:
    params = dict(model.named_parameters())
    deleted = 0
    for name, keep in masks.items():
        weight = params[name]
        weight.mul_(keep.to(device=weight.device, dtype=weight.dtype))
        deleted += int((~keep).sum())
    return deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    deleted = apply_mask(model, load_mask(args.mask_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    transformers.AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output_dir)
    print(f"deleted {deleted:,} weights; saved to {args.output_dir}")


if __name__ == "__main__":
    main()
