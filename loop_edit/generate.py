"""Thinking-mode generation with early 8-gram repetition stop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .metrics import GenMetrics, max_ngram_count, score_generation


class EightGramRepeatStop(StoppingCriteria):
    """Stop when any 8-gram in newly generated tokens repeats `threshold` times."""

    def __init__(self, prompt_len: int, threshold: int = 3, n: int = 8):
        self.prompt_len = prompt_len
        self.threshold = threshold
        self.n = n
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen = input_ids[0, self.prompt_len :].tolist()
        if len(gen) < self.n * self.threshold:
            return False
        # Check only recent window for speed.
        window = gen[-(self.n * 32) :]
        if max_ngram_count(window, n=self.n) >= self.threshold:
            self.triggered = True
            return True
        return False


def resolve_device(device: str = "auto") -> str:
    """Prefer XPU (Intel Arc), then CUDA, else CPU."""
    if device != "auto":
        return device
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def default_dtype_for_device(device: str) -> torch.dtype:
    if device in ("xpu", "cuda"):
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    dtype: torch.dtype | None = None,
    device: str = "cpu",
):
    if dtype is None:
        dtype = default_dtype_for_device(device)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model, tok


def format_prompt(tokenizer, question_prompt: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": question_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 0.0,
    seed: int | None = 0,
    early_repeat_threshold: int = 3,
    stop_on_repeat: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    if seed is not None:
        torch.manual_seed(seed)
    text_prompt = format_prompt(tokenizer, prompt, enable_thinking=True)
    inputs = tokenizer(text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    stopper = EightGramRepeatStop(prompt_len=prompt_len, threshold=early_repeat_threshold)

    gen_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if stop_on_repeat:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
    if temperature and temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    else:
        gen_kwargs["do_sample"] = False

    # presence_penalty is not always supported by HF generate; skip if unavailable.
    out = model.generate(**gen_kwargs)
    gen_ids = out[0, prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    metrics = score_generation(
        text=text,
        token_ids=gen_ids,
        max_new_tokens=max_new_tokens,
        early_repeat_threshold=early_repeat_threshold,
    )
    # Treat early-stop repetition as a loop even if under token cap.
    if stop_on_repeat and stopper.triggered:
        metrics.early_repeat = True
        metrics.finished_clean = False
    return {
        "prompt": prompt,
        "text": text,
        "token_ids": gen_ids,
        "metrics": metrics.to_dict(),
        "stopped_on_repeat": bool(stop_on_repeat and stopper.triggered),
    }


def generate_dataset(
    model,
    tokenizer,
    problems: list[dict],
    out_path: Path,
    max_new_tokens: int = 1024,
    seed: int = 0,
    device: str = "cpu",
    early_repeat_threshold: int = 3,
    stop_on_repeat: bool = True,
) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, prob in enumerate(problems):
        row = generate_one(
            model,
            tokenizer,
            prompt=prob["prompt"],
            max_new_tokens=max_new_tokens,
            seed=seed + i,
            device=device,
            early_repeat_threshold=early_repeat_threshold,
            stop_on_repeat=stop_on_repeat,
        )
        row.update({"id": prob["id"], "source": prob.get("source"), "answer": prob.get("answer")})
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as f:
            # Drop huge token lists from the live log copy? Keep them for activation reuse.
            f.write(json.dumps({k: v for k, v in row.items()}, ensure_ascii=False) + "\n")
        m = GenMetrics(**row["metrics"])
        print(
            f"[{i+1}/{len(problems)}] {prob['id']} tokens={m.n_tokens} "
            f"hit_cap={m.hit_cap} early={m.early_repeat} first_rep={m.first_repeat_at} "
            f"m8={m.max_8gram_count} boxed={m.has_boxed}",
            flush=True,
        )
    return rows
