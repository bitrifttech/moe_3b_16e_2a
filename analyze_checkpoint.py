#!/usr/bin/env python3
"""
Analyze a GPT-style checkpoint (including GPT2WithMoE) and optionally run quick
quality probes.

Usage
-----
python analyze_checkpoint.py /path/to/checkpoint \
       [--device cuda] [--samples 100] \
       [--summarise] [--truthfulqa] [--hellaswag]

Quick metrics implemented
1. Model report (architecture, parameter counts, MoE specifics)
2. PPL on Wikitext-2 subset (always)
3. ROUGE-L & BLEU on CNN/DailyMail summarisation (--summarise)
4. TruthfulQA MC accuracy             (--truthfulqa)
5. HellaSwag MC accuracy              (--hellaswag)

The optional suites use HuggingFace `datasets` & `evaluate`.  They will be
skipped gracefully if those packages or datasets are unavailable.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

# -----------------------------------------------------------------------------
# Helpers to handle both dict and ModelOutput
# -----------------------------------------------------------------------------

def _extract_logits(outputs):
    """Return logits tensor from HF ModelOutput **or** custom dict."""
    if outputs is None:
        raise ValueError("Model returned None outputs")
    if isinstance(outputs, dict):
        return outputs.get("logits")
    # HuggingFace models return CausalLMOutputWithCrossAttentions or similar
    return getattr(outputs, "logits", None)

def _greedy_generate(model, tokenizer, input_ids, max_new_tokens: int = 60, device: torch.device = None):
    """Very small helper that performs greedy decoding when `model.generate` is unavailable."""
    if hasattr(model, "generate"):
        try:
            return model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        except AttributeError:
            pass  # fall through to manual greedy

    if device is None:
        device = input_ids.device
    cur_ids = input_ids
    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(cur_ids)
        logits = _extract_logits(outputs)[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_id], dim=-1)
    return cur_ids

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def human_count(n: int) -> str:
    """Return a human-readable parameter count (e.g. 1.3B)."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def load_model(checkpoint: str, device: torch.device):
    """Load a checkpoint. Tries custom GPT2WithMoE first, then generic."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint)

    # Auto detection of custom architecture
    try:
        from moe_model import GPT2WithMoE  # noqa: F401
        model_cls = GPT2WithMoE
        model = model_cls.from_pretrained(checkpoint_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        model.to(device)
        return model, tokenizer
    except Exception:
        # Fallback to generic
        cfg = AutoConfig.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=cfg)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        model.to(device)
        return model, tokenizer


def report_model(model, tokenizer):
    cfg = model.config
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=========== MODEL REPORT ===========")
    print(f"Model class     : {model.__class__.__name__}")
    print(f"Config class    : {cfg.__class__.__name__}")
    print(f"Vocabulary size : {getattr(cfg, 'vocab_size', 'n/a')}")
    for attr in ["n_layer", "n_embd", "n_head", "n_inner", "num_hidden_layers"]:
        if hasattr(cfg, attr):
            print(f"{attr:<15}: {getattr(cfg, attr)}")
    # MoE-specific
    for attr in ["num_experts", "expert_interval", "moe_layer_freq", "top_k"]:
        if hasattr(cfg, attr):
            print(f"{attr:<15}: {getattr(cfg, attr)}")

    print(f"Total parameters: {human_count(total)}")
    print(f"Trainable params: {human_count(trainable)}")
    bytes_fp32 = total * 4
    print(f"Memory (fp32)   : {bytes_fp32/1e9:.2f} GB")
    print("====================================\n")

# -----------------------------------------------------------------------------
# Quick perplexity on Wikitext-2
# -----------------------------------------------------------------------------

def perplexity(model, tokenizer, device: torch.device, samples: int = 100, max_len: int = 128) -> float:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = ds["text"]
    # Remove empty lines
    texts = [t for t in texts if t.strip()][: samples]

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, truncation=True, max_length=max_len)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids[:-1]], device=device)
            target_ids = torch.tensor([ids[1:]], device=device)
            logits = _extract_logits(model(input_ids))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            losses.append(loss.item())
    ppl = math.exp(sum(losses) / len(losses))
    return ppl

# -----------------------------------------------------------------------------
# Summarisation – ROUGE / BLEU
# -----------------------------------------------------------------------------

def summarisation_metrics(model, tokenizer, device: torch.device, samples: int = 50, max_new_tokens: int = 60):
    try:
        from datasets import load_dataset
        import evaluate
    except ImportError:
        print("Skipping summarisation metrics – install 'datasets evaluate rouge-score sacrebleu' to enable.")
        return None

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="validation[:{}]".format(samples))

    preds, refs = [], []
    model.eval()
    for article, ref in zip(ds["article"], ds["highlights"]):
        prompt = f"Summarize: {article.strip()}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = _greedy_generate(model, tokenizer, inputs.input_ids, max_new_tokens=max_new_tokens)
        summary = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(summary)
        refs.append(ref)

    rouge_res = rouge.compute(predictions=preds, references=refs, use_aggregator=True)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    return rouge_res, bleu_res

# -----------------------------------------------------------------------------
# Multiple-choice scoring helpers
# -----------------------------------------------------------------------------

def choice_logprob(model, tokenizer, context: str, choice: str, device: torch.device, max_ctx: int = 256) -> float:
    # Returns total log prob of `choice` tokens conditioned on `context`.
    full_text = context + " " + choice
    full_ids = tokenizer(full_text, truncation=True, max_length=max_ctx, return_tensors="pt").input_ids[0].to(device)
    ctx_ids = tokenizer(context, truncation=True, max_length=max_ctx, return_tensors="pt").input_ids[0].to(device)
    # Ensure full_ids longer than ctx_ids
    if full_ids.shape[0] <= ctx_ids.shape[0] + 1:
        return -float("inf")
    labels = full_ids.clone()
    labels[: ctx_ids.shape[0]] = -100  # mask context
    with torch.no_grad():
        logits = _extract_logits(model(full_ids.unsqueeze(0))).squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    choice_logp = 0.0
    for idx in range(ctx_ids.shape[0], full_ids.shape[0]):
        token_id = full_ids[idx]
        choice_logp += log_probs[idx - 1, token_id].item()  # predict token idx from idx-1 logits
    return choice_logp

# -----------------------------------------------------------------------------
# TruthfulQA evaluation
# -----------------------------------------------------------------------------

def truthfulqa_eval(model, tokenizer, device: torch.device, samples: int = 50):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skipping TruthfulQA – install 'datasets' to enable.")
        return None
    ds = load_dataset("truthful_qa", "multiple_choice")
    ds = ds["validation"].select(range(samples))

    correct = 0
    for ex in ds:
        ctx = ex["question"]
        choices = ex["answer_choices"]
        gold = ex["correct_answers"]
        scores = [choice_logprob(model, tokenizer, ctx, c, device) for c in choices]
        picked = choices[int(torch.tensor(scores).argmax())]
        if picked in gold:
            correct += 1
    acc = correct / len(ds)
    return acc

# -----------------------------------------------------------------------------
# HellaSwag evaluation
# -----------------------------------------------------------------------------

def hellaswag_eval(model, tokenizer, device: torch.device, samples: int = 100):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skipping HellaSwag – install 'datasets' to enable.")
        return None
    ds = load_dataset("hellaswag", split="validation[:{}]".format(samples))

    correct = 0
    for ex in ds:
        ctx = ex["ctx_a"] + " " + ex["ctx_b"]
        choices = ex["endings"]
        label = int(ex["label"])
        scores = [choice_logprob(model, tokenizer, ctx, c, device) for c in choices]
        picked = int(torch.tensor(scores).argmax())
        if picked == label:
            correct += 1
    acc = correct / len(ds)
    return acc

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint & run quick tests")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples for quick metrics")
    parser.add_argument("--summarise", action="store_true", help="Run CNN/DailyMail summarisation test")
    parser.add_argument("--truthfulqa", action="store_true", help="Run TruthfulQA multiple-choice test")
    parser.add_argument("--hellaswag", action="store_true", help="Run HellaSwag multiple-choice test")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_model(args.checkpoint, device)

    report_model(model, tokenizer)

    print("Running perplexity on Wikitext-2 …", end=" ", flush=True)
    start = time.time()
    ppl = perplexity(model, tokenizer, device, samples=min(args.samples, 200))
    print(f"PPL = {ppl:.2f} (time {time.time()-start:.1f}s)")

    if args.summarise:
        print("Summarisation test …", flush=True)
        res = summarisation_metrics(model, tokenizer, device, samples=min(args.samples, 60))
        if res is not None:
            rouge_res, bleu_res = res
            print("  ROUGE-L: {:.3f}".format(rouge_res["rougeL" ]))
            print("  BLEU    : {:.3f}".format(bleu_res["score"]))

    if args.truthfulqa:
        print("TruthfulQA test …", flush=True)
        acc = truthfulqa_eval(model, tokenizer, device, samples=min(args.samples, 100))
        if acc is not None:
            print(f"  Accuracy: {acc*100:.1f}%")

    if args.hellaswag:
        print("HellaSwag test …", flush=True)
        acc = hellaswag_eval(model, tokenizer, device, samples=min(args.samples, 100))
        if acc is not None:
            print(f"  Accuracy: {acc*100:.1f}%")


if __name__ == "__main__":
    main() 