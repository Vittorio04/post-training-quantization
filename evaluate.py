#!/usr/bin/env python3
"""
LLM evaluation script with bitsandbytes quantization for thesis.
Models: Qwen2.5-7B-Instruct / Qwen2.5-3B-Instruct (or Qwen3-4B)
Formats: FP16 (baseline), INT8 (LLM.int8()), INT4 (NF4)

Usage:
python evaluate_qwen.py --model qwen7b --quant fp16 --output results_7b_fp16.txt
python evaluate_qwen.py --model qwen3b --quant int8 --output results_3b_int8.txt
python evaluate_qwen.py --model qwen7b --quant int4 --output results_7b_int4.txt

Note: set the environment variable BEFORE launching:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"""

import os
import random
import time
import argparse

import numpy as np
import torch
import evaluate
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ── Costanti ────────────────────────────────────────────────────────────────

SEED = 42

MODEL_MAPPING = {
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2b": "google/gemma-2-2b-it",
    "gemma9b": "google/gemma-2-9b-it",
}

DATASET_NAME = "google/wmt24pp"
CONFIG       = "en-it_IT"
SRC_LANG     = "en"
TGT_LANG     = "Italian"
MAX_SAMPLES  = 300          # Same number for ALL tests
BATCH_SIZE   = 1            

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the text from English to Italian. "
    "Output ONLY the final translation. "
    "No explanations, no introduction, no conversational text."
)

# This clears the variable at Python runtime before PyTorch starts
if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
    del os.environ["CUBLAS_WORKSPACE_CONFIG"]


# ── Reproducibility ────────────────────────────────────────────────────────

def set_deterministic(seed: int = SEED):
    """Fix all random generators and force deterministic CUDA kernels"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic kernels (requires CUBLAS_WORKSPACE_CONFIG=:4096:8) 
    # --- THESIS-SAFE MODIFICATION: DISABLED THE BLOCK TO RESTORE SPEED --- 
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check environment variable
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    if ":4096:8" not in cublas:
        print(
            "⚠  CUBLAS_WORKSPACE_CONFIG not set. "
            "Run: export CUBLAS_WORKSPACE_CONFIG=:4096:8  "
            "before launching the script for full reproducibility."
        )


# ── Model Loading ────────────────────────────────────────────────────

def setup_model(model_key: str, quantization_level: str):
    """Load tokenizer and model with the requested quantization."""
    if model_key not in MODEL_MAPPING:
        raise ValueError(
            f"model '{model_key}' not supported. "
            f"Choose from: {', '.join(MODEL_MAPPING.keys())}"
        )

    model_id = MODEL_MAPPING[model_key]
    print(f"\n{'='*60}")
    print(f"Loading: {model_id}  |  Precision: {quantization_level.upper()}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if not torch.cuda.is_available():
        print("⚠  No GPU detected — running on CPU (very slow).")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
        return tokenizer, model, model_id, "cpu"

    device = "cuda:0"

    # ── Quantization Configuration ──
    load_kwargs = dict(device_map={"": 0})

    if quantization_level == "fp16":
        load_kwargs["torch_dtype"] = torch.bfloat16

    elif quantization_level == "int8":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    elif quantization_level == "int4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError("Level not supported. Use: fp16, int8, int4.")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return tokenizer, model, model_id, device


# ── Evaluation ────────────────────────────────────────────────────────────

def run_evaluation(model_key: str, quantization_level: str, output_file: str):
    set_deterministic(SEED)

    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    tokenizer, model, model_id, device = setup_model(model_key, quantization_level)

    # ── VRAM after loading ──
    if device != "cpu":
        torch.cuda.synchronize()
    vram_model_load = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device != "cpu" else 0.0
    )
    print(f"Model idle VRAM: {vram_model_load:.2f} GB")

    # ── Loading Dataset ──
    print(f"Loading dataset {DATASET_NAME} ({CONFIG})...")
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    ds = load_dataset(DATASET_NAME, CONFIG, split="train", streaming=True)

    sorgenti = []
    riferimenti = []
    for i, ex in enumerate(ds):
        if i >= MAX_SAMPLES:
            break
        sorgenti.append(ex["source"])
        riferimenti.append(ex["target"])
    print(f"Loaded {len(sorgenti)} source-target pairs.")

    # ── Inference ──
    print(f"\nStarting translation ({len(sorgenti)} sentences, batch_size={BATCH_SIZE})...")
    predizioni = []
    total_generated_tokens = 0  # only real tokens, no padding

    # Synchronize and start timer
    if device != "cpu":
        torch.cuda.synchronize()
    start_time = time.time()

    for i in tqdm(range(0, len(sorgenti), BATCH_SIZE)):
        batch = sorgenti[i : i + BATCH_SIZE]

        # Build model-specific structured prompt
        prompts = []
        for s in batch:
            if "gemma" in model_key:
                # For Gemma: embed the system prompt in the user message
                messages = [
                    {
                        "role": "user", 
                        "content": f"{SYSTEM_PROMPT}\n\nText to translate: {s}"
                    },
                ]
            else:
                # For Qwen: maintain the original structure
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Text to translate: {s}"},
                ]
            
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(device)

        bad_words = [tokenizer.encode("<think>", add_special_tokens=False)]

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,    # deterministic greedy decoding
                num_beams=1,
                bad_words_ids=bad_words 
            )

        # Count only new tokens (excluding prompt) and only non-pad tokens
        prompt_len = inputs.input_ids.shape[1]
        new_tokens = generated[:, prompt_len:]

        for row in new_tokens:
            # Count real tokens (different from pad/eos) for this sentence
            real = (row != tokenizer.pad_token_id).sum().item()
            total_generated_tokens += real

        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for d in decoded:
            # Clean the string of any residual tags before saving
            testo_pulito = d.replace("</think>", "").replace("<think>", "").strip()
            predizioni.append(testo_pulito)

    # Synchronize and stop timer
    if device != "cpu":
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    throughput = total_generated_tokens / total_time if total_time > 0 else 0.0

    # ── VRAM Peak ──
    vram_peak = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device != "cpu" else 0.0
    )

    # ── Linguistic metrics ──
    print("\nCalculating linguistic metrics...")
    bleu_metric   = evaluate.load("sacrebleu")
    rouge_metric  = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    bleu_result   = bleu_metric.compute(
        predictions=predizioni,
        references=[[r] for r in riferimenti],
    )
    rouge_result  = rouge_metric.compute(
        predictions=predizioni,
        references=riferimenti,
    )
    meteor_result = meteor_metric.compute(
        predictions=predizioni,
        references=riferimenti,
    )

    # ── Report writing ──
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"THESIS REPORT - Model: {model_id}\n")
        f.write(f"Quantization level: {quantization_level.upper()}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Samples: {len(sorgenti)}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write("-" * 50 + "\n")

        f.write("\n=== LINGUISTIC METRICS ===\n")
        f.write(f"SacreBLEU:  {bleu_result['score']:.2f}\n")
        f.write(f"METEOR:     {meteor_result['meteor']:.4f}\n")
        f.write(f"ROUGE-L:    {rouge_result['rougeL']:.4f}\n")

        f.write("\n=== HARDWARE PROFILING ===\n")
        f.write(f"Idle model VRAM:      {vram_model_load:.2f} GB\n")
        f.write(f"VRAM Peak (Inference): {vram_peak:.2f} GB\n")
        f.write(f"Total Execution Time:    {total_time:.2f} seconds\n")
        f.write(f"Generated Tokens (real):     {total_generated_tokens}\n")
        f.write(f"Throughput:                 {throughput:.2f} Tokens/second\n")

        f.write("\n=== CONFIGURATION ===\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}\n")
        f.write(f"CUDA: {torch.version.cuda if device != 'cpu' else 'N/A'}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"Decoding: greedy (do_sample=False, num_beams=1)\n")
        f.write(f"Max new tokens: 256\n")

        f.write("\n=== TRANSLATION EXAMPLES (first 20) ===\n")
        for j, (s, p, r) in enumerate(zip(sorgenti, predizioni, riferimenti)):
            if j >= 20:
                break
            f.write(f"\n[{j+1}]\n")
            f.write(f"  SRC: {s}\n")
            f.write(f"  HYP: {p}\n")
            f.write(f"  REF: {r}\n")

    print(f"\n✅ Results saved to: {output_file}")
    print(f"   BLEU={bleu_result['score']:.2f}  "
          f"METEOR={meteor_result['meteor']:.4f}  "
          f"ROUGE-L={rouge_result['rougeL']:.4f}")
    print(f"   VRAM={vram_peak:.2f}GB  "
          f"Throughput={throughput:.2f} tok/s  "
          f"Time={total_time:.1f}s")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM evaluation with bitsandbytes quantization (thesis)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_MAPPING.keys()),
        help="Key model (e.g., qwen7b, qwen3b)"
    )
    parser.add_argument(
        "--quant", type=str, required=True,
        choices=["fp16", "int8", "int4"],
        help="Quantization level"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output file path (.txt)"
    )
    args = parser.parse_args()

    run_evaluation(args.model, args.quant, args.output)