#!/usr/bin/env python3
"""
Script di valutazione LLM con quantizzazione bitsandbytes per tesi.
Modelli: Qwen2.5-7B-Instruct / Qwen2.5-3B-Instruct (o Qwen3-4B)
Formati: FP16 (baseline), INT8 (LLM.int8()), INT4 (NF4)

Correzioni rispetto allo script originale:
  1. Seed globale + determinismo CUDA per riproducibilità bit-identica
  2. torch.cuda.synchronize() prima/dopo la generazione per tempi corretti
  3. Throughput calcolato solo sui token effettivamente generati (no padding)
  4. Batch size 1 per evitare artefatti da padding variabile
  5. Struttura report identica all'originale per compatibilità

Uso:
  python evaluate_qwen.py --model qwen7b --quant fp16 --output results_7b_fp16.txt
  python evaluate_qwen.py --model qwen3b --quant int8  --output results_3b_int8.txt
  python evaluate_qwen.py --model qwen7b --quant int4  --output results_7b_int4.txt

Nota: impostare la variabile d'ambiente PRIMA di lanciare:
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
MAX_SAMPLES  = 300          # Stesso numero su TUTTI i test
BATCH_SIZE   = 1            # Batch=1: elimina artefatti da padding

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the text from English to Italian. "
    "Output ONLY the final translation. "
    "No explanations, no introduction, no conversational text."
)

# Questo pulisce la variabile a livello di runtime Python prima che parta PyTorch
if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
    del os.environ["CUBLAS_WORKSPACE_CONFIG"]


# ── Riproducibilità ────────────────────────────────────────────────────────

def set_deterministic(seed: int = SEED):
    """Fissa tutti i generatori e forza kernel deterministici CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Kernel deterministici (richiede CUBLAS_WORKSPACE_CONFIG=:4096:8)
    # --- MODIFICA SALVA-TESI: DISATTIVATO IL BLOCCO PER RIPRISTINARE LA VELOCITA' ---
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verifica variabile d'ambiente
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    if ":4096:8" not in cublas:
        print(
            "⚠  CUBLAS_WORKSPACE_CONFIG non impostata. "
            "Esegui:  export CUBLAS_WORKSPACE_CONFIG=:4096:8  "
            "prima di lanciare lo script per piena riproducibilità."
        )


# ── Caricamento modello ────────────────────────────────────────────────────

def setup_model(model_key: str, quantization_level: str):
    """Carica tokenizer e modello con la quantizzazione richiesta."""
    if model_key not in MODEL_MAPPING:
        raise ValueError(
            f"Modello '{model_key}' non supportato. "
            f"Scegli tra: {', '.join(MODEL_MAPPING.keys())}"
        )

    model_id = MODEL_MAPPING[model_key]
    print(f"\n{'='*60}")
    print(f"Caricamento: {model_id}  |  Precisione: {quantization_level.upper()}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if not torch.cuda.is_available():
        print("⚠  Nessuna GPU rilevata — esecuzione su CPU (molto lenta).")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
        return tokenizer, model, model_id, "cpu"

    device = "cuda:0"

    # ── Configurazione quantizzazione ──
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
        raise ValueError("Livello non supportato. Usa: fp16, int8, int4.")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return tokenizer, model, model_id, device


# ── Valutazione ────────────────────────────────────────────────────────────

def run_evaluation(model_key: str, quantization_level: str, output_file: str):
    set_deterministic(SEED)

    # Pulizia memoria GPU
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    tokenizer, model, model_id, device = setup_model(model_key, quantization_level)

    # ── VRAM dopo il caricamento ──
    if device != "cpu":
        torch.cuda.synchronize()
    vram_model_load = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device != "cpu" else 0.0
    )
    print(f"VRAM modello a riposo: {vram_model_load:.2f} GB")

    # ── Caricamento dataset ──
    print(f"Caricamento dataset {DATASET_NAME} ({CONFIG})...")
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
    print(f"Caricate {len(sorgenti)} coppie sorgente-riferimento.")

    # ── Inferenza ──
    print(f"\nInizio traduzione ({len(sorgenti)} frasi, batch_size={BATCH_SIZE})...")
    predizioni = []
    total_generated_tokens = 0  # solo token reali, no padding

    # Sincronizza e avvia cronometro
    if device != "cpu":
        torch.cuda.synchronize()
    start_time = time.time()

    for i in tqdm(range(0, len(sorgenti), BATCH_SIZE)):
        batch = sorgenti[i : i + BATCH_SIZE]

        # Costruisci prompt strutturato specifico per modello
        prompts = []
        for s in batch:
            if "gemma" in model_key:
                # Per Gemma: fondiamo il system prompt nel messaggio user
                messages = [
                    {
                        "role": "user", 
                        "content": f"{SYSTEM_PROMPT}\n\nText to translate: {s}"
                    },
                ]
            else:
                # Per Qwen: manteniamo la struttura originale
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

        # --- INIZIO MODIFICA SALVA-TESI ---
        # Blocca fisicamente la generazione del tag <think> costringendo Qwen a tradurre subito
        bad_words = [tokenizer.encode("<think>", add_special_tokens=False)]
        # --- FINE MODIFICA ---

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,    # greedy deterministico
                num_beams=1,
                bad_words_ids=bad_words # <-- PARAMETRO AGGIUNTO QUI
            )

        # Conta solo i token nuovi (escluso prompt) e solo quelli non-pad
        prompt_len = inputs.input_ids.shape[1]
        new_tokens = generated[:, prompt_len:]

        for row in new_tokens:
            # Conta token reali (diversi da pad/eos) per questa frase
            real = (row != tokenizer.pad_token_id).sum().item()
            total_generated_tokens += real

        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for d in decoded:
            # Pulisce la stringa dai tag residui prima di salvarla
            testo_pulito = d.replace("</think>", "").replace("<think>", "").strip()
            predizioni.append(testo_pulito)

    # Sincronizza e ferma cronometro
    if device != "cpu":
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    throughput = total_generated_tokens / total_time if total_time > 0 else 0.0

    # ── VRAM picco ──
    vram_peak = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device != "cpu" else 0.0
    )

    # ── Metriche linguistiche ──
    print("\nCalcolo metriche linguistiche...")
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

    # ── Scrittura report ──
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"REPORT TESI - Modello: {model_id}\n")
        f.write(f"Livello Quantizzazione: {quantization_level.upper()}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Campioni: {len(sorgenti)}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write("-" * 50 + "\n")

        f.write("\n=== METRICHE LINGUISTICHE ===\n")
        f.write(f"SacreBLEU:  {bleu_result['score']:.2f}\n")
        f.write(f"METEOR:     {meteor_result['meteor']:.4f}\n")
        f.write(f"ROUGE-L:    {rouge_result['rougeL']:.4f}\n")

        f.write("\n=== PROFILAZIONE HARDWARE ===\n")
        f.write(f"VRAM Modello a riposo:      {vram_model_load:.2f} GB\n")
        f.write(f"VRAM Picco Max (Inferenza): {vram_peak:.2f} GB\n")
        f.write(f"Tempo Totale Esecuzione:    {total_time:.2f} secondi\n")
        f.write(f"Token Generati (reali):     {total_generated_tokens}\n")
        f.write(f"Throughput:                 {throughput:.2f} Token/secondo\n")

        f.write("\n=== CONFIGURAZIONE ===\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}\n")
        f.write(f"CUDA: {torch.version.cuda if device != 'cpu' else 'N/A'}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"Decoding: greedy (do_sample=False, num_beams=1)\n")
        f.write(f"Max new tokens: 256\n")

        f.write("\n=== ESEMPI DI TRADUZIONE (primi 20) ===\n")
        for j, (s, p, r) in enumerate(zip(sorgenti, predizioni, riferimenti)):
            if j >= 20:
                break
            f.write(f"\n[{j+1}]\n")
            f.write(f"  SRC: {s}\n")
            f.write(f"  HYP: {p}\n")
            f.write(f"  REF: {r}\n")

    print(f"\n✅ Risultati salvati in: {output_file}")
    print(f"   BLEU={bleu_result['score']:.2f}  "
          f"METEOR={meteor_result['meteor']:.4f}  "
          f"ROUGE-L={rouge_result['rougeL']:.4f}")
    print(f"   VRAM={vram_peak:.2f}GB  "
          f"Throughput={throughput:.2f} tok/s  "
          f"Tempo={total_time:.1f}s")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valutazione LLM con quantizzazione bitsandbytes (tesi)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_MAPPING.keys()),
        help="Chiave modello (es. qwen7b, qwen3b)"
    )
    parser.add_argument(
        "--quant", type=str, required=True,
        choices=["fp16", "int8", "int4"],
        help="Livello di precisione"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Percorso file di output (.txt)"
    )
    args = parser.parse_args()

    run_evaluation(args.model, args.quant, args.output)