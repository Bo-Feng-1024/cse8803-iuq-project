#!/usr/bin/env python3
"""Elicit LLM prior from gpt-3.5-turbo (Sep-2021 cutoff, pre-Horton2023).

Identical protocol/prompt to elicit_llm_prior.py but with a model whose
training cutoff predates the Horton[2023] 'Large Language Models as
Simulated Economic Agents' paper. Used to ablate possible training-data
leakage in the gpt-4o LLM prior.

Usage:
    cd project/
    python scripts/elicit_llm_prior_gpt35.py
"""

import os
import json
import re
import time
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API"))

M_TARGET = 500
BATCH_SIZE = 50
MODEL = "gpt-3.5-turbo"
SEED = 42
OUTPUT_PATH = PROJECT_ROOT / "data" / "llm_prior_samples_gpt35.npy"
OUTPUT_JSON = PROJECT_ROOT / "results" / "llm_prior_raw_gpt35.json"

SYSTEM_PROMPT = (
    "You are a participant in a behavioral economics experiment. "
    "Answer exactly as instructed. Output ONLY the requested numbers, "
    "nothing else."
)

USER_PROMPT_TEMPLATE = """You are participating in a dictator game experiment.

Rules:
- You receive EUR 10.
- You must decide how much to give to an anonymous stranger (between EUR 0.00 and EUR 10.00).
- The stranger has no say in your decision.
- You keep whatever you don't give away.

Please simulate {n} independent participants making this decision.
For each participant, output ONLY the amount given (in euros, to 2 decimal places), one per line.
Output {n} numbers, nothing else."""


def parse_amounts(text, expected_n):
    numbers = re.findall(r"\d+\.?\d*", text)
    out = []
    for s in numbers:
        v = float(s)
        if 0 <= v <= 10:
            out.append(v)
    return out[:expected_n]


def query_batch(n_per_batch, batch_idx):
    prompt = USER_PROMPT_TEMPLATE.format(n=n_per_batch)
    # gpt-3.5-turbo does not support `seed` reliably; try, fall back without.
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=1000,
            seed=SEED + batch_idx,
        )
    except Exception:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=1000,
        )
    text = response.choices[0].message.content.strip()
    return parse_amounts(text, n_per_batch), text


def main():
    print(f"Eliciting LLM prior from {MODEL} (cutoff Sep-2021)...")
    print(f"Target: {M_TARGET} pseudo-observations")

    all_amounts = []
    raw = []
    n_batches = (M_TARGET + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        remaining = M_TARGET - len(all_amounts)
        n_this = min(BATCH_SIZE, remaining)
        if n_this <= 0:
            break
        print(f"  Batch {i+1}/{n_batches} (n={n_this})...", end=" ", flush=True)
        try:
            amounts, raw_text = query_batch(n_this, i)
            all_amounts.extend(amounts)
            raw.append({
                "batch": i, "requested": n_this,
                "received": len(amounts), "raw_text": raw_text,
            })
            print(f"got {len(amounts)}")
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(5)
            continue
        time.sleep(1)

    amounts = np.array(all_amounts[:M_TARGET])
    sharing_ratios = amounts / 10.0
    sharing_ratios = np.clip(sharing_ratios, 1e-3, 1 - 1e-3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, sharing_ratios)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "model": MODEL,
            "game_type": "giving",
            "n_collected": len(sharing_ratios),
            "mean_sharing_ratio": float(np.mean(sharing_ratios)),
            "std_sharing_ratio": float(np.std(sharing_ratios)),
            "median": float(np.median(sharing_ratios)),
            "responses": raw,
        }, f, indent=2)

    print(f"\nResults ({MODEL}):")
    print(f"  Collected: {len(sharing_ratios)}")
    print(f"  Mean: {np.mean(sharing_ratios):.3f}")
    print(f"  Std:  {np.std(sharing_ratios):.3f}")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
