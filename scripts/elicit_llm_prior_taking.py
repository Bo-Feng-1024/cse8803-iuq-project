#!/usr/bin/env python3
"""Elicit LLM prior for the TAKING-game variant by querying GPT-4o.

Same protocol as elicit_llm_prior.py but with the taking-game framing
(von Blanckenburg et al. 2023): the dictator starts with nothing and may
take up to EUR 10 from a counterpart. The reported quantity is the
counterpart-kept share, sharing_ratio = (10 - amount_taken) / 10, so it
remains in [0, 1] and aligns with the giving-game convention.

Usage:
    cd project/
    python scripts/elicit_llm_prior_taking.py
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
MODEL = "gpt-4o"
SEED = 42
OUTPUT_PATH = PROJECT_ROOT / "data" / "llm_prior_samples_taking.npy"
OUTPUT_JSON = PROJECT_ROOT / "results" / "llm_prior_raw_taking.json"

SYSTEM_PROMPT = (
    "You are a participant in a behavioral economics experiment. "
    "Answer exactly as instructed. Output ONLY the requested numbers, "
    "nothing else."
)

USER_PROMPT_TEMPLATE = """You are participating in a TAKING-game variant of the dictator game.

Rules:
- An anonymous stranger has been given EUR 10. You start with EUR 0.
- You may take any amount from EUR 0.00 up to EUR 10.00 from the stranger.
- The stranger has no say in your decision.
- Whatever you do not take, the stranger keeps.

Please simulate {n} independent participants making this decision.
For each participant, output ONLY the amount TAKEN (in euros, to 2 decimal places), one per line.
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
    text = response.choices[0].message.content.strip()
    return parse_amounts(text, n_per_batch), text


def main():
    print(f"Eliciting TAKING-game LLM prior from {MODEL}...")
    print(f"Target: {M_TARGET} pseudo-observations")

    all_taken = []
    raw = []
    n_batches = (M_TARGET + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        remaining = M_TARGET - len(all_taken)
        n_this = min(BATCH_SIZE, remaining)
        if n_this <= 0:
            break
        print(f"  Batch {i+1}/{n_batches} (n={n_this})...", end=" ", flush=True)
        try:
            amounts, raw_text = query_batch(n_this, i)
            all_taken.extend(amounts)
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

    taken = np.array(all_taken[:M_TARGET])
    # Convert "amount taken" -> "counterpart sharing ratio" = (10 - taken)/10
    sharing_ratios = (10.0 - taken) / 10.0
    sharing_ratios = np.clip(sharing_ratios, 1e-3, 1 - 1e-3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, sharing_ratios)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "model": MODEL,
            "game_type": "taking",
            "n_collected": len(sharing_ratios),
            "mean_sharing_ratio": float(np.mean(sharing_ratios)),
            "std_sharing_ratio": float(np.std(sharing_ratios)),
            "median": float(np.median(sharing_ratios)),
            "amount_taken_mean": float(np.mean(taken)),
            "amount_taken_std": float(np.std(taken)),
            "responses": raw,
        }, f, indent=2)

    print("\nResults (taking-game):")
    print(f"  Collected: {len(sharing_ratios)}")
    print(f"  Mean sharing_ratio (counterpart-kept): {np.mean(sharing_ratios):.3f}")
    print(f"  Std:  {np.std(sharing_ratios):.3f}")
    print(f"  Mean amount TAKEN: EUR {np.mean(taken):.2f}")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
