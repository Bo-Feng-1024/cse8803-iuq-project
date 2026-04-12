#!/usr/bin/env python3
"""Elicit LLM prior by querying GPT-4o to simulate dictator game decisions.

Collects M=500 simulated sharing ratios from GPT-4o and saves them for use
as a Bayesian prior in the main experiment.

Usage:
    cd project/
    python scripts/elicit_llm_prior.py
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

M_TARGET = 500  # total pseudo-observations
BATCH_SIZE = 50  # responses per API call
MODEL = "gpt-4o"
SEED = 42
OUTPUT_PATH = PROJECT_ROOT / "data" / "llm_prior_samples.npy"
OUTPUT_JSON = PROJECT_ROOT / "results" / "llm_prior_raw.json"

SYSTEM_PROMPT = (
    "You are a participant in a behavioral economics experiment. "
    "Answer exactly as instructed. Output ONLY the requested numbers, "
    "nothing else."
)

USER_PROMPT_TEMPLATE = """You are participating in a dictator game experiment.

Rules:
- You receive €10.
- You must decide how much to give to an anonymous stranger (between €0.00 and €10.00).
- The stranger has no say in your decision.
- You keep whatever you don't give away.

Please simulate {n} independent participants making this decision.
For each participant, output ONLY the amount given (in euros, to 2 decimal places), one per line.
Output {n} numbers, nothing else."""


def parse_amounts(text, expected_n):
    """Extract numeric amounts from GPT response text."""
    # Find all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    amounts = []
    for num_str in numbers:
        val = float(num_str)
        if 0 <= val <= 10:
            amounts.append(val)
    return amounts[:expected_n]


def query_batch(n_per_batch, batch_idx):
    """Query GPT-4o for a batch of simulated dictator game decisions."""
    prompt = USER_PROMPT_TEMPLATE.format(n=n_per_batch)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,  # maximum diversity
        max_tokens=1000,
        seed=SEED + batch_idx,
    )

    text = response.choices[0].message.content.strip()
    amounts = parse_amounts(text, n_per_batch)
    return amounts, text


def main():
    print(f"Eliciting LLM prior from {MODEL}...")
    print(f"Target: {M_TARGET} pseudo-observations")
    print(f"Batch size: {BATCH_SIZE}")

    all_amounts = []
    raw_responses = []
    n_batches = (M_TARGET + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        remaining = M_TARGET - len(all_amounts)
        n_this = min(BATCH_SIZE, remaining)
        if n_this <= 0:
            break

        print(f"  Batch {i+1}/{n_batches} (requesting {n_this})...", end=" ", flush=True)
        try:
            amounts, raw_text = query_batch(n_this, i)
            all_amounts.extend(amounts)
            raw_responses.append({
                "batch": i,
                "requested": n_this,
                "received": len(amounts),
                "raw_text": raw_text,
            })
            print(f"got {len(amounts)} amounts")
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(5)
            continue

        # Rate limiting
        time.sleep(1)

    # Convert to sharing ratios
    amounts = np.array(all_amounts[:M_TARGET])
    sharing_ratios = amounts / 10.0
    sharing_ratios = np.clip(sharing_ratios, 1e-3, 1 - 1e-3)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, sharing_ratios)

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "model": MODEL,
            "n_collected": len(sharing_ratios),
            "mean_sharing_ratio": float(np.mean(sharing_ratios)),
            "std_sharing_ratio": float(np.std(sharing_ratios)),
            "median": float(np.median(sharing_ratios)),
            "responses": raw_responses,
        }, f, indent=2)

    print(f"\nResults:")
    print(f"  Collected: {len(sharing_ratios)} sharing ratios")
    print(f"  Mean: {np.mean(sharing_ratios):.3f}")
    print(f"  Std:  {np.std(sharing_ratios):.3f}")
    print(f"  Min:  {np.min(sharing_ratios):.3f}")
    print(f"  Max:  {np.max(sharing_ratios):.3f}")
    print(f"  Saved to: {OUTPUT_PATH}")

    # Distribution
    from collections import Counter
    binned = Counter([round(float(r), 1) for r in sharing_ratios])
    print("\n  Distribution (sharing ratio → count):")
    for k in sorted(binned.keys()):
        bar = "#" * binned[k]
        print(f"    {k:.1f}: {binned[k]:3d} {bar}")


if __name__ == "__main__":
    main()
