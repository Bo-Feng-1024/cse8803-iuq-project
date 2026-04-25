#!/usr/bin/env python3
"""Dense misspecification grid: shift gpt-4o LLM prior by delta in [-3, +3] std
in 0.25 steps and measure 95% posterior-predictive coverage on the giving-game
held-out test set. Replaces the 5-point table with a continuous curve.

Usage:
    cd project/
    python scripts/run_failure_mode_dense.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dictator_data, subsample, train_test_split
from src.mcmc import metropolis_hastings, make_log_posterior
from src.evaluation import empirical_coverage

DELTA_GRID = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.25), 4)
N = 50
N_MCMC = 15000
BURNIN = 3000
LLM_PRIOR_WEIGHT = 20.0
SEED = 42

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def main():
    print("=" * 60)
    print("Dense misspecification grid (giving game, n=50, gpt-4o prior)")
    print(f"  Grid: delta in [{DELTA_GRID[0]}, {DELTA_GRID[-1]}] step 0.25 ({len(DELTA_GRID)} points)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pooled data (giving + taking) to match the original n=50 baseline.
    d = load_dictator_data()
    all_data = d["sharing_ratio"]
    train_data, test_data = train_test_split(all_data, test_frac=0.3, seed=SEED)
    pseudo = np.load(PROJECT_ROOT / "data" / "llm_prior_samples.npy")
    pseudo_std = float(np.std(pseudo))

    rng = np.random.default_rng(SEED)
    data_sub = subsample(train_data, N, rng=rng)

    coverages = []
    pseudo_means = []
    for delta in DELTA_GRID:
        shifted = np.clip(pseudo + delta * pseudo_std, 1e-3, 1 - 1e-3)
        log_post, _ = make_log_posterior(
            data_sub, prior_type="llm", pseudo_data=shifted, weight=LLM_PRIOR_WEIGHT,
        )
        samples, _ = metropolis_hastings(
            log_post, init=(2.0, 2.0), n_samples=N_MCMC, burnin=BURNIN,
            proposal_std=0.3, seed=SEED,
        )
        cov = empirical_coverage(samples, test_data)
        coverages.append(cov)
        pseudo_means.append(float(np.mean(shifted)))
        print(f"  delta={delta:+.2f}  pseudo_mean={np.mean(shifted):.3f}  coverage={cov:.3f}")

    coverages = np.array(coverages)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(DELTA_GRID, coverages, "o-", color="C3", markersize=4, lw=1.5)
    ax.axhline(0.95, color="gray", ls="--", alpha=0.7, label="Nominal 95%")
    ax.axvline(0.0, color="black", ls=":", alpha=0.5, label=r"Original GPT-4o prior")
    ax.set_xlabel(r"Prior mean shift $\delta$ (in prior std-devs)")
    ax.set_ylabel("Empirical coverage of 95% PI")
    ax.set_title("Coverage under prior misspecification")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_mode_dense.pdf", dpi=150)
    plt.savefig(FIGURES_DIR / "failure_mode_dense.png", dpi=150)
    plt.close()

    out = {
        "n": int(N),
        "delta_grid": [float(x) for x in DELTA_GRID],
        "pseudo_means": pseudo_means,
        "coverages": [float(c) for c in coverages],
        "llm_prior_weight": LLM_PRIOR_WEIGHT,
        "pseudo_std": pseudo_std,
        "model": "gpt-4o",
    }
    with open(RESULTS_DIR / "failure_mode_dense.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/failure_mode_dense.json")
    print(f"Saved: figures/failure_mode_dense.pdf")


if __name__ == "__main__":
    main()
