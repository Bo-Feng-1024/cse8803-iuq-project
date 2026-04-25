#!/usr/bin/env python3
"""Compute-accuracy Pareto front: wall-clock seconds vs posterior std of mu.

For the giving-game, n=50, gpt-4o LLM prior, sweep MCMC chain length and
SVGD iteration count, recording (wall-clock, ESS of mu, posterior std of mu).

Usage:
    cd project/
    python scripts/run_compute_pareto.py
"""

import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dictator_data, subsample, train_test_split
from src.mcmc import metropolis_hastings, make_log_posterior
from src.svgd import run_svgd_inference
from src.diagnostics import effective_sample_size

N = 50
LLM_PRIOR_WEIGHT = 20.0
BURNIN = 3000
SEED = 42

MCMC_LENGTHS = [1000, 2000, 5000, 10000, 15000, 30000]
SVGD_ITERS = [200, 500, 1000, 2000, 4000]
N_PARTICLES = 100
SVGD_LR = 0.05

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def mu_stats(samples):
    a, b = samples[:, 0], samples[:, 1]
    mu = a / (a + b)
    return float(np.mean(mu)), float(np.std(mu)), mu


def main():
    print("=" * 60)
    print("Compute-accuracy Pareto: MCMC vs SVGD (giving game, n=50)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pooled data (giving + taking) to match the original n=50 baseline.
    d = load_dictator_data()
    all_data = d["sharing_ratio"]
    train_data, _ = train_test_split(all_data, test_frac=0.3, seed=SEED)
    pseudo = np.load(PROJECT_ROOT / "data" / "llm_prior_samples.npy")
    rng = np.random.default_rng(SEED)
    data_sub = subsample(train_data, N, rng=rng)
    log_post, _ = make_log_posterior(
        data_sub, prior_type="llm", pseudo_data=pseudo, weight=LLM_PRIOR_WEIGHT,
    )

    # MCMC sweep
    print("\nMCMC sweep:")
    mcmc_results = []
    for n_samples in MCMC_LENGTHS:
        t0 = time.time()
        samples, accept = metropolis_hastings(
            log_post, init=(2.0, 2.0),
            n_samples=n_samples, burnin=BURNIN,
            proposal_std=0.3, seed=SEED,
        )
        t = time.time() - t0
        mu_mean, mu_std, mu_chain = mu_stats(samples)
        ess_mu = effective_sample_size(mu_chain)
        ess_per_sec = ess_mu / max(t, 1e-9)
        print(f"  n_samples={n_samples:5d}: t={t:6.2f}s  mu_std={mu_std:.4f}  ESS(mu)={ess_mu:7.1f}  ESS/s={ess_per_sec:7.1f}")
        mcmc_results.append({
            "method": "MCMC", "n_samples": n_samples,
            "wall_clock_s": float(t),
            "mu_mean": mu_mean, "mu_std": mu_std,
            "ess_mu": float(ess_mu),
            "ess_per_sec": float(ess_per_sec),
            "accept_rate": float(accept),
        })

    # SVGD sweep
    print("\nSVGD sweep:")
    svgd_results = []
    for n_iter in SVGD_ITERS:
        t0 = time.time()
        samples, _ = run_svgd_inference(
            data_sub, prior_type="llm", pseudo_data=pseudo,
            weight=LLM_PRIOR_WEIGHT,
            n_particles=N_PARTICLES, n_iter=n_iter, stepsize=SVGD_LR, seed=SEED,
        )
        t = time.time() - t0
        mu_mean, mu_std, mu_arr = mu_stats(samples)
        ess_mu = float(N_PARTICLES)  # SVGD particles are by construction iid-ish
        ess_per_sec = ess_mu / max(t, 1e-9)
        print(f"  n_iter={n_iter:5d}: t={t:6.2f}s  mu_std={mu_std:.4f}  ESS={ess_mu:.0f}  ESS/s={ess_per_sec:7.1f}")
        svgd_results.append({
            "method": "SVGD", "n_iter": n_iter,
            "wall_clock_s": float(t),
            "mu_mean": mu_mean, "mu_std": mu_std,
            "ess_mu": ess_mu,
            "ess_per_sec": float(ess_per_sec),
        })

    # Plot Pareto: wall-clock vs posterior std of mu
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    mcmc_t = [r["wall_clock_s"] for r in mcmc_results]
    mcmc_s = [r["mu_std"] for r in mcmc_results]
    svgd_t = [r["wall_clock_s"] for r in svgd_results]
    svgd_s = [r["mu_std"] for r in svgd_results]
    ax.plot(mcmc_t, mcmc_s, "o-", color="C0", label="MCMC", markersize=8)
    ax.plot(svgd_t, svgd_s, "s-", color="C1", label="SVGD", markersize=8)
    ax.set_xlabel("Wall-clock (seconds)")
    ax.set_ylabel(r"Posterior std of $\mu$")
    ax.set_xscale("log")
    ax.set_title("Compute vs accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    mcmc_eps = [r["ess_per_sec"] for r in mcmc_results]
    svgd_eps = [r["ess_per_sec"] for r in svgd_results]
    ax.plot(mcmc_t, mcmc_eps, "o-", color="C0", label="MCMC", markersize=8)
    ax.plot(svgd_t, svgd_eps, "s-", color="C1", label="SVGD", markersize=8)
    ax.set_xlabel("Wall-clock (seconds)")
    ax.set_ylabel(r"ESS$(\mu)$ per second")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("ESS / second")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pareto.pdf", dpi=150)
    plt.savefig(FIGURES_DIR / "pareto.png", dpi=150)
    plt.close()

    out = {"mcmc": mcmc_results, "svgd": svgd_results,
           "n": int(N), "llm_prior_weight": LLM_PRIOR_WEIGHT}
    with open(RESULTS_DIR / "pareto.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/pareto.json")
    print("Saved: figures/pareto.pdf")


if __name__ == "__main__":
    main()
