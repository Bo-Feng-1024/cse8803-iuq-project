#!/usr/bin/env python3
"""Run the full pipeline (MCMC + SVGD, flat vs LLM prior) on the TAKING-game
variant and report rho_taking. Mirrors scripts/run_experiment.py but loads
the taking subset of the data and the taking-game LLM prior.

Usage:
    cd project/
    python scripts/run_experiment_taking.py
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
from src.mcmc import metropolis_hastings, make_log_posterior, posterior_summary
from src.svgd import run_svgd_inference
from src.evaluation import (
    empirical_coverage, posterior_mu_variance, find_nstar, compute_reduction_ratio,
)

# Sample sizes are slightly smaller than giving (only 68 obs total -> ~47 train).
SAMPLE_SIZES = [5, 10, 20, 35, 45]
N_MCMC = 15000
BURNIN = 3000
N_PARTICLES = 100
SVGD_ITER = 2000
SVGD_LR = 0.05
LLM_PRIOR_WEIGHT = 20.0
SEED = 42
GAME_TYPE = "taking"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def run_pair(data_full, n, prior_type, pseudo_data=None, weight=1.0, seed=42):
    rng = np.random.default_rng(seed)
    data_sub = subsample(data_full, n, rng=rng)

    log_post, _ = make_log_posterior(
        data_sub, prior_type=prior_type, pseudo_data=pseudo_data, weight=weight,
    )
    t0 = time.time()
    mcmc_samples, accept = metropolis_hastings(
        log_post, init=(2.0, 2.0), n_samples=N_MCMC, burnin=BURNIN,
        proposal_std=0.3, seed=seed,
    )
    mcmc_t = time.time() - t0
    summ = posterior_summary(mcmc_samples)

    t0 = time.time()
    svgd_samples, _ = run_svgd_inference(
        data_sub, prior_type=prior_type, pseudo_data=pseudo_data, weight=weight,
        n_particles=N_PARTICLES, n_iter=SVGD_ITER, stepsize=SVGD_LR, seed=seed,
    )
    svgd_t = time.time() - t0
    return {
        "n": n, "prior_type": prior_type,
        "mcmc": {"samples": mcmc_samples, "summary": summ,
                 "accept_rate": accept, "time": mcmc_t},
        "svgd": {"samples": svgd_samples,
                 "summary": posterior_summary(svgd_samples), "time": svgd_t},
    }


def main():
    print("=" * 60)
    print(f"TAKING-game pipeline: dictator game (variant = {GAME_TYPE})")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading taking-game data...")
    d = load_dictator_data(game_type=GAME_TYPE)
    all_data = d["sharing_ratio"]
    train_data, test_data = train_test_split(all_data, test_frac=0.3, seed=SEED)
    print(f"  Total: {len(all_data)}, Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"  Mean sharing_ratio: {np.mean(all_data):.3f}, Std: {np.std(all_data):.3f}")

    print("\n[2/4] Loading taking-game LLM prior...")
    pseudo_path = PROJECT_ROOT / "data" / "llm_prior_samples_taking.npy"
    if not pseudo_path.exists():
        raise SystemExit(f"Run elicit_llm_prior_taking.py first; missing {pseudo_path}")
    pseudo_data = np.load(pseudo_path)
    print(f"  Pseudo-obs: {len(pseudo_data)}, mean={np.mean(pseudo_data):.3f}, "
          f"std={np.std(pseudo_data):.3f}")

    print("\n[3/4] Running flat vs LLM prior at each n...")
    results_flat, results_llm = [], []
    vars_flat, vars_llm = [], []
    cov_flat_n, cov_llm_n = {}, {}
    for n in SAMPLE_SIZES:
        if n > len(train_data):
            print(f"  skip n={n} (only {len(train_data)} train obs)")
            continue
        print(f"  n={n}: flat...", end=" ", flush=True)
        rf = run_pair(train_data, n, "flat", seed=SEED)
        vars_flat.append(rf["mcmc"]["summary"]["mu_var"])
        results_flat.append(rf)
        print(f"mu={rf['mcmc']['summary']['mu_mean']:.3f} | LLM...", end=" ", flush=True)
        rl = run_pair(train_data, n, "llm",
                      pseudo_data=pseudo_data, weight=LLM_PRIOR_WEIGHT, seed=SEED)
        vars_llm.append(rl["mcmc"]["summary"]["mu_var"])
        results_llm.append(rl)
        print(f"mu={rl['mcmc']['summary']['mu_mean']:.3f}")

        # Coverage at every n
        cov_flat_n[n] = empirical_coverage(rf["mcmc"]["samples"], test_data)
        cov_llm_n[n] = empirical_coverage(rl["mcmc"]["samples"], test_data)

    sizes_used = [r["n"] for r in results_flat]
    target_var = (vars_flat[0] + vars_flat[-1]) / 2
    nstar_flat = find_nstar(sizes_used, vars_flat, target_var)
    nstar_llm = find_nstar(sizes_used, vars_llm, target_var)
    rho = compute_reduction_ratio(nstar_flat, nstar_llm)

    print("\n[4/4] Plotting and saving...")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes_used, vars_flat, "o-", color="C0", label="Flat prior (MCMC)")
    ax.plot(sizes_used, vars_llm, "s-", color="C1", label="LLM prior (MCMC)")
    ax.axhline(target_var, color="gray", ls="--", alpha=0.7,
               label=f"Target var = {target_var:.4g}")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(r"Posterior variance of $\mu$")
    ax.set_title(f"Taking game: posterior precision vs $n$ (rho={rho:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nstar_curve_taking.pdf", dpi=150)
    plt.savefig(FIGURES_DIR / "nstar_curve_taking.png", dpi=150)
    plt.close()

    summary = {
        "game_type": GAME_TYPE,
        "sample_sizes": sizes_used,
        "vars_flat_mcmc": [float(v) for v in vars_flat],
        "vars_llm_mcmc": [float(v) for v in vars_llm],
        "nstar_flat": float(nstar_flat),
        "nstar_llm": float(nstar_llm),
        "rho": float(rho),
        "target_var": float(target_var),
        "coverage_flat_per_n": {str(k): float(v) for k, v in cov_flat_n.items()},
        "coverage_llm_per_n": {str(k): float(v) for k, v in cov_llm_n.items()},
        "llm_prior_weight": LLM_PRIOR_WEIGHT,
        "llm_prior_mean": float(np.mean(pseudo_data)),
        "llm_prior_std": float(np.std(pseudo_data)),
        "data_mean": float(np.mean(all_data)),
        "data_std": float(np.std(all_data)),
        "n_total": len(all_data),
        "n_train": len(train_data),
        "n_test": len(test_data),
    }
    with open(RESULTS_DIR / "experiment_results_taking.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"  rho_taking = {rho:.3f}")
    print(f"  n*_flat = {nstar_flat:.1f}, n*_LLM = {nstar_llm:.1f}")
    print(f"  saved: results/experiment_results_taking.json")
    print(f"  saved: figures/nstar_curve_taking.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
