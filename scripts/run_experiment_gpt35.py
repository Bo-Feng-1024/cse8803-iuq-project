#!/usr/bin/env python3
"""Re-run the giving-game pipeline with gpt-3.5-turbo's LLM prior to ablate
training-data leakage from Horton[2023] in gpt-4o.

Usage:
    cd project/
    python scripts/run_experiment_gpt35.py
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
    empirical_coverage, find_nstar, compute_reduction_ratio,
)

SAMPLE_SIZES = [5, 10, 20, 50, 90]
N_MCMC = 15000
BURNIN = 3000
N_PARTICLES = 100
SVGD_ITER = 2000
SVGD_LR = 0.05
LLM_PRIOR_WEIGHT = 20.0
SEED = 42

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def run_pair(data_full, n, prior_type, pseudo_data=None, weight=1.0, seed=42):
    rng = np.random.default_rng(seed)
    data_sub = subsample(data_full, n, rng=rng)
    log_post, _ = make_log_posterior(
        data_sub, prior_type=prior_type, pseudo_data=pseudo_data, weight=weight,
    )
    mcmc_samples, accept = metropolis_hastings(
        log_post, init=(2.0, 2.0), n_samples=N_MCMC, burnin=BURNIN,
        proposal_std=0.3, seed=seed,
    )
    summ = posterior_summary(mcmc_samples)
    svgd_samples, _ = run_svgd_inference(
        data_sub, prior_type=prior_type, pseudo_data=pseudo_data, weight=weight,
        n_particles=N_PARTICLES, n_iter=SVGD_ITER, stepsize=SVGD_LR, seed=seed,
    )
    return {
        "n": n, "prior_type": prior_type,
        "mcmc": {"samples": mcmc_samples, "summary": summ, "accept_rate": accept},
        "svgd": {"samples": svgd_samples, "summary": posterior_summary(svgd_samples)},
    }


def main():
    print("=" * 60)
    print("Giving-game pipeline with gpt-3.5-turbo LLM prior (leakage ablation)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading pooled data and gpt-3.5 prior...")
    # Pooled data to match the original gpt-4o pipeline (96 train, 41 test).
    d = load_dictator_data()
    all_data = d["sharing_ratio"]
    train_data, test_data = train_test_split(all_data, test_frac=0.3, seed=SEED)
    print(f"  pooled train={len(train_data)}, test={len(test_data)}")

    pseudo_path = PROJECT_ROOT / "data" / "llm_prior_samples_gpt35.npy"
    if not pseudo_path.exists():
        raise SystemExit(f"Run elicit_llm_prior_gpt35.py first; missing {pseudo_path}")
    pseudo = np.load(pseudo_path)
    print(f"  gpt-3.5 prior: n={len(pseudo)}, mean={np.mean(pseudo):.3f}, std={np.std(pseudo):.3f}")

    print("\n[2/3] Running flat vs gpt-3.5-LLM prior at each n...")
    vars_flat, vars_llm = [], []
    samples_flat, samples_llm = [], []
    sizes_used = []
    for n in SAMPLE_SIZES:
        if n > len(train_data):
            continue
        print(f"  n={n}: flat...", end=" ", flush=True)
        rf = run_pair(train_data, n, "flat", seed=SEED)
        print(f"mu={rf['mcmc']['summary']['mu_mean']:.3f}, var={rf['mcmc']['summary']['mu_var']:.4f} | LLM...", end=" ", flush=True)
        rl = run_pair(train_data, n, "llm",
                      pseudo_data=pseudo, weight=LLM_PRIOR_WEIGHT, seed=SEED)
        print(f"mu={rl['mcmc']['summary']['mu_mean']:.3f}, var={rl['mcmc']['summary']['mu_var']:.4f}")
        vars_flat.append(rf["mcmc"]["summary"]["mu_var"])
        vars_llm.append(rl["mcmc"]["summary"]["mu_var"])
        samples_flat.append(rf["mcmc"]["samples"])
        samples_llm.append(rl["mcmc"]["samples"])
        sizes_used.append(n)

    target_var = (vars_flat[0] + vars_flat[-1]) / 2
    nstar_flat = find_nstar(sizes_used, vars_flat, target_var)
    nstar_llm = find_nstar(sizes_used, vars_llm, target_var)
    rho = compute_reduction_ratio(nstar_flat, nstar_llm)

    cov_flat = empirical_coverage(samples_flat[sizes_used.index(50)], test_data) \
        if 50 in sizes_used else None
    cov_llm = empirical_coverage(samples_llm[sizes_used.index(50)], test_data) \
        if 50 in sizes_used else None

    print("\n[3/3] Saving...")
    summary = {
        "model": "gpt-3.5-turbo",
        "game_type": "giving",
        "sample_sizes": sizes_used,
        "vars_flat_mcmc": [float(v) for v in vars_flat],
        "vars_llm_mcmc": [float(v) for v in vars_llm],
        "nstar_flat": float(nstar_flat),
        "nstar_llm": float(nstar_llm),
        "rho": float(rho),
        "target_var": float(target_var),
        "coverage_flat_n50": float(cov_flat) if cov_flat is not None else None,
        "coverage_llm_n50": float(cov_llm) if cov_llm is not None else None,
        "llm_prior_weight": LLM_PRIOR_WEIGHT,
        "llm_prior_mean": float(np.mean(pseudo)),
        "llm_prior_std": float(np.std(pseudo)),
        "data_mean": float(np.mean(all_data)),
        "data_std": float(np.std(all_data)),
    }
    with open(RESULTS_DIR / "experiment_results_gpt35.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"  rho_gpt35 = {rho:.3f}")
    print(f"  coverage (flat n=50) = {cov_flat}, coverage (LLM n=50) = {cov_llm}")
    print(f"  saved: results/experiment_results_gpt35.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
