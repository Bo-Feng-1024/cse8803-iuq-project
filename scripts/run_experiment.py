#!/usr/bin/env python3
"""Main experiment script: baseline MCMC vs frontier SVGD on dictator game data.

Usage:
    cd project/
    python scripts/run_experiment.py
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dictator_data, subsample, train_test_split
from src.mcmc import metropolis_hastings, make_log_posterior, posterior_summary
from src.svgd import run_svgd_inference
from src.evaluation import (
    empirical_coverage,
    posterior_mu_variance,
    find_nstar,
    compute_reduction_ratio,
)

# ── Configuration ──────────────────────────────────────────────
SAMPLE_SIZES = [5, 10, 20, 50, 90]
N_MCMC = 15000
BURNIN = 3000
N_PARTICLES = 100
SVGD_ITER = 2000
SVGD_LR = 0.05
LLM_PRIOR_WEIGHT = 20.0  # effective pseudo-observations from LLM
SEED = 42
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_or_generate_llm_prior(path=None):
    """Load LLM prior pseudo-data, or generate synthetic placeholder."""
    if path and Path(path).exists():
        data = np.load(path)
        return data

    # Placeholder: simulate what GPT-4o might produce for a dictator game
    # LLMs tend to be generous, clustering around 0.4-0.5 sharing ratio
    rng = np.random.default_rng(123)
    # Beta(3, 3.5) gives mean ≈ 0.46, moderate spread — plausible LLM behavior
    pseudo = rng.beta(3.0, 3.5, size=500)
    pseudo = np.clip(pseudo, 1e-3, 1 - 1e-3)
    return pseudo


def run_single_experiment(data_full, n, prior_type, pseudo_data=None, weight=1.0, seed=42):
    """Run MCMC and SVGD for a single sample size and prior configuration."""
    rng = np.random.default_rng(seed)
    data_sub = subsample(data_full, n, rng=rng)

    # ── MCMC ──
    t0 = time.time()
    log_post, _ = make_log_posterior(
        data_sub, prior_type=prior_type, pseudo_data=pseudo_data, weight=weight
    )
    mcmc_samples, accept_rate = metropolis_hastings(
        log_post, init=(2.0, 2.0), n_samples=N_MCMC, burnin=BURNIN,
        proposal_std=0.3, seed=seed,
    )
    mcmc_time = time.time() - t0
    mcmc_summary = posterior_summary(mcmc_samples)

    # ── SVGD ──
    t0 = time.time()
    svgd_samples, svgd_history = run_svgd_inference(
        data_sub, prior_type=prior_type, pseudo_data=pseudo_data, weight=weight,
        n_particles=N_PARTICLES, n_iter=SVGD_ITER, stepsize=SVGD_LR, seed=seed,
    )
    svgd_time = time.time() - t0
    svgd_summary = posterior_summary(svgd_samples)

    return {
        "n": n,
        "prior_type": prior_type,
        "mcmc": {
            "samples": mcmc_samples,
            "summary": mcmc_summary,
            "accept_rate": accept_rate,
            "time": mcmc_time,
        },
        "svgd": {
            "samples": svgd_samples,
            "history": svgd_history,
            "summary": svgd_summary,
            "time": svgd_time,
        },
    }


def plot_posterior_comparison(results_flat, results_llm, n, save_dir):
    """Plot posterior comparison for a given sample size."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, key, label in zip(
        axes,
        ["alpha", "beta", "mu"],
        [r"$\alpha$", r"$\beta$", r"$\mu = \alpha/(\alpha+\beta)$"],
    ):
        if key == "mu":
            flat_vals = results_flat["mcmc"]["samples"][:, 0] / (
                results_flat["mcmc"]["samples"][:, 0] + results_flat["mcmc"]["samples"][:, 1]
            )
            llm_vals = results_llm["mcmc"]["samples"][:, 0] / (
                results_llm["mcmc"]["samples"][:, 0] + results_llm["mcmc"]["samples"][:, 1]
            )
            svgd_flat = results_flat["svgd"]["samples"][:, 0] / (
                results_flat["svgd"]["samples"][:, 0] + results_flat["svgd"]["samples"][:, 1]
            )
            svgd_llm = results_llm["svgd"]["samples"][:, 0] / (
                results_llm["svgd"]["samples"][:, 0] + results_llm["svgd"]["samples"][:, 1]
            )
        else:
            idx = 0 if key == "alpha" else 1
            flat_vals = results_flat["mcmc"]["samples"][:, idx]
            llm_vals = results_llm["mcmc"]["samples"][:, idx]
            svgd_flat = results_flat["svgd"]["samples"][:, idx]
            svgd_llm = results_llm["svgd"]["samples"][:, idx]

        ax.hist(flat_vals, bins=40, alpha=0.4, density=True, label="MCMC (flat)", color="C0")
        ax.hist(llm_vals, bins=40, alpha=0.4, density=True, label="MCMC (LLM)", color="C1")
        # SVGD as scatter on x-axis
        ax.scatter(svgd_flat, np.zeros(len(svgd_flat)) - 0.1, marker="|", alpha=0.5, color="C0", label="SVGD (flat)")
        ax.scatter(svgd_llm, np.zeros(len(svgd_llm)) - 0.2, marker="|", alpha=0.5, color="C1", label="SVGD (LLM)")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        if key == "mu":
            ax.legend(fontsize=7)

    fig.suptitle(f"Posterior comparison: n = {n}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / f"posterior_n{n}.pdf", dpi=150)
    plt.savefig(save_dir / f"posterior_n{n}.png", dpi=150)
    plt.close()


def plot_nstar_curve(sample_sizes, vars_flat, vars_llm, target_var, save_dir):
    """Plot posterior variance vs sample size for both methods."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sample_sizes, vars_flat, "o-", color="C0", label="Flat prior (MCMC)")
    ax.plot(sample_sizes, vars_llm, "s-", color="C1", label="LLM prior (MCMC)")
    ax.axhline(target_var, color="gray", linestyle="--", alpha=0.7, label=f"Target var = {target_var:.4f}")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel(r"Posterior variance of $\mu$")
    ax.set_title("Sample Size vs Posterior Precision")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_dir / "nstar_curve.pdf", dpi=150)
    plt.savefig(save_dir / "nstar_curve.png", dpi=150)
    plt.close()


def plot_failure_mode(shifts, coverages, save_dir):
    """Plot coverage degradation under prior misspecification."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(shifts, coverages, "o-", color="C3")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.7, label="Nominal 95%")
    ax.set_xlabel("Prior mean shift (in std devs)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Failure Mode: Prior Misspecification")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "failure_mode.pdf", dpi=150)
    plt.savefig(save_dir / "failure_mode.png", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("LLM-Informed Bayesian Priors: Dictator Game Experiment")
    print("=" * 60)

    # ── Setup ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n[1/6] Loading data...")
    data_dict = load_dictator_data()
    all_data = data_dict["sharing_ratio"]
    train_data, test_data = train_test_split(all_data, test_frac=0.3, seed=SEED)
    print(f"  Total: {len(all_data)}, Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"  Mean sharing ratio: {np.mean(all_data):.3f}, Std: {np.std(all_data):.3f}")

    # ── LLM prior ──
    print("\n[2/6] Loading LLM prior pseudo-data...")
    llm_prior_path = PROJECT_ROOT / "data" / "llm_prior_samples.npy"
    pseudo_data = load_or_generate_llm_prior(llm_prior_path)
    print(f"  Pseudo-observations: {len(pseudo_data)}")
    print(f"  LLM prior mean: {np.mean(pseudo_data):.3f}, std: {np.std(pseudo_data):.3f}")
    print(f"  Weight: {LLM_PRIOR_WEIGHT}")

    # ── Run experiments across sample sizes ──
    print("\n[3/6] Running experiments...")
    results_all = {"flat": [], "llm": []}
    vars_flat_mcmc, vars_llm_mcmc = [], []

    for n in SAMPLE_SIZES:
        print(f"\n  --- n = {n} ---")

        # Flat prior
        print(f"  Flat prior: MCMC + SVGD...", end=" ", flush=True)
        r_flat = run_single_experiment(train_data, n, "flat", seed=SEED)
        vars_flat_mcmc.append(r_flat["mcmc"]["summary"]["mu_var"])
        results_all["flat"].append(r_flat)
        print(f"done (MCMC accept={r_flat['mcmc']['accept_rate']:.2f}, "
              f"mu={r_flat['mcmc']['summary']['mu_mean']:.3f}±{r_flat['mcmc']['summary']['mu_std']:.3f})")

        # LLM prior
        print(f"  LLM prior:  MCMC + SVGD...", end=" ", flush=True)
        r_llm = run_single_experiment(
            train_data, n, "llm", pseudo_data=pseudo_data, weight=LLM_PRIOR_WEIGHT, seed=SEED
        )
        vars_llm_mcmc.append(r_llm["mcmc"]["summary"]["mu_var"])
        results_all["llm"].append(r_llm)
        print(f"done (MCMC accept={r_llm['mcmc']['accept_rate']:.2f}, "
              f"mu={r_llm['mcmc']['summary']['mu_mean']:.3f}±{r_llm['mcmc']['summary']['mu_std']:.3f})")

    # ── Compute n* and rho ──
    print("\n[4/6] Computing sample size reduction ratio...")
    # Target variance: midpoint between variance at largest and smallest n (flat prior)
    # This ensures the target is achievable and meaningful
    var_at_max_n = vars_flat_mcmc[-1]
    var_at_min_n = vars_flat_mcmc[0]
    target_var = (var_at_max_n + var_at_min_n) / 2
    print(f"  Var at n={SAMPLE_SIZES[0]} (flat): {var_at_min_n:.6f}")
    print(f"  Var at n={SAMPLE_SIZES[-1]} (flat): {var_at_max_n:.6f}")
    print(f"  Target variance: {target_var:.6f}")

    nstar_flat = find_nstar(SAMPLE_SIZES, vars_flat_mcmc, target_var)
    nstar_llm = find_nstar(SAMPLE_SIZES, vars_llm_mcmc, target_var)
    rho = compute_reduction_ratio(nstar_flat, nstar_llm)

    print(f"  n*_flat = {nstar_flat:.1f}")
    print(f"  n*_LLM  = {nstar_llm:.1f}")
    print(f"  rho = n*_LLM / n*_flat = {rho:.3f}")

    # ── Calibration ──
    print("\n[5/6] Computing calibration (coverage on held-out data)...")
    # Use n=50 results for calibration
    idx_50 = SAMPLE_SIZES.index(50)
    cov_flat = empirical_coverage(results_all["flat"][idx_50]["mcmc"]["samples"], test_data)
    cov_llm = empirical_coverage(results_all["llm"][idx_50]["mcmc"]["samples"], test_data)
    print(f"  Coverage (flat prior, n=50): {cov_flat:.3f}")
    print(f"  Coverage (LLM prior, n=50):  {cov_llm:.3f}")

    # ── Failure mode: prior misspecification ──
    print("\n[5b/6] Failure mode analysis (prior shift)...")
    shifts = [-2, -1, 0, 1, 2]
    coverages_shifted = []
    pseudo_mean = np.mean(pseudo_data)
    pseudo_std = np.std(pseudo_data)

    for shift in shifts:
        shifted_pseudo = np.clip(pseudo_data + shift * pseudo_std, 1e-3, 1 - 1e-3)
        r = run_single_experiment(
            train_data, 50, "llm", pseudo_data=shifted_pseudo,
            weight=LLM_PRIOR_WEIGHT, seed=SEED
        )
        cov = empirical_coverage(r["mcmc"]["samples"], test_data)
        coverages_shifted.append(cov)
        print(f"  Shift={shift:+d} std: coverage={cov:.3f}")

    # ── Plots ──
    print("\n[6/6] Generating figures...")
    for i, n in enumerate(SAMPLE_SIZES):
        plot_posterior_comparison(
            results_all["flat"][i], results_all["llm"][i], n, FIGURES_DIR
        )

    plot_nstar_curve(SAMPLE_SIZES, vars_flat_mcmc, vars_llm_mcmc, target_var, FIGURES_DIR)
    plot_failure_mode(shifts, coverages_shifted, FIGURES_DIR)

    # ── Save results ──
    summary = {
        "sample_sizes": SAMPLE_SIZES,
        "vars_flat_mcmc": [float(v) for v in vars_flat_mcmc],
        "vars_llm_mcmc": [float(v) for v in vars_llm_mcmc],
        "nstar_flat": float(nstar_flat),
        "nstar_llm": float(nstar_llm),
        "rho": float(rho),
        "target_var": float(target_var),
        "coverage_flat_n50": float(cov_flat),
        "coverage_llm_n50": float(cov_llm),
        "failure_mode_shifts": shifts,
        "failure_mode_coverages": [float(c) for c in coverages_shifted],
        "llm_prior_weight": LLM_PRIOR_WEIGHT,
        "llm_prior_mean": float(np.mean(pseudo_data)),
        "llm_prior_std": float(np.std(pseudo_data)),
        "data_mean": float(np.mean(all_data)),
        "data_std": float(np.std(all_data)),
        "n_total": len(all_data),
        "n_train": len(train_data),
        "n_test": len(test_data),
    }

    with open(RESULTS_DIR / "experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Sample size reduction ratio: rho = {rho:.3f}")
    print(f"  Coverage (flat, n=50):  {cov_flat:.3f}")
    print(f"  Coverage (LLM, n=50):   {cov_llm:.3f}")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Results saved to: {RESULTS_DIR / 'experiment_results.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
