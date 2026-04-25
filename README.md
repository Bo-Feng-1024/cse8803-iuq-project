# LLM-Informed Bayesian Priors for Reducing Experimental Cost in Behavioral Economics

**CSE 8803 IUQ** (Introduction to Uncertainty Quantification) — Spring 2026, Georgia Tech

**Author:** Bo Feng (bfeng66@gatech.edu)

**Final report:** [`final_report/final_report.pdf`](final_report/final_report.pdf)
**Slide deck:** [`final_report/slides/slides.pdf`](final_report/slides/slides.pdf)
**Recorded video:** see MediaSpace link in `final_report/final_report.pdf` (page 1)

## Overview

This project implements a Bayesian framework in which LLM (GPT-4o) simulations serve as an informative prior and real human observations provide the likelihood. We compare a flat-prior MCMC baseline against an LLM-informed SVGD frontier method on dictator game data, measuring the sample-size reduction ratio $\rho = n^{\ast}_{\text{LLM}} / n^{\ast}_{\text{flat}}$.

## Repository Structure

```
project/
├── src/                    # Library code
│   ├── data_loader.py      # Load and preprocess dictator game data
│   ├── mcmc.py             # Metropolis-Hastings MCMC with Beta likelihood
│   ├── svgd.py             # Stein Variational Gradient Descent
│   ├── evaluation.py       # Calibration, n*, reduction ratio metrics
│   └── diagnostics.py      # Effective sample size (Geyer 1992)
├── scripts/
│   ├── elicit_llm_prior.py             # Elicit GPT-4o giving-game prior
│   ├── elicit_llm_prior_taking.py      # Elicit GPT-4o taking-game prior
│   ├── elicit_llm_prior_gpt35.py       # Elicit gpt-3.5 (leakage ablation)
│   ├── run_experiment.py               # Main: baseline + frontier (giving)
│   ├── run_experiment_taking.py        # Cross-game replication (taking)
│   ├── run_experiment_gpt35.py         # gpt-3.5 leakage ablation
│   ├── run_failure_mode_dense.py       # 25-point misspecification curve
│   └── run_compute_pareto.py           # MCMC vs SVGD compute-accuracy
├── data/
│   ├── dictator_game.csv               # 137 obs, von Blanckenburg et al. 2023
│   ├── llm_prior_samples.npy           # 500 gpt-4o giving-game pseudo-obs
│   ├── llm_prior_samples_taking.npy    # 500 gpt-4o taking-game pseudo-obs
│   └── llm_prior_samples_gpt35.npy     # 500 gpt-3.5-turbo pseudo-obs
├── results/                            # Generated JSON outputs
├── figures/                            # Generated PDF + PNG figures
├── checkin/                            # Check-in report (PDF + LaTeX)
├── project_proposal/                   # Submitted proposal (PDF + LaTeX)
├── final_report/
│   ├── final_report.tex / .pdf         # 9-page final report
│   ├── references.bib                  # Bibliography
│   └── slides/
│       ├── slides.tex / .pdf           # Beamer deck
│       └── slides.md                   # Slidev source (alternative)
├── environment.yml                     # Conda environment spec
└── README.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate cse8803-iuq-project
```

Or via pip:
```bash
pip install numpy scipy matplotlib openai python-dotenv
```

To re-run the LLM elicitation, create a `.env` file:
```
OPENAI_API="sk-..."
```
LLM elicitation is **optional**: pseudo-observations are committed to `data/llm_prior_samples*.npy`, so the rest of the pipeline reproduces deterministically without an API key.

## Reproduce Main Results

```bash
# Step 1 (optional): Re-elicit priors from OpenAI
python scripts/elicit_llm_prior.py            # gpt-4o, giving game
python scripts/elicit_llm_prior_taking.py     # gpt-4o, taking game
python scripts/elicit_llm_prior_gpt35.py      # gpt-3.5-turbo, leakage ablation

# Step 2: Run full pipeline (5 scripts, end-to-end < 1 hour on a laptop)
python scripts/run_experiment.py              # rho ~ 0.26 (giving game baseline)
python scripts/run_experiment_taking.py       # rho ~ 0.55 (taking-game replication)
python scripts/run_experiment_gpt35.py        # rho ~ 0.26 with gpt-3.5 prior
python scripts/run_failure_mode_dense.py      # 25-point misspecification grid
python scripts/run_compute_pareto.py          # MCMC vs SVGD Pareto

# Step 3 (optional): Compile report and slides
cd final_report && latexmk -pdf final_report.tex
cd slides && latexmk -pdf slides.tex
```

This generates:
- `results/experiment_results.json` — main giving-game results
- `results/experiment_results_taking.json` — taking-game results
- `results/experiment_results_gpt35.json` — gpt-3.5 leakage ablation
- `results/failure_mode_dense.json` — 25-point coverage grid
- `results/pareto.json` — compute-accuracy sweep
- `figures/nstar_curve.pdf` — main figure (giving)
- `figures/nstar_curve_taking.pdf` — cross-game replication
- `figures/failure_mode_dense.pdf` — continuous coverage curve
- `figures/pareto.pdf` — MCMC vs SVGD compute-accuracy

**Expected key numbers:**
```
Sample size reduction ratio (giving):  rho = 0.261
Sample size reduction ratio (taking):  rho = 0.547
gpt-3.5 leakage ablation:              rho = 0.261, coverage(LLM,n=50) = 0.829
Coverage (gpt-4o LLM, n=50):           0.976
```

## Key Results

| Metric | Value |
|--------|-------|
| $\rho$ (gpt-4o, giving game) | **0.26** |
| $\rho$ (gpt-4o, taking game) | **0.55** |
| $\rho$ (gpt-3.5, giving game, leakage ablation) | **0.26** |
| Coverage (gpt-4o LLM prior, n=50) | 97.6% |
| Coverage (gpt-3.5 LLM prior, n=50) | 82.9% |
| Coverage (flat prior, n=50) | 97.6% |
| Failure mode | $\delta \in [+0.25, +1.5]$ shifts coverage to 0.83 |
| MCMC ESS/sec at $d=2, n=50$ | $\sim 1.2 \times 10^4$ |
| SVGD ESS/sec at $d=2, n=50$ | $\sim$14--270 |

## Data Sources

- **Human data:** von Blanckenburg, K., Tebbe, E., & Iseke, A. (2023). *Giving and taking in dictator games — differences by gender?* Journal of Comments and Replications in Economics, 2(2023-1). DOI: 10.18718/81781.27
- **LLM priors:** 500 simulated decisions each from `gpt-4o` and `gpt-3.5-turbo` (temperature=1.0).

## Reproducibility

All experiments use `seed=42`. The LLM elicitation uses `seed=42+batch_index` per API call. Random initial states (MCMC starting point, SVGD particle init) are deterministic functions of this seed. With elicited pseudo-data committed to the repo, the pipeline reproduces bit-for-bit without network access.
