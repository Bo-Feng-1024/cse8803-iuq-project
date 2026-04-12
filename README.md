# LLM-Informed Bayesian Priors for Reducing Experimental Cost in Behavioral Economics

**CSE 8803 IUQ** (Introduction to Uncertainty Quantification) — Spring 2026, Georgia Tech

**Author:** Bo Feng (bfeng66@gatech.edu)

## Overview

This project implements a Bayesian framework in which LLM (GPT-4o) simulations serve as an informative prior and real human observations provide the likelihood. We compare a flat-prior MCMC baseline against an LLM-informed SVGD frontier method on dictator game data, measuring the sample size reduction ratio $\rho = n^*_{\text{LLM}} / n^*_{\text{flat}}$.

## Repository Structure

```
project/
├── src/                    # Library code
│   ├── data_loader.py      # Load and preprocess dictator game data
│   ├── mcmc.py             # Metropolis-Hastings MCMC with Beta likelihood
│   ├── svgd.py             # Stein Variational Gradient Descent
│   └── evaluation.py       # Calibration, n*, reduction ratio metrics
├── scripts/
│   ├── run_experiment.py   # Main experiment: baseline vs frontier comparison
│   └── elicit_llm_prior.py # Query GPT-4o for dictator game simulations
├── data/
│   ├── dictator_game.csv   # Human data (137 obs, von Blanckenburg et al. 2023)
│   └── llm_prior_samples.npy  # 500 GPT-4o pseudo-observations
├── results/                # Generated experiment outputs (JSON)
├── figures/                # Generated plots (PDF + PNG)
├── checkin/                # Project check-in report (LaTeX + PDF)
└── project_proposal/       # Submitted proposal (LaTeX + PDF)
```

## Setup

**Requirements:** Python 3.10+, NumPy, SciPy, Matplotlib, OpenAI (for LLM elicitation only)

```bash
pip install numpy scipy matplotlib openai python-dotenv
```

To re-run the LLM elicitation, create a `.env` file:
```
OPENAI_API="sk-..."
```

## Reproduce Main Results

```bash
# Step 1 (optional): Re-elicit LLM prior from GPT-4o
python scripts/elicit_llm_prior.py

# Step 2: Run full experiment (baseline MCMC + frontier SVGD + evaluation)
python scripts/run_experiment.py
```

This generates:
- `results/experiment_results.json` — all numerical results
- `figures/nstar_curve.pdf` — sample size vs posterior variance (main figure)
- `figures/posterior_n{5,10,20,50,90}.pdf` — posterior comparisons
- `figures/failure_mode.pdf` — coverage under prior misspecification

**Expected output:**
```
Sample size reduction ratio: rho = 0.261
Coverage (flat, n=50):  0.976
Coverage (LLM, n=50):   0.976
```

## Key Results

| Metric | Value |
|--------|-------|
| Sample size reduction ratio ($\rho$) | 0.26 |
| Coverage (flat prior, n=50) | 97.6% |
| Coverage (LLM prior, n=50) | 97.6% |
| Failure mode | Prior shift +1 std degrades coverage to 83% |

## Data Sources

- **Human data:** von Blanckenburg, K., Tebbe, E., & Iseke, A. (2023). *Giving and taking in dictator games — differences by gender?* Journal of Comments and Replications in Economics, 2(2023-1). DOI: 10.18718/81781.27
- **LLM prior:** 500 simulated dictator game decisions from GPT-4o (`gpt-4o`, temperature=1.0)

## Random Seeds

All experiments use `seed=42` for reproducibility. The LLM elicitation uses `seed=42+batch_index` per API call.
