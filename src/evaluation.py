"""Evaluation metrics: calibration, sample-size reduction ratio, compute tradeoff."""

import numpy as np
from scipy.interpolate import interp1d


def empirical_coverage(samples, test_data, level=0.95):
    """Compute empirical coverage of credible intervals on held-out data.

    For each test observation y, check if it falls within the (level) posterior
    predictive interval implied by the posterior samples of (alpha, beta).

    Parameters
    ----------
    samples : np.ndarray of shape (n_samples, 2), columns (alpha, beta)
    test_data : np.ndarray of sharing ratios
    level : float, credible level (e.g. 0.95)

    Returns
    -------
    coverage : float, fraction of test points inside the interval
    """
    from scipy.stats import beta as beta_dist

    lo_q = (1 - level) / 2
    hi_q = 1 - lo_q

    covered = 0
    for y in test_data:
        # For each posterior sample, compute predictive CDF at y
        cdf_vals = np.array(
            [beta_dist.cdf(y, a, b) for a, b in samples]
        )
        # Posterior predictive CDF at y is the mean over posterior samples
        pred_cdf = np.mean(cdf_vals)
        if lo_q <= pred_cdf <= hi_q:
            covered += 1

    return covered / len(test_data)


def posterior_mu_variance(samples):
    """Variance of posterior mean mu = alpha/(alpha+beta)."""
    mu = samples[:, 0] / (samples[:, 0] + samples[:, 1])
    return np.var(mu)


def find_nstar(sample_sizes, variances, target_var):
    """Find minimum n* where posterior variance drops below target.

    Uses log-log interpolation.

    Parameters
    ----------
    sample_sizes : list of int
    variances : list of float
    target_var : float

    Returns
    -------
    n_star : float (interpolated), clipped to [min_n, 2*max_n]
    """
    sample_sizes = np.array(sample_sizes, dtype=float)
    variances = np.array(variances)

    # If all variances already below target, return smallest n
    if np.all(variances <= target_var):
        return sample_sizes[0]

    # If all variances above target, extrapolate in log-log space
    if np.all(variances > target_var):
        coeffs = np.polyfit(np.log(sample_sizes), np.log(variances), 1)
        log_n_star = (np.log(target_var) - coeffs[1]) / coeffs[0]
        return min(np.exp(log_n_star), 2 * sample_sizes[-1])

    # Interpolate in log-log space (variance decreases with n)
    log_n = np.log(sample_sizes)
    log_v = np.log(variances)
    # Sort by variance descending for interpolation
    sort_idx = np.argsort(-variances)
    f = interp1d(log_v[sort_idx], log_n[sort_idx], kind="linear", fill_value="extrapolate")
    n_star = np.exp(float(f(np.log(target_var))))
    return np.clip(n_star, sample_sizes[0], 2 * sample_sizes[-1])


def compute_reduction_ratio(nstar_flat, nstar_llm):
    """Compute sample size reduction ratio rho = n*_LLM / n*_flat."""
    return nstar_llm / nstar_flat
