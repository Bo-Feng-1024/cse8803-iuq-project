"""Metropolis-Hastings MCMC for Beta-likelihood inference on sharing ratios."""

import numpy as np
from scipy.special import betaln, digamma


def log_beta_likelihood(alpha, beta, data):
    """Log-likelihood: data ~ Beta(alpha, beta).

    Parameters
    ----------
    alpha, beta : float (must be > 0)
    data : np.ndarray of sharing ratios in (0, 1)

    Returns
    -------
    float
    """
    if alpha <= 0 or beta <= 0:
        return -np.inf
    n = len(data)
    return (
        -n * betaln(alpha, beta)
        + (alpha - 1) * np.sum(np.log(data))
        + (beta - 1) * np.sum(np.log(1 - data))
    )


def grad_log_beta_likelihood(alpha, beta, data):
    """Gradient of log-likelihood w.r.t. (alpha, beta).

    Returns
    -------
    np.ndarray of shape (2,)
    """
    n = len(data)
    psi_ab = digamma(alpha + beta)
    d_alpha = np.sum(np.log(data)) - n * digamma(alpha) + n * psi_ab
    d_beta = np.sum(np.log(1 - data)) - n * digamma(beta) + n * psi_ab
    return np.array([d_alpha, d_beta])


def log_prior_flat(alpha, beta, max_val=100.0):
    """Flat (uniform) prior on (0, max_val) x (0, max_val)."""
    if alpha <= 0 or beta <= 0 or alpha > max_val or beta > max_val:
        return -np.inf
    return 0.0


def log_prior_llm(alpha, beta, pseudo_data, weight=1.0):
    """LLM power prior: weighted pseudo-likelihood from LLM simulations.

    p_LLM(alpha, beta) ∝ prod_j Beta(y_j; alpha, beta)^(w/M)
    where y_j are LLM pseudo-observations, w is the effective weight,
    and M = len(pseudo_data).

    Parameters
    ----------
    alpha, beta : float
    pseudo_data : np.ndarray of LLM-simulated sharing ratios
    weight : float
        Effective weight (number of pseudo-observations to count).
        weight=10 means the prior is worth 10 real observations.
    """
    if alpha <= 0 or beta <= 0:
        return -np.inf
    M = len(pseudo_data)
    scale = weight / M
    return scale * (
        -M * betaln(alpha, beta)
        + (alpha - 1) * np.sum(np.log(pseudo_data))
        + (beta - 1) * np.sum(np.log(1 - pseudo_data))
    )


def grad_log_prior_llm(alpha, beta, pseudo_data, weight=1.0):
    """Gradient of LLM power prior w.r.t. (alpha, beta)."""
    if alpha <= 0 or beta <= 0:
        return np.array([0.0, 0.0])
    M = len(pseudo_data)
    scale = weight / M
    psi_ab = digamma(alpha + beta)
    d_alpha = scale * (np.sum(np.log(pseudo_data)) - M * digamma(alpha) + M * psi_ab)
    d_beta = scale * (np.sum(np.log(1 - pseudo_data)) - M * digamma(beta) + M * psi_ab)
    return np.array([d_alpha, d_beta])


def make_log_posterior(data, prior_type="flat", pseudo_data=None, weight=1.0):
    """Create log-posterior function.

    Parameters
    ----------
    data : np.ndarray
    prior_type : str, 'flat' or 'llm'
    pseudo_data : np.ndarray, required if prior_type='llm'
    weight : float, LLM prior weight

    Returns
    -------
    log_post : callable(alpha, beta) -> float
    grad_log_post : callable(alpha, beta) -> np.ndarray of shape (2,)
    """
    def log_post(alpha, beta):
        ll = log_beta_likelihood(alpha, beta, data)
        if prior_type == "flat":
            lp = log_prior_flat(alpha, beta)
        else:
            lp = log_prior_flat(alpha, beta) + log_prior_llm(
                alpha, beta, pseudo_data, weight
            )
        return ll + lp

    def grad_log_post(alpha, beta):
        g = grad_log_beta_likelihood(alpha, beta, data)
        if prior_type == "llm":
            g += grad_log_prior_llm(alpha, beta, pseudo_data, weight)
        return g

    return log_post, grad_log_post


def metropolis_hastings(
    log_posterior, init, n_samples=10000, burnin=2000, proposal_std=0.3, seed=42
):
    """Metropolis-Hastings sampler in log-space (log alpha, log beta).

    Works in log-transformed space for positivity, with Gaussian proposals.

    Parameters
    ----------
    log_posterior : callable(alpha, beta) -> float
    init : tuple (alpha0, beta0)
    n_samples : int
    burnin : int
    proposal_std : float
    seed : int

    Returns
    -------
    samples : np.ndarray of shape (n_samples, 2), columns are (alpha, beta)
    accept_rate : float
    """
    rng = np.random.default_rng(seed)
    total = n_samples + burnin

    # Work in log-space
    log_theta = np.log(np.array(init, dtype=float))
    samples = np.zeros((total, 2))
    n_accept = 0

    current_lp = log_posterior(*np.exp(log_theta))

    for i in range(total):
        # Propose in log-space
        log_theta_prop = log_theta + rng.normal(0, proposal_std, size=2)
        theta_prop = np.exp(log_theta_prop)

        prop_lp = log_posterior(*theta_prop)

        # Jacobian correction for log-transform
        log_hastings = np.sum(log_theta_prop) - np.sum(log_theta)
        log_accept = prop_lp - current_lp + log_hastings

        if np.log(rng.uniform()) < log_accept:
            log_theta = log_theta_prop
            current_lp = prop_lp
            n_accept += 1

        samples[i] = np.exp(log_theta)

    return samples[burnin:], n_accept / total


def posterior_summary(samples):
    """Compute posterior summary statistics.

    Parameters
    ----------
    samples : np.ndarray of shape (n, 2), columns are (alpha, beta)

    Returns
    -------
    dict with posterior mean, std, credible intervals for alpha, beta, and mu=alpha/(alpha+beta)
    """
    alpha, beta = samples[:, 0], samples[:, 1]
    mu = alpha / (alpha + beta)

    def ci(x, level=0.95):
        lo = np.percentile(x, 100 * (1 - level) / 2)
        hi = np.percentile(x, 100 * (1 + level) / 2)
        return (lo, hi)

    return {
        "alpha_mean": np.mean(alpha),
        "alpha_std": np.std(alpha),
        "alpha_ci95": ci(alpha),
        "beta_mean": np.mean(beta),
        "beta_std": np.std(beta),
        "beta_ci95": ci(beta),
        "mu_mean": np.mean(mu),
        "mu_std": np.std(mu),
        "mu_ci95": ci(mu),
        "mu_var": np.var(mu),
    }
