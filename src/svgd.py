"""Stein Variational Gradient Descent (SVGD) for posterior inference."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def rbf_kernel(X, h=None):
    """Compute RBF kernel matrix and its gradient.

    Parameters
    ----------
    X : np.ndarray of shape (n_particles, d)
    h : float or None
        Bandwidth. If None, use median heuristic.

    Returns
    -------
    K : np.ndarray of shape (n, n), kernel matrix
    dK : np.ndarray of shape (n, n, d), gradient of K w.r.t. X
    """
    pairwise_sq = squareform(pdist(X, "sqeuclidean"))

    if h is None:
        med = np.median(pairwise_sq[pairwise_sq > 0])
        h = med / np.log(X.shape[0] + 1)
        h = max(h, 1e-5)

    K = np.exp(-pairwise_sq / (2 * h))

    # dK[i, j] = K[i,j] * (X[j] - X[i]) / h
    n, d = X.shape
    dK = np.zeros((n, n, d))
    for i in range(n):
        diff = X - X[i]  # (n, d)
        dK[i] = K[i, :, None] * diff / h

    return K, dK


def svgd(
    grad_log_prob,
    init_particles,
    n_iter=1000,
    stepsize=0.01,
    bandwidth=None,
    adam=True,
    seed=None,
):
    """Run SVGD to approximate a posterior distribution.

    Parameters
    ----------
    grad_log_prob : callable(particles) -> np.ndarray of shape (n, d)
        Gradient of log-posterior evaluated at each particle.
    init_particles : np.ndarray of shape (n_particles, d)
    n_iter : int
    stepsize : float
    bandwidth : float or None (median heuristic)
    adam : bool, use Adam optimizer

    Returns
    -------
    particles : np.ndarray of shape (n_particles, d)
    history : list of np.ndarray, particle positions at each iteration
    """
    particles = init_particles.copy()
    n, d = particles.shape

    # Adam parameters
    m = np.zeros_like(particles)
    v = np.zeros_like(particles)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    history = [particles.copy()]

    for t in range(1, n_iter + 1):
        # Compute kernel
        K, dK = rbf_kernel(particles, h=bandwidth)

        # Compute score (gradient of log prob) for each particle
        scores = grad_log_prob(particles)  # (n, d)

        # SVGD update direction: phi(x_i) = (1/n) sum_j [K(x_j, x_i) * score(x_j) + dK(x_j, x_i)]
        phi = (K @ scores + np.sum(dK, axis=0)) / n

        if adam:
            m = beta1 * m + (1 - beta1) * phi
            v = beta2 * v + (1 - beta2) * phi**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            particles += stepsize * m_hat / (np.sqrt(v_hat) + eps_adam)
        else:
            particles += stepsize * phi

        if t % 100 == 0 or t == n_iter:
            history.append(particles.copy())

    return particles, history


def run_svgd_inference(
    data,
    prior_type="flat",
    pseudo_data=None,
    weight=1.0,
    n_particles=100,
    n_iter=2000,
    stepsize=0.05,
    seed=42,
):
    """Run SVGD for Beta-likelihood inference on sharing ratios.

    Works in log-space (log alpha, log beta) for positivity constraint.

    Parameters
    ----------
    data : np.ndarray of sharing ratios
    prior_type : str, 'flat' or 'llm'
    pseudo_data : np.ndarray, LLM pseudo-observations (if prior_type='llm')
    weight : float, LLM prior weight
    n_particles : int
    n_iter : int
    stepsize : float
    seed : int

    Returns
    -------
    samples : np.ndarray of shape (n_particles, 2), columns are (alpha, beta)
    history : list of particle snapshots (in original space)
    """
    from .mcmc import make_log_posterior

    log_post, grad_log_post = make_log_posterior(
        data, prior_type=prior_type, pseudo_data=pseudo_data, weight=weight
    )

    rng = np.random.default_rng(seed)

    # Initialize particles in log-space around reasonable values
    # MoM estimate for initialization
    m = np.mean(data)
    v = np.var(data)
    if v > 0 and m > 0 and m < 1:
        kappa_init = m * (1 - m) / v - 1
        kappa_init = max(kappa_init, 2.0)
        alpha_init = m * kappa_init
        beta_init = (1 - m) * kappa_init
    else:
        alpha_init, beta_init = 2.0, 2.0

    log_init = np.log(np.array([alpha_init, beta_init]))
    init_particles = log_init + rng.normal(0, 0.3, size=(n_particles, 2))

    def grad_log_prob_batch(log_particles):
        """Compute gradients for all particles in log-space."""
        grads = np.zeros_like(log_particles)
        for i in range(log_particles.shape[0]):
            lp = log_particles[i]
            # Clamp to avoid extreme values
            lp = np.clip(lp, -5, 5)
            alpha, beta = np.exp(lp)
            # Gradient in original space
            g = grad_log_post(alpha, beta)
            # Chain rule for log-transform: d/d(log θ) = θ * d/dθ
            g_log = np.array([alpha, beta]) * g
            # Clip gradients for stability
            g_log = np.clip(g_log, -100, 100)
            grads[i] = g_log
        return grads

    final_log_particles, history_log = svgd(
        grad_log_prob_batch,
        init_particles,
        n_iter=n_iter,
        stepsize=stepsize,
        seed=seed,
    )

    # Convert back to original space
    samples = np.exp(final_log_particles)
    history = [np.exp(h) for h in history_log]

    return samples, history
