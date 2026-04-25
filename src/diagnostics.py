"""Posterior diagnostics: effective sample size (ESS) for MCMC chains."""

import numpy as np


def autocorr_1d(x, max_lag=None):
    """Autocorrelation of a 1-D array via FFT."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:max_lag]
    acf /= acf[0] + 1e-12
    return acf


def effective_sample_size(samples):
    """ESS via initial-positive-sequence (Geyer 1992).

    Parameters
    ----------
    samples : 1-D np.ndarray, MCMC chain values.

    Returns
    -------
    ess : float, 0 < ess <= n.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    n = len(samples)
    if n < 4:
        return float(n)
    acf = autocorr_1d(samples, max_lag=min(n - 1, 1000))
    # Sum pairs (rho_{2k} + rho_{2k+1}); stop at first non-positive pair.
    tau = 1.0
    k = 1
    while 2 * k + 1 < len(acf):
        pair = acf[2 * k] + acf[2 * k + 1]
        if pair <= 0:
            break
        tau += 2 * pair
        k += 1
    return n / max(tau, 1.0)
