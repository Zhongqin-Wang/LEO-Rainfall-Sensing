"""Pilot channel power without subtracting a noisy short noise-only record."""
import numpy as np


def coherent_pilot_power(csi, pilot_indices, max_lag=32):
    """Zero-lag signal covariance from nonzero temporal lags.

    Input CSI has CFO/SFO removed, but has NOT undergone data-dependent CPE
    rotation. White receiver noise has zero covariance at nonzero symbol lags.
    A local linear extrapolation cancels first-order phase decorrelation over
    max_lag*4.4 us. This assumes a quasi-static channel over that short interval;
    it is not unbiased under arbitrary interference or rapid phase dynamics.
    Negative estimates are retained, never clipped into a positive signal.
    """
    values = np.asarray(csi)[2:, pilot_indices]
    if values.ndim != 2 or not np.isfinite(values).all() or not 2 <= max_lag < len(values):
        raise ValueError("Need finite known-pilot CSI and at least three symbols")
    lags = np.arange(1, max_lag + 1, dtype=float)
    correlations = np.asarray([np.mean(values[int(lag):] * np.conj(values[:-int(lag)]), axis=0).real
                               for lag in lags])
    # First row of the OLS pseudoinverse estimates the intercept, retaining
    # signed cross-products instead of using amplitudes/noise-derived clipping.
    design = np.c_[np.ones(len(lags)), lags]
    weights = np.linalg.pinv(design)[0]
    return weights @ correlations
