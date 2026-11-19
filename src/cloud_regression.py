"""Shared feature extraction and ridge regression for cloud rain-rate models."""
import numpy as np

FEATURE_NAMES = ['sequence_mean_k', 'mean_change_k', 'last_frame_spatial_std_k']


def extract_features(cloud, valid, windows):
    """Use only historical cloud inputs; never use the rain labels here."""
    values = np.where(valid & np.isfinite(cloud), cloud, np.nan)
    features = []
    for indices in windows:
        frames = values[indices]
        means = np.nanmean(frames, axis=(1, 2))
        features.append([np.mean(means), means[-1]-means[0], np.nanstd(frames[-1])])
    result = np.asarray(features, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Feature extraction requires valid cloud pixels in every frame')
    return result


def fit_ridge(x, y, alpha):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x-mean)/scale
    intercept = float(y.mean())
    # Minimize squared error + alpha * squared coefficient norm; intercept is unpenalized.
    coefficients = np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]), z.T@(y-intercept))
    return mean, scale, coefficients, intercept


def predict(x, mean, scale, coefficients, intercept):
    return (x-mean)/scale@coefficients+intercept

