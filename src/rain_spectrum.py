"""Configurable spectral rain attenuation and coherent propagation phase."""
import numpy as np
from scipy.fft import fft, ifft, fftfreq, next_fast_len
from itur.models import itu838


def spectral_coefficients(frequency_hz, elevation_deg, tilt_deg):
    # itur 0.4 returns (N,2) for vector frequency, but (2,N) for vector
    # elevation. Isolate this API asymmetry instead of silently mixing k/alpha.
    frequency = np.atleast_1d(np.asarray(frequency_hz, float))
    coefficients = np.asarray(itu838.rain_specific_attenuation_coefficients(
        frequency / 1e9, float(elevation_deg), float(tilt_deg)))
    if coefficients.shape != (frequency.size, 2):
        raise ValueError("Unexpected ITU rain coefficient dimensions")
    return coefficients[:, 0], coefficients[:, 1]


def attenuation_spectrum(frequency_hz, elevation_deg, tilt_deg, path_km, rates_mm_h):
    frequency = np.asarray(frequency_hz, dtype=float)
    rates = np.atleast_1d(np.asarray(rates_mm_h, dtype=float))
    if (frequency.ndim != 1 or not frequency.size or rates.ndim != 1 or not rates.size
            or not np.isfinite(frequency).all() or np.any((frequency < 1e9) | (frequency > 1e12))
            or not np.isfinite(rates).all() or np.any(rates < 0)
            or not np.isfinite([elevation_deg, tilt_deg, path_km]).all()
            or not 0 < elevation_deg <= 90 or path_km < 0):
        raise ValueError("Invalid P.838 rain spectrum inputs")
    k, alpha = spectral_coefficients(frequency, elevation_deg, tilt_deg)
    return np.mean(np.asarray(k)[:, None] * rates[None, :] ** np.asarray(alpha)[:, None], axis=1) * path_km


def row_rain_spectrum(row, config, frequency_offsets_hz, case="rain"):
    mode = config["propagation"].get("rain_frequency_model", "center_frequency_flat")
    if mode not in ("center_frequency_flat", "itu838_frequency_dependent", "mie_spherical_mp"):
        raise ValueError("Unknown propagation.rain_frequency_model")
    offsets = np.asarray(frequency_offsets_hz, dtype=float)
    if case == "reference" or not config["propagation"]["enabled"]["rain"]:
        return np.zeros_like(offsets)
    if mode == "mie_spherical_mp":
        from src.rain_complex import row_mie_spectrum
        return row_mie_spectrum(row, config, offsets, case)[0]
    if mode == "center_frequency_flat":
        return np.full_like(offsets, float(row["rain_db"]))
    rates = row.get("rain_rate_samples_mm_h", row.get("rain_rate_mm_h"))
    if rates is None:
        raise ValueError("Frequency-dependent rain generation requires weather samples")
    return attenuation_spectrum(float(config["link"]["carrier_hz"]) + offsets,
        float(row["elevation_deg"]), float(config["propagation"]["polarization_tilt_deg"]),
        float(row["rain_path_km"]), rates)


def row_rain_phase(row, config, frequency_offsets_hz, case="rain"):
    prop = config["propagation"]
    model = prop.get("rain_phase_model", "none")
    if model not in ("none", "mie_spherical_mp"):
        raise ValueError("Unknown propagation.rain_phase_model")
    if prop.get("rain_frequency_model") == "mie_spherical_mp" and model != "mie_spherical_mp":
        raise ValueError("Full complex Mie rain requires rain_phase_model: mie_spherical_mp")
    if model == "none":
        return np.zeros_like(np.asarray(frequency_offsets_hz, float))
    from src.rain_complex import row_mie_spectrum
    return row_mie_spectrum(row, config, frequency_offsets_hz, case)[1]


def apply_spectral_loss(iq, fs, frequency_offsets_hz, loss_db, guard=4096, phase_rad=None):
    """Guarded linear-record approximation to a smooth complex RF transfer.

    phase_rad must be unwrapped before interpolation. This is an excess-rain
    transfer in a frame-aligned time origin, not absolute satellite TOA.
    """
    x = np.asarray(iq, dtype=complex)
    f, loss = np.asarray(frequency_offsets_hz, float), np.asarray(loss_db, float)
    phase = np.zeros_like(loss) if phase_rad is None else np.asarray(phase_rad, float)
    if (x.ndim != 1 or not x.size or f.ndim != 1 or loss.shape != f.shape or f.size < 2
            or not np.isfinite(x).all() or not np.isfinite([fs]).all() or fs <= 0
            or phase.shape != f.shape or not np.isfinite(phase).all()
            or not np.isfinite(f).all() or not np.isfinite(loss).all() or guard < 1):
        raise ValueError("Invalid spectral channel inputs")
    order = np.argsort(f)
    if np.any(np.diff(f[order]) <= 0):
        raise ValueError("Spectral frequencies must be unique")
    if np.ptp(loss) < 1e-14 and np.ptp(phase) < 1e-14:
        return x * 10 ** (-loss[0] / 20) * np.exp(1j*phase[0])
    length = next_fast_len(len(x) + 2 * guard)
    padded = np.zeros(length, complex)
    padded[guard:guard + len(x)] = x
    offsets = fftfreq(length, 1/fs)
    gain = (10 ** (-np.interp(offsets, f[order], loss[order]) / 20)
            * np.exp(1j*np.interp(offsets, f[order], phase[order])))
    return ifft(fft(padded) * gain)[guard:guard + len(x)]
