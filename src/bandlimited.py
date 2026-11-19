"""Amplitude-preserving resampling for nearly Nyquist-wide complex signals."""
import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import map_coordinates
from scipy.signal import resample


def sample_at(iq, positions, oversample=8):
    """Bandlimited FFT interpolation followed by cubic interpolation on an 8x grid.

    Zero guards isolate the finite capture from periodic FFT wraparound. This
    retains the original band to its Nyquist edge; unlike linear interpolation,
    it does not impose a large timing-dependent attenuation on edge pilots.
    """
    iq = np.asarray(iq, dtype=complex)
    positions = np.asarray(positions, dtype=float)
    if iq.ndim != 1 or positions.ndim != 1 or not len(iq):
        raise ValueError("Need nonempty one-dimensional IQ and sample positions")
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite resampling positions")
    if np.array_equal(positions, np.arange(len(iq))):
        return iq.copy()
    guard = 256
    length = next_fast_len(len(iq) + 2 * guard)
    padded = np.zeros(length, dtype=complex)
    padded[guard:guard + len(iq)] = iq
    high_rate = resample(padded, int(oversample * length))
    coordinates = ((positions + guard) * oversample)[None, :]
    output = map_coordinates(high_rate, coordinates, order=3, mode="constant", cval=0)
    output[(positions < -guard) | (positions >= len(iq) + guard)] = 0
    return output
