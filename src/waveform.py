"""Complex baseband OFDM, NumPy FFT order, unitary FFT normalization."""
import numpy as np


def modulate(grid, cp):
    nfft = grid.shape[-1]
    if not 0 <= cp < nfft:
        raise ValueError("CP must lie in [0, nfft)")
    time = np.fft.ifft(grid, axis=-1, norm="ortho")
    return (np.concatenate([time[:, -cp:], time], axis=1) if cp else time).ravel()
#IFFT

def demodulate(iq, nfft, cp, symbols):
    if len(iq) < symbols*(nfft+cp):
        raise ValueError("Not enough IQ samples")
    blocks = iq[:symbols*(nfft+cp)].reshape(symbols, nfft+cp)
    return np.fft.fft(blocks[:, cp:], axis=-1, norm="ortho")


def make_frame(nfft, cp, symbols, pilot_spacing, seed):
    if symbols < 3 or pilot_spacing < 1 or nfft < 8:
        raise ValueError(
            "Need two training symbols, one data symbol, "
            "and positive pilot spacing"
        )
    rng = np.random.default_rng(seed)
    grid = ( (2 * rng.integers(0, 2, (symbols, nfft)) - 1)
        + 1j * (2 * rng.integers(0, 2, (symbols, nfft)) - 1) ) / np.sqrt(2)
    training_rng = np.random.default_rng(2026)
    training = (
        (2 * training_rng.integers(0, 2, nfft) - 1)
        + 1j * (2 * training_rng.integers(0, 2, nfft) - 1)
    ) / np.sqrt(2)

    mask = np.zeros(grid.shape, dtype=bool)
    mask[:2, :] = True
    mask[2:, ::pilot_spacing] = True
    grid[mask] = 1.0 + 0.0j
    grid[0, :] = training
    grid[1, :] = training

    return grid, mask, modulate(grid, cp)
#Generate random QPSK symbols, the preamble and pilots