import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize_scalar
from src.bandlimited import sample_at
from src.waveform import demodulate

def find_frame(rx, training, nfft, cp, symbols):
    if len(rx) < symbols*(nfft+cp):
        raise ValueError("Capture is shorter than the configured frame")
    template = np.fft.ifft(training, norm="ortho")
    block = nfft + cp #One OFDM symbol, including the cyclic prefix
    #Cross-correlation
    correlation = correlate( rx, template, mode="valid", method="fft" )
    energy = np.convolve( np.abs(rx)**2, np.ones(nfft), mode="valid" )
    floor = max(float(energy.max()) * 1e-8, 1e-300) #Guard against division by zero
    score = np.abs(correlation)**2 / (
        np.maximum(energy, floor) * np.sum(np.abs(template)**2)
    )

    starts = np.arange(len(rx) - symbols * block + 1)
    useful = starts + cp
    paired_score = np.sqrt( score[useful] * score[useful + block] )
    best = int(np.argmax(paired_score))
    return int(starts[best]), float(paired_score[best])

def estimate_pilot_csi(received_grid, pilot_mask, pilot_values):
    if received_grid.shape != pilot_mask.shape:
        raise ValueError("Pilot mask shape does not match received grid")
    if pilot_values.shape != received_grid.shape:
        raise ValueError("Pilot values shape does not match received grid")
    if np.any(np.abs(pilot_values[pilot_mask]) == 0):
        raise ValueError("Pilot values must be nonzero")
    csi = np.full(
        received_grid.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    ) #Initialise the estimator
    csi[pilot_mask] = ( received_grid[pilot_mask] / pilot_values[pilot_mask] )

    return csi

def correct_cfo(rx, fs, nfft, cp):
    block = nfft + cp
    first = rx[cp:block]
    second = rx[block + cp:2 * block]
    phase = np.angle(np.vdot(first, second))
    cfo_hz = phase * fs / (2 * np.pi * block)
    t = np.arange(len(rx)) / fs
    corrected = rx * np.exp(-2j * np.pi * cfo_hz * t)

    return corrected, cfo_hz

def correct_orbit(rx, fs, doppler_hz, time_scale):
    if not np.isfinite([fs, doppler_hz, time_scale]).all() or min(fs, time_scale) <= 0:
        raise ValueError("Invalid orbit synchronization parameters")
    t = np.arange(len(rx))/fs
    shifted = rx*np.exp(-2j*np.pi*doppler_hz*t)
    count = int(np.floor((len(rx)-1)*time_scale))+1
    return sample_at(shifted, np.arange(count)/time_scale)


def estimate_sfo(pilot_csi, pilot_mask, nfft, cp, max_abs_ppm=50.):
    """Complex pilot fit, without frequency-axis phase unwrap or amplitude loss."""
    rows = []
    frequencies = []
    offsets = []
    frequency = np.fft.fftfreq(nfft)
    reference = pilot_csi[1]
    for m in range(2, len(pilot_csi)):
        indices = np.flatnonzero(pilot_mask[m] & pilot_mask[1] & np.isfinite(reference) & np.isfinite(pilot_csi[m]))
        if len(indices) < 4:
            continue
        products = pilot_csi[m,indices]*np.conj(reference[indices])
        strength = np.abs(products)
        keep = strength > .05*np.median(strength)
        if keep.sum() < 4 or np.ptp(frequency[indices[keep]]) < .1:
            continue
        products = products[keep]
        rows.append(products / max(float(np.linalg.norm(products)), 1e-300))
        frequencies.append(frequency[indices[keep]])
        offsets.append((m-1)*(nfft+cp))
    if len(rows) < 3:
        raise ValueError("Insufficient pilots for identifiable sample-clock estimation")
    def objective(ppm):
        return -float(np.mean([abs(np.sum(z*np.exp(2j*np.pi*f*offset*ppm*1e-6)))**2/len(z)
                               for z,f,offset in zip(rows,frequencies,offsets)]))
    grid = np.linspace(-max_abs_ppm,max_abs_ppm,201)
    best = int(np.argmin([objective(ppm) for ppm in grid]))
    if best in (0,len(grid)-1):
        raise ValueError("Sampling offset lies outside the supported search range")
    fit = minimize_scalar(objective, bounds=(grid[best-1],grid[best+1]), method="bounded")
    if not fit.success or -fit.fun < .05:
        raise ValueError("Sampling-clock fit has insufficient pilot coherence")
    ppm = float(fit.x)
    return ppm, np.asarray(offsets)*ppm*1e-6


def correct_sfo(rx, fs, estimated_ppm):
    time_scale = 1.0 / (1.0 + estimated_ppm * 1e-6)
    return correct_orbit(
        rx,
        fs,
        doppler_hz=0.0,
        time_scale=time_scale,
    )


def estimate_csi_noise_power(
    rx, fs, frame_start, doppler_hz, time_scale, cfo_hz,
    sfo_ppm, nfft, cp, symbols, pilot_mask, pilot_values,
    trials=32, seed=92831,
):
    """Propagate estimated white noise through fixed synchronization operators."""
    # Map the detected start back to raw IQ, leaving a guard for interpolation.
    stop = min(len(rx), int(frame_start / time_scale) - 128)
    noise_segment = rx[64:stop]
    if stop <= 64 or len(noise_segment) < 128:
        raise ValueError("Not enough pre-frame noise samples")
    noise_power = float(np.mean(np.abs(noise_segment) ** 2))
    rng = np.random.default_rng(seed)
    accumulated = np.zeros(pilot_mask.shape, dtype=float)

    for _ in range(trials):
        noise = np.sqrt(noise_power / 2) * (
            rng.normal(size=len(rx)) + 1j * rng.normal(size=len(rx))
        )
        corrected = correct_orbit(noise, fs, doppler_hz, time_scale)
        corrected = corrected[frame_start:]
        t = np.arange(len(corrected)) / fs
        corrected *= np.exp(-2j * np.pi * cfo_hz * t)
        corrected = correct_sfo(corrected, fs, sfo_ppm)
        grid = demodulate(corrected, nfft, cp, symbols)
        accumulated[pilot_mask] += np.abs(
            grid[pilot_mask] / pilot_values[pilot_mask]
        ) ** 2

    csi_noise_power = np.full(pilot_mask.shape, np.nan)
    csi_noise_power[pilot_mask] = accumulated[pilot_mask] / trials
    return noise_power, len(noise_segment), csi_noise_power
