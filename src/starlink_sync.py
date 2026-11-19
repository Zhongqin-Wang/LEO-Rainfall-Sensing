"""Orbit-aided acquisition and pilot-only CSI for the public Starlink profile."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import correlate, find_peaks
from scipy.fft import fft,ifft,next_fast_len

from src.bandlimited import sample_at
from src.starlink_waveform import (FS, NFFT, CP, SYMBOLS, BLOCK, FRAME_SAMPLES,
                                   FRAME_GUARD_SAMPLES,EDGE_PILOTS, acquisition_template)
from src.waveform import demodulate
from src.sync import estimate_pilot_csi
from src.pilot_power import coherent_pilot_power


def acquisition_threshold(trials, template_samples, false_alarm_probability=1e-6):
    """Conservative union bound for normalized correlation in complex white noise.

    Under that null a single squared correlation is Beta(1,N-1). Interference
    and coloured front ends need separate calibration; this is not a field FAR.
    """
    if trials < 1 or template_samples < 2 or not 0 < false_alarm_probability < 1:
        raise ValueError("Invalid acquisition detector configuration")
    return float(-np.expm1(np.log(false_alarm_probability / trials) / (template_samples - 1)))


def correct_motion(rx, doppler_hz, time_scale):
    if not np.isfinite([doppler_hz, time_scale]).all() or time_scale <= 0:
        raise ValueError("Invalid orbit correction")
    n = np.arange(len(rx))
    shifted = rx * np.exp(-2j * np.pi * doppler_hz * n / FS)
    count = int(np.floor((len(rx) - 1) * time_scale)) + 1
    return sample_at(shifted, np.arange(count) / time_scale)


def acquire(rx, prefer_earliest=False):
    template = acquisition_template()
    count = len(rx) - FRAME_SAMPLES + 1
    if count <= 0:
        raise ValueError("Capture is shorter than one complete Starlink frame")
    correlation = correlate(rx, template, mode="valid", method="fft")[:count]
    energy = np.r_[0, np.cumsum(np.abs(rx) ** 2)]
    length = len(template)
    local_energy = energy[length:length + count] - energy[:count]
    score = np.abs(correlation) ** 2 / np.maximum(local_energy * np.vdot(template, template).real, 1e-300)
    # A truly empty slot has no signal to normalize. FFT roundoff divided by
    # zero must not turn silence into the strongest acquisition candidate.
    score[local_energy <= max(float(np.max(local_energy))*1e-12,np.finfo(float).tiny)] = 0.
    index = int(np.argmax(score))
    threshold = acquisition_threshold(count, len(template))
    if prefer_earliest:
        height = max(threshold,.5*float(score[index]))
        candidates = list(find_peaks(score,height=height,distance=FRAME_SAMPLES//2)[0])
        if score[0]>=height and (len(score)==1 or score[0]>=score[1]):
            candidates.append(0)
        if candidates:
            index = int(min(candidates))
    if not np.isfinite(score[index]) or score[index] < threshold:
        raise ValueError("Starlink PSS+SSS frame acquisition failed")
    return index, float(score[index])


def pss_carrier_offset(frame):
    pieces = frame[CP:CP + NFFT].reshape(8, 128).copy()
    pieces[0] *= -1
    correlation = np.sum(pieces[1:] * np.conj(pieces[:-1]))
    return float(np.angle(correlation) * FS / (2 * np.pi * 128))


def estimate_clock_offsets(grid, known):
    """Estimate carrier and sampling offsets separately from 4800 edge pilots.

    Each pilot has an unknown complex channel gain, eliminated by coherent
    profile likelihood. No simulated channel, clock truth, or unknown payload
    is used. Temporal FFT initialization avoids fragile phase unwrapping over
    the large frequency gap between the two pilot bands.
    """
    csi = grid[2:, EDGE_PILOTS] / known[2:, EDGE_PILOTS]
    normalization = np.sqrt(np.mean(np.abs(csi) ** 2))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("No finite Starlink edge-pilot power")
    csi = csi / normalization
    times = (np.arange(2, SYMBOLS) * BLOCK + CP + (NFFT - 1) / 2) / FS
    times -= times[0]
    frequency = np.fft.fftfreq(NFFT, 1 / FS)[EDGE_PILOTS]
    window = np.hanning(len(times))
    weighted = csi * window[:, None]
    fft_size = 8192
    temporal_frequency = np.fft.fftfreq(fft_size, BLOCK / FS)
    transform = np.fft.fft(weighted, n=fft_size, axis=0)
    estimates = []
    spacing = FS / BLOCK / fft_size
    for band in (slice(0, 8), slice(8, 16)):
        powers = np.sum(np.abs(transform[:, band]) ** 2, axis=1)
        initial = temporal_frequency[int(np.argmax(powers))]
        def cost(hz):
            combined = np.sum(weighted[:, band] * np.exp(-2j * np.pi * hz * times[:, None]), axis=0)
            return -float(np.sum(np.abs(combined) ** 2))
        fit = minimize_scalar(cost, bounds=(initial - spacing, initial + spacing), method="bounded")
        estimates.append(float(fit.x))
    positive, negative = np.mean(frequency[:8]), np.mean(frequency[8:])
    sfo = -(estimates[0] - estimates[1]) / (positive - negative) * 1e6
    carrier = estimates[0] + positive * sfo * 1e-6
    if abs(sfo) > 100 or abs(carrier) > 50000:
        raise ValueError("Starlink clock acquisition is outside the supported search range")
    scale = window.sum() ** 2 * len(EDGE_PILOTS)
    def objective(parameters):
        carrier_hz, ppm = parameters[0] * 1000, parameters[1]
        rates = carrier_hz - frequency * ppm * 1e-6
        aligned = weighted * np.exp(-2j * np.pi * times[:, None] * rates[None, :])
        return -float(np.sum(np.abs(np.sum(aligned, axis=0)) ** 2) / scale)
    fit = minimize(objective, [carrier / 1000, sfo], method="L-BFGS-B",
                   bounds=[(-50, 50), (-100, 100)], options={"ftol": 1e-12, "gtol": 1e-8})
    if not fit.success:
        raise ValueError(f"Starlink clock fit failed: {fit.message}")
    carrier, sfo = float(fit.x[0] * 1000), float(fit.x[1])
    coherence = -objective(fit.x)
    # Receiver design threshold, not a Starlink specification. The white-noise
    # maximum-coherence approximation and null tests motivate this guard;
    # unknown interference requires a separate detection calibration.
    if abs(sfo) > 50 or coherence < 0.02:
        raise ValueError(f"Unreliable Starlink synchronization: SFO={sfo:.2f} ppm, coherence={coherence:.3f}")
    return carrier, sfo, coherence


def search_coarse_cfo(rx,limit_hz=400000.,step_hz=25000.):
    """Blind PSS+SSS search for one complete frame, without orbit/clock truth.

    The finite search range/spacing are receiver design choices. Capture center
    tuning and LNB error must place the signal within this range.
    """
    if not np.isfinite([limit_hz,step_hz]).all() or min(limit_hz,step_hz)<=0 or step_hz>limit_hz:
        raise ValueError("Invalid coarse carrier search range")
    rx=np.asarray(rx)
    count=len(rx)-FRAME_SAMPLES+1
    if count<=0:
        raise ValueError("Need a complete frame for coarse carrier search")
    template=acquisition_template()
    nfft=next_fast_len(len(rx)+len(template)-1)
    received=fft(rx,nfft)
    energy=np.r_[0.,np.cumsum(abs(rx)**2)]
    denominator=np.maximum((energy[len(template):len(template)+count]-energy[:count])*np.vdot(template,template).real,1e-300)
    times=np.arange(len(template))/FS
    best=(-np.inf,0.,0)
    frequencies = np.arange(-limit_hz,limit_hz+step_hz*.5,step_hz)
    for frequency in frequencies:
        replica=template*np.exp(2j*np.pi*frequency*times)
        correlation=ifft(received*np.conj(fft(replica,nfft)))[:count]
        scores=abs(correlation)**2/denominator
        index=int(np.argmax(scores))
        if scores[index]>best[0]:
            best=(float(scores[index]),float(frequency),index)
    threshold = acquisition_threshold(count * len(frequencies), len(template))
    if not np.isfinite(best[0]) or best[0]<threshold:
        raise ValueError("No reliable full-frame PSS+SSS match in the carrier search range")
    return dict(coarse_cfo_hz=best[1],coarse_frame_start_samples=best[2],coarse_capture_score=best[0],
                coarse_capture_threshold=threshold)


def receive_frame(rx, known, predicted_doppler_hz, predicted_time_scale,noise_source="preframe", shared_sync=None, prefer_earliest=False):
    """Estimate effective CSI; retain unresolved global phase and front-end gain."""
    corrected = correct_motion(rx, predicted_doppler_hz, predicted_time_scale)
    if shared_sync is None:
        start, score = acquire(corrected,prefer_earliest=prefer_earliest)
    else:
        start, score = int(shared_sync['frame_start_samples']), float(shared_sync['capture_score'])
    frame = corrected[start:]
    mask = np.abs(known) > 0
    # Preserve the sensing observation before fitted CFO/SFO/CPE can absorb
    # physical rain phase. This is effective CSI and may contain residual ICI;
    # it is not a clock-free estimate of the physical propagation channel.
    orbit_only_csi = estimate_pilot_csi(demodulate(frame, NFFT, CP, SYMBOLS), mask, known)
    if shared_sync is None:
        coarse_cfo = pss_carrier_offset(frame)
        n = np.arange(len(frame))
        # PSS self-correlation becomes noisy before coherent edge-pilot tracking.
        # An acquired PSS+SSS match already constrains the residual CFO. Compare
        # the PSS initialisation against zero residual, using known pilots only.
        fits, failures = [], []
        for initial_cfo in (coarse_cfo, 0.):
            coarse_corrected = frame * np.exp(-2j * np.pi * initial_cfo * n / FS)
            initial_grid = demodulate(coarse_corrected, NFFT, CP, SYMBOLS)
            try:
                fine_cfo, ppm, quality = estimate_clock_offsets(initial_grid, known)
                fits.append((quality, initial_cfo + fine_cfo, ppm))
            except ValueError as error:
                failures.append(str(error))
        if not fits:
            raise ValueError("Starlink clock initialisations failed: " + "; ".join(failures))
        coherence, total_cfo, sfo = max(fits, key=lambda fit: fit[0])
    else:
        total_cfo = float(shared_sync['estimated_residual_cfo_hz'])
        sfo = float(shared_sync['estimated_residual_sfo_ppm'])
        coherence = float(shared_sync['synchronization_coherence'])
        n = np.arange(len(frame))
    carrier_corrected = frame * np.exp(-2j * np.pi * total_cfo * n / FS)
    # Positive residual SFO means samples arrive too slowly in nominal indices.
    positions = np.arange(FRAME_SAMPLES) * (1 + sfo * 1e-6)
    if positions[-1] >= len(carrier_corrected) - 1:
        raise ValueError("Insufficient post-frame guard for sampling-clock correction")
    corrected_frame = sample_at(carrier_corrected, positions)
    grid = demodulate(corrected_frame, NFFT, CP, SYMBOLS)
    initial_csi = estimate_pilot_csi(grid, mask, known)
    signal_power = np.full(NFFT, np.nan)
    signal_power[EDGE_PILOTS] = coherent_pilot_power(initial_csi, EDGE_PILOTS)
    # Only rotate phase: amplitude normalization here would erase the rain signal.
    reference = initial_csi[1, EDGE_PILOTS]
    relative = np.sum(initial_csi[2:, EDGE_PILOTS] * np.conj(reference)[None, :], axis=1)
    cpe = (np.r_[0., 0., np.angle(relative)] if shared_sync is None
           else np.asarray(shared_sync['cpe_rad']).copy())
    grid[2:] *= np.exp(-1j * cpe[2:, None])
    csi = estimate_pilot_csi(grid, mask, known)
    mapping_rate = (1 + sfo * 1e-6) / predicted_time_scale
    if noise_source=="preframe":
        # Existing isolated synthetic captures have a genuinely empty prefix.
        raw_stop = int(start / predicted_time_scale) - 256
        if raw_stop < 320:
            raise ValueError("Insufficient pre-frame noise for calibrated CSI power")
        noise_segment = rx[64:raw_stop]
        noise_power = float(np.mean(np.abs(noise_segment) ** 2))
        pilot_noise_power=noise_power/mapping_rate
        noise_model="guarded_preframe_white_noise_in_band_psd"
    elif noise_source=="frame_guard":
        # In real IQ, samples before a frame may contain earlier traffic. Only
        # the known empty guard is used, trimming resampling/filter transients.
        guard_start=FRAME_SAMPLES-FRAME_GUARD_SAMPLES
        noise_segment=corrected_frame[guard_start+128:FRAME_SAMPLES-128]
        noise_power=float(np.mean(abs(noise_segment)**2))
        pilot_noise_power=noise_power
        noise_model="trimmed_frame_guard_noise_plus_unmodeled_interference"
    else:
        raise ValueError("noise_source must be preframe or frame_guard")
    noise_csi = np.full(csi.shape, np.nan)
    # In-band PSD scales inversely with the resampling dilation; no amplitude
    # correction is inferred from the unknown transmitted payload.
    noise_csi[mask] = pilot_noise_power / np.abs(known[mask]) ** 2
    return dict(pilot_csi=csi, pilot_csi_orbit_only=orbit_only_csi,
                pilot_csi_before_cpe=initial_csi,
                phase_observation_note="orbit_only retains unresolved clocks/rain/ICI; before_cpe has fitted CFO/SFO removed; final has CPE removed",
                residual_cfo_interpretation="clock_offset_plus_orbit_prediction_error_plus_physical_phase_rate_not_clock_alone",
                pilot_mask=mask, csi_noise_power=noise_csi,
                signal_power_by_frequency=signal_power,
                signal_power_estimator="multi_lag_channel_autocorrelation_intercept",
                signal_power_max_lag_symbols=32,
                estimated_noise_power_w=noise_power, noise_sample_count=len(noise_segment),
                frame_start_samples=start, capture_score=score, capture_success=True,
                capture_threshold=acquisition_threshold(len(corrected)-FRAME_SAMPLES+1,len(acquisition_template())),
                estimated_residual_cfo_hz=total_cfo, estimated_residual_sfo_ppm=sfo,
                synchronization_coherence=coherence, cpe_rad=cpe,
                cpe_corrected=True, sfo_corrected=True, cfo_corrected=True, orbit_corrected=True,
                noise_model=noise_model)
