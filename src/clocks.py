"""Motion and receiver impairments shared by the generic OFDM profile."""
import numpy as np
from src.bandlimited import sample_at
from src.receiver_noise import noise_budget


def receive_chain_phase(settings):
    """Fixed hardware phase for the implemented single receive RF chain."""
    phases = np.asarray(settings.get('rf_chain_phase_offsets_rad', [0.]), float)
    if phases.shape != (1,) or not np.isfinite(phases).all():
        raise ValueError('This single-channel simulator requires one fixed RF-chain phase')
    return float(phases[0])


def held_receiver_gain(noisy, settings):
    """One voltage gain over a whole frame, not independent symbol gains."""
    mode = settings['agc']
    power = float(np.mean(abs(noisy)**2))
    if mode == 'logged':
        return 1/np.sqrt(power) if power > 0 else 1.
    if mode == 'off':
        return 1.
    if mode == 'fixed':
        gain = float(settings['fixed_gain_linear'])
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError('Fixed receiver voltage gain must be positive')
        return gain
    raise ValueError('receiver.agc must be logged, fixed or off')


def resample_at(iq, positions):
    return sample_at(iq, positions)


def apply_motion(iq, fs, carrier_hz, doppler_hz, initial_delay_samples=0.):
    iq = np.asarray(iq, dtype=complex)
    if not np.isfinite([fs, carrier_hz, doppler_hz, initial_delay_samples]).all() or min(fs, carrier_hz) <= 0:
        raise ValueError("Invalid motion parameters")
    n = np.arange(len(iq))
    ratio = 1 + doppler_hz/carrier_hz
    if ratio <= 0:
        raise ValueError("Invalid motion time scale")
    positions = n*ratio + initial_delay_samples*(1-ratio)
    return sample_at(iq, positions) * np.exp(2j*np.pi*doppler_hz*n/fs)


def receiver(iq, fs, clock, receiver_config, seed, noise_seed=None):
    rng = np.random.default_rng(seed)
    def draw(name):
        bounds = np.asarray(clock[name], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid clock setting: {name}")
        return float(rng.uniform(*bounds))
    lo, ppm, timing, linewidth = [draw(k) for k in
        ("lo_offset_hz", "sfo_ppm", "timing_offset_samples", "phase_noise_linewidth_hz")]
    if not np.isfinite(fs) or fs <= 0 or linewidth < 0 or 1+ppm*1e-6 <= 0:
        raise ValueError("Invalid clock or sample rate")
    fs_rx = fs*(1+ppm*1e-6)
    t = np.arange(len(iq))/fs_rx
    signal = resample_at(iq, np.arange(len(iq))/(1+ppm*1e-6)-timing)
    phase = np.cumsum(rng.normal(0, np.sqrt(2*np.pi*linewidth/fs_rx), len(iq)))
    po = receive_chain_phase(receiver_config)
    signal *= np.exp(1j*(-2*np.pi*lo*t+phase-po))
    budget = noise_budget(receiver_config, fs)
    noise_power = budget["noise_power_w"]
    noise_rng = np.random.default_rng(noise_seed if noise_seed is not None else np.random.SeedSequence([int(seed), 91]))
    noisy = signal + np.sqrt(noise_power/2)*(noise_rng.normal(size=len(iq))+1j*noise_rng.normal(size=len(iq)))
    gain = held_receiver_gain(noisy, receiver_config)
    snr = 10*np.log10(np.mean(np.abs(signal)**2)/noise_power) if noise_power > 0 else np.inf
    return noisy*gain, dict(lo_offset_hz=lo, sfo_ppm=ppm, timing_offset_samples=timing,
        phase_noise_linewidth_hz=linewidth, **budget, agc_gain_linear=gain,
        input_snr_db=float(snr), rf_chain_phase_offset_rad=po,
        po_time_model='fixed_RF_chain_parameter_not_frame_or_symbol_random',
        agc_time_model='one_held_voltage_gain_per_frame')
