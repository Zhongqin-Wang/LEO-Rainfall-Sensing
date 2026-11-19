import argparse
import json
from pathlib import Path
import numpy as np
from src.sync import find_frame, estimate_pilot_csi, correct_cfo, correct_orbit, estimate_sfo,correct_sfo
from src.sync import estimate_csi_noise_power
from src.waveform import demodulate
import matplotlib.pyplot as plt

def process_capture(capture_dir, plot=True):
    capture_dir = Path(capture_dir)
    (capture_dir / "csi_pilots_corrected.npz").unlink(missing_ok=True)
    rx = np.asarray(
        np.load(capture_dir / "rx_iq.npy", allow_pickle=False),
        dtype=np.complex128,
    )

    if rx.ndim not in (1,2) or not len(rx) or not np.isfinite(rx).all():
        raise ValueError("Received IQ must be finite [sample] or [sample,antenna]")

    with np.load(
        capture_dir / "frame_config.npz",
        allow_pickle=False,
    ) as metadata:
        fs = float(metadata["sample_rate_hz"].item())
        nfft = int(metadata["nfft"].item())
        cp = int(metadata["cp_samples"].item())
        symbols = int(metadata["symbols"].item())
        pilot_mask = metadata["pilot_mask"].astype(bool)
        pilot_values = metadata["pilot_values"].copy()
        profile = str(metadata["waveform_profile"]) if "waveform_profile" in metadata else "custom_ofdm"
        recorded_rate=float(metadata["capture_sample_rate_hz"]) if "capture_sample_rate_hz" in metadata else fs
    if profile == "starlink_public_v1":
        from src.starlink_waveform import FS,NFFT,CP,SYMBOLS,public_sequences
        if (fs,nfft,cp,symbols) != (FS,NFFT,CP,SYMBOLS) or pilot_values.shape != (SYMBOLS,NFFT):
            raise ValueError("Starlink frame metadata violates the public waveform profile")
        _,sss,edge,_ = public_sequences()
        expected = edge.copy();expected[1] = sss
        expected_mask = abs(expected)>0
        if not np.array_equal(pilot_mask,expected_mask) or not np.isfinite(pilot_values).all() or np.any(pilot_values[~expected_mask]!=0):
            raise ValueError("Starlink receiver references must contain only SSS and published edge pilots")
        scale = np.median(abs(pilot_values[expected_mask]/expected[expected_mask]))
        if scale <= 0 or not np.allclose(pilot_values,expected*scale,rtol=1e-10,atol=1e-15):
            raise ValueError("Starlink known-symbol values do not match the published sequences")
        from src.starlink_link import process_starlink_capture
        from src.sdr_capture import convert_rate
        if not np.isfinite(recorded_rate) or recorded_rate<FS:
            raise ValueError("This full-channel receiver does not support narrowband input")
        rx=convert_rate(rx,recorded_rate,FS)
        gain = json.loads((capture_dir / "receiver_log.json").read_text())["agc_gain_linear"]
        return process_starlink_capture(capture_dir, rx, pilot_values, gain, plot=plot)
    if profile != "custom_ofdm":
        raise ValueError(f"Unknown waveform profile: {profile}")
    if rx.ndim != 1:
        raise ValueError("Multi-channel IQ requires the public Starlink receiver")
    receiver_log = json.loads(
        (capture_dir / "receiver_log.json").read_text(encoding="utf-8")
    )
    agc_gain = float(receiver_log["agc_gain_linear"])
    if not np.isfinite(agc_gain) or agc_gain <= 0:
        raise ValueError("AGC gain must be finite and positive")
    rx = rx / agc_gain
    with np.load(capture_dir / "orbit_prediction.npz") as prediction:
        predicted_doppler = float(prediction["predicted_doppler_hz"])
        predicted_scale = float(prediction["predicted_time_scale"])

    rx_orbit = correct_orbit( rx, fs, predicted_doppler, predicted_scale )
    frame_start, capture_score = find_frame( rx_orbit, pilot_values[0], nfft, cp, symbols )
    capture_success = capture_score > 0.08
    print(f"Frame start: {frame_start} samples")
    print(f"Capture score: {capture_score:.3f}")
    if not capture_success:
        raise ValueError("Frame capture failed")
    rx_frame = rx_orbit[frame_start:]
    rx_corrected, cfo_hz = correct_cfo( rx_frame, fs, nfft, cp )
    initial_grid = demodulate(rx_corrected, nfft, cp, symbols)
    print(f"Predicted Doppler: {predicted_doppler:.3f} Hz")
    print(f"Estimated residual CFO: {cfo_hz:.3f} Hz")
    initial_csi = estimate_pilot_csi(
        initial_grid,
        pilot_mask,
        pilot_values,
    )
    estimated_sfo_ppm = 0.
    for iteration in range(5):
        rx_sfo_corrected = correct_sfo(rx_corrected,fs,estimated_sfo_ppm)
        received_grid = demodulate(rx_sfo_corrected,nfft,cp,symbols)
        partial = estimate_pilot_csi(received_grid,pilot_mask,pilot_values)
        residual, _ = estimate_sfo(partial,pilot_mask,nfft,cp)
        estimated_sfo_ppm = ((1+estimated_sfo_ppm*1e-6)*(1+residual*1e-6)-1)*1e6
        if abs(estimated_sfo_ppm)>50:
            raise ValueError("Refined sampling offset exceeds supported clock range")
        if abs(residual)<.02:
            break
    timing_drift_samples = np.arange(1,symbols-1)*(nfft+cp)*estimated_sfo_ppm*1e-6
    print(f"Estimated residual SFO: {estimated_sfo_ppm:.3f} ppm")
    rx_sfo_corrected = correct_sfo(rx_corrected,fs,estimated_sfo_ppm)
    received_grid = demodulate(rx_sfo_corrected,nfft,cp,symbols)
    pilot_csi = estimate_pilot_csi(
        received_grid,
        pilot_mask,
        pilot_values,
    )
    common_phase = np.zeros(symbols)
    for m in range(2,symbols):
        indices = pilot_mask[m] & pilot_mask[1]
        common_phase[m] = np.angle(np.sum(pilot_csi[m,indices]*np.conj(pilot_csi[1,indices])))
        received_grid[m] *= np.exp(-1j*common_phase[m])
    pilot_csi = estimate_pilot_csi(received_grid,pilot_mask,pilot_values)
    noise_trials = 32
    noise_seed = 92831
    noise_power_w, noise_samples, csi_noise_power = estimate_csi_noise_power(
        rx, fs, frame_start, predicted_doppler, predicted_scale,
        cfo_hz, estimated_sfo_ppm, nfft, cp, symbols,
        pilot_mask, pilot_values, trials=noise_trials, seed=noise_seed,
    )
    print(f"Estimated input noise: {noise_power_w:.3e} W ({noise_samples} samples)")

    frequency_offset_hz = np.fft.fftfreq(nfft, d=1.0 / fs)

    symbol_center_s = (
        np.arange(symbols) * (nfft + cp)
        + cp
        + (nfft - 1) / 2
    ) / fs

    np.savez_compressed(
        capture_dir / "csi_pilots_corrected.npz",
        pilot_csi=pilot_csi,
        csi_noise_power=csi_noise_power,
        estimated_noise_power_w=noise_power_w,
        noise_sample_count=noise_samples,
        noise_mc_trials=noise_trials,
        noise_mc_seed=noise_seed,
        pilot_mask=pilot_mask,
        frequency_offset_hz=frequency_offset_hz,
        symbol_center_s=symbol_center_s,
        sample_rate_hz=fs,
        nfft=nfft,
        cp_samples=cp,
        agc_removed=True,
        agc_gain_used=agc_gain,
        predicted_doppler_hz=predicted_doppler,
        predicted_time_scale=predicted_scale,
        estimated_residual_cfo_hz=cfo_hz,
        orbit_corrected=True,
        cfo_corrected=True,
        sfo_corrected=True,
        cpe_corrected=True,
        cpe_rad=common_phase,
        frame_start_samples=frame_start,
        capture_score=capture_score,
        capture_success=capture_success,
        estimated_residual_sfo_ppm=estimated_sfo_ppm,
        timing_drift_samples=timing_drift_samples,
    )

    print(f"CSI shape: {pilot_csi.shape}")
    print(f"Pilot estimates: {np.count_nonzero(pilot_mask)}")
    print("Applied orbit, frame timing, coarse CFO and SFO correction")
    print(f"Saved: {capture_dir / 'csi_pilots_corrected.npz'}")

    result = {
        "capture_score": float(capture_score),
        "estimated_residual_cfo_hz": float(cfo_hz),
        "estimated_residual_sfo_ppm": float(estimated_sfo_ppm),
    }
    if not plot:
        return result

    indices = np.flatnonzero(np.all(pilot_mask[2:], axis=0))
    indices = indices[np.argsort(frequency_offset_hz[indices])]
    frequency_mhz = frequency_offset_hz[indices] / 1e6
    data_csi = pilot_csi[2:, indices]
    magnitude_db = 20 * np.log10(np.maximum(np.abs(data_csi), 1e-30))
    phase_rad = np.angle(data_csi)
    data_symbols = np.arange(1, len(data_csi) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    selected = np.unique([0, len(data_csi) // 2, len(data_csi) - 1])
    for m in selected:
        label = f"Data symbol {m + 1}"
        axes[0, 0].plot(frequency_mhz, magnitude_db[m], label=label)
        axes[1, 0].plot(frequency_mhz, phase_rad[m], ".", label=label)

    axes[0, 0].set_ylabel("CSI magnitude (dB)")
    axes[1, 0].set_ylabel("CSI phase (rad)")
    axes[0, 0].set_title("Selected data symbols")
    for ax in axes[:, 0]:
        ax.legend()
        ax.grid(alpha=0.3)

    magnitude_map = axes[0, 1].pcolormesh(
        frequency_mhz, data_symbols, magnitude_db,
        shading="nearest", cmap="viridis",
    )
    phase_map = axes[1, 1].pcolormesh(
        frequency_mhz, data_symbols, phase_rad,
        shading="nearest", cmap="twilight", vmin=-np.pi, vmax=np.pi,
    )
    fig.colorbar(magnitude_map, ax=axes[0, 1], label="CSI magnitude (dB)")
    fig.colorbar(phase_map, ax=axes[1, 1], label="CSI phase (rad)")
    axes[0, 1].set_title("All data symbols (preamble excluded)")
    for ax in axes[:, 1]:
        ax.set_ylabel("Data symbol number")
    for ax in axes[1, :]:
        ax.set_xlabel("Baseband frequency (MHz)")

    fig.suptitle("Pilot CSI after CFO and SFO correction")
    fig.tight_layout()
    fig.savefig(capture_dir / "csi.png", dpi=150)
    plt.show()
    plt.close(fig)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    process_capture(args.capture)


if __name__ == "__main__":
    main()
