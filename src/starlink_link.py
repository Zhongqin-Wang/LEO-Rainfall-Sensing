"""Synthetic Starlink public-structure IQ with a separate receiver and truth log."""
import json
import hashlib
from pathlib import Path

import numpy as np

from src.bandlimited import sample_at
from src.starlink_waveform import (FS, NFFT, CP, SYMBOLS, BLOCK, PROFILE,
                                   make_starlink_frame, validate_config)
from src.starlink_sync import receive_frame
from src.orbit_prediction import predict_orbit
from src.reflection import compute_ground_reflection, channel_response, validate_polarization
from src.weather import retrieval_config
from src.sdr_capture import capture_rate,convert_rate
from src.rain_spectrum import row_rain_spectrum, row_rain_phase, apply_spectral_loss
from src.receiver_noise import noise_budget, case_receiver_settings
from src.paper_channel import prepare_frame_paths, additional_response
from src.antenna import path_antenna_gains
from src.clocks import receive_chain_phase, held_receiver_gain

C = 299792458.0


def draw_clock(config, seed):
    rng = np.random.default_rng(seed)
    values = {}
    for key in ("lo_offset_hz", "sfo_ppm", "timing_offset_samples", "phase_noise_linewidth_hz"):
        bounds = np.asarray(config[key], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid clock scenario: {key}")
        values[key] = float(rng.uniform(*bounds))
    if values["phase_noise_linewidth_hz"] < 0 or 1 + values["sfo_ppm"] * 1e-6 <= 0:
        raise ValueError("Clock linewidth and sample rate must be nonnegative / positive")
    return values


def simulate_capture(tx, loss_db, carrier_hz, doppler_hz, doppler_rate_hz_s,
                     tx_clock, rx_clock, receiver_config, phase_seed, noise_seed,
                     prefix=1536, suffix=2048, reflection=None, spectral_loss_db=None,
                     spectral_phase_rad=None, rain_phase_rate_rad_s=0., additional_paths=None,
                     log_amplitude_rate_per_s=0., direct_relative_delay_s=0.):
    """Compose propagation and two independent clocks into one interpolation.

    Carrier clock and sample clock offsets are separate. Motion alters both
    carrier and modulation time scale. Each capture is local to a frame, without
    claiming modeled absolute TOA or inter-frame carrier phase coherence.
    """
    if not np.isfinite([loss_db, carrier_hz, doppler_hz, doppler_rate_hz_s]).all() or carrier_hz <= 0:
        raise ValueError("Invalid propagation parameters")
    additional_paths = additional_paths or []
    for path in additional_paths:
        values = [path['delay_samples'], path['doppler_difference_hz'],
                  path['relative_coefficient_real'], path['relative_coefficient_imag']]
        if not np.isfinite(values).all() or not 0 <= values[0] <= FS*.001:
            raise ValueError("Additional paths require finite parameters and delays in [0,1 ms]")
    if additional_paths:
        suffix += int(np.ceil(max(path['delay_samples'] for path in additional_paths)))
    unfiltered = np.pad(tx, (prefix, suffix))
    padded = unfiltered
    if not np.isfinite([rain_phase_rate_rad_s, log_amplitude_rate_per_s]).all():
        raise ValueError("Rain phase and path-amplitude rates must be finite")
    if spectral_loss_db is not None or spectral_phase_rad is not None:
        loss = np.zeros(NFFT) if spectral_loss_db is None else spectral_loss_db
        padded = apply_spectral_loss(padded, FS, np.fft.fftfreq(NFFT, 1 / FS), loss,
                                     phase_rad=spectral_phase_rad)
    n = np.arange(len(padded))
    rx_ratio = 1 + rx_clock["sfo_ppm"] * 1e-6
    tx_ratio = 1 + tx_clock["sfo_ppm"] * 1e-6
    rx_time = (n - prefix) / (FS * rx_ratio)
    path_time = rx_time - rx_clock["timing_offset_samples"] / FS
    ratio = 1 + doppler_hz / carrier_hz
    retarded_time = ratio * path_time + 0.5 * doppler_rate_hz_s / carrier_hz * path_time ** 2
    if not np.isfinite(direct_relative_delay_s):
        raise ValueError('Array differential propagation delay must be finite')
    positions = (prefix + FS * tx_ratio * (retarded_time-direct_relative_delay_s)
                 - tx_clock["timing_offset_samples"])
    signal = sample_at(padded, positions) * 10 ** (-loss_db / 20)
    phase_rng = np.random.default_rng(phase_seed)
    # Draw before the variable-length Wiener sequence: adding a delayed path
    # must not change the oscillator's initial phase merely by extending guard.
    initial_phase = phase_rng.uniform(-np.pi, np.pi)
    linewidth = tx_clock["phase_noise_linewidth_hz"] + rx_clock["phase_noise_linewidth_hz"]
    phase_noise = np.cumsum(phase_rng.normal(0, np.sqrt(2 * np.pi * linewidth / (FS * rx_ratio)), len(n)))
    phase = 2 * np.pi * (doppler_hz * path_time + 0.5 * doppler_rate_hz_s * path_time ** 2
                        + tx_clock["lo_offset_hz"] * retarded_time
                        - rx_clock["lo_offset_hz"] * rx_time)
    phase += rain_phase_rate_rad_s*path_time
    # This is geometric delay between RX phase centres, not a new RX TO.
    phase -= 2*np.pi*(carrier_hz+tx_clock['lo_offset_hz'])*direct_relative_delay_s
    signal *= np.exp(1j * (phase + phase_noise + initial_phase)+log_amplitude_rate_per_s*path_time)
    if reflection is not None:
        delay = float(reflection["delay_samples"])
        if not 0 <= delay <= CP:
            raise ValueError("Reflection delay exceeds the fixed Starlink CP")
        extra_doppler = float(reflection.get("doppler_difference_hz", 0.))
        delta_ratio = extra_doppler/carrier_hz
        reflected_positions = positions - delay*tx_ratio + FS*tx_ratio*delta_ratio*path_time
        coefficient = 10**(reflection["gain_db"]/20)*np.exp(1j*np.deg2rad(reflection["phase_deg"]))
        reflected_signal = sample_at(padded,reflected_positions)*10**(-loss_db/20)*coefficient
        reflected_phase = phase + 2*np.pi*(extra_doppler+tx_clock["lo_offset_hz"]*delta_ratio)*path_time
        signal += reflected_signal*np.exp(1j*(reflected_phase+phase_noise+initial_phase))
    for path in additional_paths:
        delay = float(path['delay_samples'])
        extra_doppler = float(path['doppler_difference_hz'])
        delta_ratio = extra_doppler/carrier_hz
        path_positions = positions-delay*tx_ratio+FS*tx_ratio*delta_ratio*path_time
        coefficient = complex(path['relative_coefficient_real'], path['relative_coefficient_imag'])
        # TX oscillator is sampled at the additional retarded time; common
        # phase noise across short excess delays is the same approximation as
        # in the local CSI factorization, not exact independent ray oscillators.
        coefficient *= np.exp(-2j*np.pi*tx_clock['lo_offset_hz']*delay/FS)
        extra_phase = 2*np.pi*(extra_doppler+tx_clock['lo_offset_hz']*delta_ratio)*path_time
        path_signal = padded
        if 'rain_frequency_offset_hz' in path:
            grid = np.fft.fftfreq(NFFT, 1/FS)
            f = np.asarray(path['rain_frequency_offset_hz'], float)
            extra_loss = np.interp(grid, f, path['rain_delta_loss_db'])
            extra_rain_phase = np.interp(grid, f, path['rain_delta_phase_rad'])
            base_loss = np.zeros(NFFT) if spectral_loss_db is None else np.asarray(spectral_loss_db)
            base_phase = np.zeros(NFFT) if spectral_phase_rad is None else np.asarray(spectral_phase_rad)
            path_signal = apply_spectral_loss(unfiltered, FS, grid, base_loss+extra_loss,
                                              phase_rad=base_phase+extra_rain_phase)
        extra_phase += float(path.get('rain_phase_rate_difference_rad_s', 0.))*path_time
        amplitude_rate = log_amplitude_rate_per_s+float(path.get('log_amplitude_rate_difference_per_s', 0.))
        signal += (sample_at(path_signal, path_positions)*10**(-loss_db/20)*coefficient
                   *np.exp(1j*(phase+extra_phase+phase_noise+initial_phase)+amplitude_rate*path_time))
    po = receive_chain_phase(receiver_config)
    signal *= np.exp(-1j*po)
    budget = noise_budget(receiver_config, FS)
    noise_power = budget["noise_power_w"]
    rng = np.random.default_rng(noise_seed)
    noise = np.sqrt(noise_power / 2) * (rng.normal(size=len(n)) + 1j * rng.normal(size=len(n)))
    noisy = signal + noise
    gain = held_receiver_gain(noisy, receiver_config)
    return noisy * gain, dict(agc_gain_linear=gain, **budget,
                             tx_clock=tx_clock, rx_clock=rx_clock, initial_phase_rad=initial_phase,
                             effective_nominal_cfo_hz=(doppler_hz + tx_clock["lo_offset_hz"] * ratio
                                                       - rx_clock["lo_offset_hz"]) / rx_ratio,
                             lo_sign_convention="positive_TX_LO_raises_baseband_frequency_positive_RX_LO_lowers_it",
                             effective_modulation_scale=ratio * tx_ratio / rx_ratio,
                             prefix_samples=prefix, doppler_hz=doppler_hz,
                             doppler_rate_hz_s=doppler_rate_hz_s,
                             rf_chain_phase_offset_rad=po,
                             po_time_model='fixed_RF_chain_parameter_not_frame_or_symbol_random',
                             agc_time_model='one_held_voltage_gain_per_frame',
                             oscillator_CFO_hz=tx_clock['lo_offset_hz']-rx_clock['lo_offset_hz'],
                             log_amplitude_rate_per_s=log_amplitude_rate_per_s,
                             direct_relative_delay_s=direct_relative_delay_s)


def generate_starlink_pair(row, before_row, after_row, config, prediction_batch,
                           out, relative_dir, frame_seed, _array_context=None):
    from src.receive_array import channel_count, array_layout, generate_array_pair
    array_layout(config)
    if channel_count(config) > 1:
        return generate_array_pair(row,before_row,after_row,config,prediction_batch,out,relative_dir,frame_seed)
    validate_config(config)
    validate_polarization(config)
    recorded_rate=capture_rate(config)
    index = int(row["satellite_index"])
    carrier = float(config["link"]["carrier_hz"])
    tx, mask, known = make_starlink_frame(frame_seed, config["ofdm"].get("payload_mode", "template_qpsk"))
    tx_power = float(config["ofdm"]["tx_power_w"])
    if not np.isfinite(tx_power) or tx_power <= 0:
        raise ValueError("TX power must be finite and positive")
    # Fixed nominal active-symbol power calibration, not unknown-payload normalization.
    scale = np.sqrt(tx_power * NFFT / (NFFT - 4))
    tx *= scale
    known *= scale
    seeds = np.random.SeedSequence([int(config["experiment"]["seed"]), index, 20261018]).generate_state(2)
    tx_clock = draw_clock(config["tx_clock"], int(seeds[0]))
    rx_clock = draw_clock(config["clock"], int(seeds[1]))
    motion = config["ofdm"].get("motion_enabled", True)
    reflection = None
    paper_enabled = config['propagation'].get('paper_channel', {}).get('enabled', False)
    if config["propagation"]["enabled"].get("ground_reflection",False) and not paper_enabled:
        from run_link import reflection_geometry_from_row
        reflection = compute_ground_reflection(reflection_geometry_from_row(row),config,FS)
        delta_rate = 0.
        if "reflected_range_rate_m_s" in reflection:
            delta_rate = reflection["reflected_range_rate_m_s"]-reflection["direct_range_rate_m_s"]
        else:
            span = (after_row.time_utc-before_row.time_utc).total_seconds()
            if span > 0:
                a = compute_ground_reflection(reflection_geometry_from_row(before_row),config,FS)
                b = compute_ground_reflection(reflection_geometry_from_row(after_row),config,FS)
                delta_rate = (b["excess_path_m"]-a["excess_path_m"])/span
        reflection["doppler_difference_hz"] = -carrier*delta_rate/C if motion else 0.
    range_rate = float(row["slant_range_rate_km_s"]) * 1000
    doppler = -carrier * range_rate / C if motion else 0.
    dt = (after_row["time_utc"] - before_row["time_utc"]).total_seconds()
    if dt < 0:
        raise ValueError("Orbit samples must be in increasing time order")
    acceleration = (float(after_row["slant_range_rate_km_s"]) -
                    float(before_row["slant_range_rate_km_s"])) * 1000 / dt if dt > 0 else 0.
    rate = -carrier * acceleration / C if motion else 0.
    prediction = predict_orbit(prediction_batch.models[index], row["time_utc"].to_pydatetime(),
                               config["station"], carrier, config.get("orbit_prediction")) if motion else {
                                   "predicted_doppler_hz": 0., "predicted_time_scale": 1.}
    losses = [("rain", float(row["net_loss_db"])), ("reference", float(row["reference_loss_db"]))]
    clock_mode = config["experiment"].get("reference_clock_mode","independent")
    if clock_mode not in ("shared","independent"):
        raise ValueError("reference_clock_mode must be shared or independent")
    configuration_hash = hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()
    for case_index, (case, loss) in enumerate(losses):
        if case_index and clock_mode=="independent":
            alternate=np.random.SeedSequence([int(config["experiment"]["seed"]),index,20261019]).generate_state(2)
            tx_clock=draw_clock(config["tx_clock"],int(alternate[0]))
            rx_clock=draw_clock(config["clock"],int(alternate[1]))
        noise_seed = int(np.random.SeedSequence([int(frame_seed), case_index, 91]).generate_state(1)[0])
        if _array_context and _array_context['noise_channel']:
            noise_seed = int(np.random.SeedSequence([noise_seed,_array_context['noise_channel']]).generate_state(1)[0])
        rain_spectrum = row_rain_spectrum(row, config, np.fft.fftfreq(NFFT, 1 / FS), case)
        rain_phase = row_rain_phase(row, config, np.fft.fftfreq(NFFT, 1 / FS), case)
        rain_rate = float(row.get("rain_phase_rate_rad_s", 0.)) if case == "rain" else 0.
        center_rain = float(row["rain_db"]) if case == "rain" else 0.
        spectral_correction = rain_spectrum - center_rain
        receiver_settings = case_receiver_settings(row, config["receiver"], case)
        supplied_paths = _array_context['paths'][case] if _array_context else None
        paper_paths, direct_evolution = prepare_frame_paths(row, config, case,paths=supplied_paths) if paper_enabled else ([], None)
        amplitude_rate = 0.
        if direct_evolution is not None:
            f = np.asarray(direct_evolution['frequency_offset_hz'])
            rain_spectrum = np.interp(np.fft.fftfreq(NFFT, 1/FS), f, direct_evolution['rain_loss_db'])
            rain_phase = np.interp(np.fft.fftfreq(NFFT, 1/FS), f, direct_evolution['rain_phase_rad'])
            revised_center = float(np.interp(0., f, direct_evolution['rain_loss_db']))
            loss += revised_center-center_rain
            center_rain = revised_center
            spectral_correction = rain_spectrum-center_rain
            rain_rate = direct_evolution['rain_phase_rate_rad_s']
            amplitude_rate = direct_evolution['log_amplitude_rate_per_s']
        rx, truth = simulate_capture(tx, loss, carrier, doppler, rate, tx_clock, rx_clock,
                                     receiver_settings, int(frame_seed) + 3 + (case_index*131 if clock_mode=="independent" else 0),
                                     noise_seed, reflection=reflection, spectral_loss_db=spectral_correction,
                                     spectral_phase_rad=rain_phase, rain_phase_rate_rad_s=rain_rate,
                                     additional_paths=paper_paths, log_amplitude_rate_per_s=amplitude_rate,
                                     direct_relative_delay_s=_array_context['delay_s'] if _array_context else 0.)
        rx=convert_rate(rx,FS,recorded_rate)
        directory = Path(out) / case / relative_dir
        directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(satellite_index=index, satellite_name=str(row["satellite_name"]),
                        norad_id=int(prediction_batch.models[index].satnum), pass_id=str(row["pass_id"]),
                        time_utc=row["time_utc"].isoformat(), elevation_deg=float(row["elevation_deg"]),
                        slant_range_km=float(row["slant_range_km"]), complete_pass=bool(row["complete_pass"]),
                        case=case, frame_seed=int(frame_seed), frame_duration_s=len(tx) / FS,
                        configuration_sha256=configuration_hash,paired_clocks=clock_mode,
                        sampling_mode="independent_complete_starlink_frames", waveform_profile=PROFILE,
                        payload_mode=config["ofdm"].get("payload_mode", "template_qpsk"),
                        clock_parameters_source="configured_engineering_scenario_not_measured_Starlink_clocks",
                        receiver_known_data="PSS_SSS_and_published_edge_pilots_only",
                        nominal_tx_power_definition="active_OFDM_symbol_expected_power_W",
                        absolute_propagation_delay_s=float(row["slant_range_km"]) * 1000 / C,
                        absolute_TOA_and_interframe_phase_modeled=False,
                        channel_scenario=("paper_S_X_R_moving_LOS_environment_and_rain" if paper_enabled
                                          else "LOS_plus_local_specular_reflection" if reflection else "LOS"),
                        paired_noise="independent_thermal_noise")
        metadata.update(reference_role="simultaneous_synthetic_counterfactual_not_a_field_observation",
                        rain_frequency_model=config["propagation"].get("rain_frequency_model", "center_frequency_flat"),
                        rain_phase_model=config["propagation"].get("rain_phase_model", "none"),
                        rain_phase_scope=("coherent_forward_plus_finite_moving_rain_scatterers" if any(p['group']=='R' for p in paper_paths)
                                          else "coherent_forward_propagation_no_droplet_microdoppler"),
                        paper_channel_groups="H_S_plus_H_X_plus_H_R" if paper_enabled else "legacy_LOS",
                        diffuse_rain_scattering=any(p['group']=='R' for p in paper_paths),
                        rain_path_assumption=("individual_ray_integrated_complex_rain" if paper_enabled
                                              else "shared_mean_rain_transfer_across_LOS_and_local_reflection"),
                        noise_model=truth["noise_model"])
        metadata.update(receive_channels=1,capture_sample_rate_hz=recorded_rate,processing_sample_rate_hz=FS,
                        acquisition_mode="independent_frame_bursts",sample_rate_interface="ideal_linear_resampling_not_measured_analog_frontend",
                        capture_selection="conditional_on_target_frame_present_not_a_traffic_or_detection_probability_model",
                        cadence_basis=config.get("acquisition",{}).get("cadence_basis","computational_baseline_not_empirically_validated_for_rainfall"))
        gains = path_antenna_gains(
            row[['sat_x_km','sat_y_km','sat_z_km']].to_numpy(float)*1000,
            row[['station_x_km','station_y_km','station_z_km']].to_numpy(float)*1000,
            config['propagation'])
        metadata.update(antenna_model=gains['antenna_model'],
            direct_tx_gain_dbi=float(gains['tx_gain_dbi']), direct_rx_gain_dbi=float(gains['rx_gain_dbi']),
            direct_tx_offaxis_deg=float(gains['tx_offaxis_deg']), direct_rx_offaxis_deg=float(gains['rx_offaxis_deg']),
            tx_pointing_mode=gains['tx_pointing_mode'], rx_pointing_mode=gains['rx_pointing_mode'],
            frame_impairment_model='constant_oscillator_CFO_and_held_AGC_affine_TO_fixed_RF_chain_PO',
            ground_reflection_included=any(p.get('path_model')=='ground_specular' for p in paper_paths) or reflection is not None)
        np.save(directory / "rx_iq.npy", rx)
        np.savez_compressed(directory / "frame_config.npz", waveform_profile=PROFILE, sample_rate_hz=FS,
                            capture_sample_rate_hz=recorded_rate,
                            nfft=NFFT, cp_samples=CP, symbols=SYMBOLS, pilot_mask=mask, pilot_values=known)
        np.savez_compressed(directory / "orbit_prediction.npz", **prediction)
        (directory / "capture_metadata.json").write_text(json.dumps(metadata, indent=2))
        (directory / "receiver_log.json").write_text(json.dumps({"agc_gain_linear": truth["agc_gain_linear"]}))
        (directory / "retrieval_config.json").write_text(json.dumps(retrieval_config(row,config),indent=2))
        (directory / "simulation_truth.json").write_text(json.dumps(dict(truth, noise_seed=noise_seed,reflection=reflection,
            rain_phase_center_rad=float(row.get("rain_phase_rad", 0.)) if case == "rain" else 0.,
            rain_phase_rate_rad_s=rain_rate, rain_excess_doppler_hz=rain_rate/(2*np.pi),
            rain_phase_rate_basis="finite_difference_of_unwrapped_phase_at_geometry_UTC_samples",
            paper_paths=paper_paths,
            direct_evolution=direct_evolution,
            additional_path_time_model="locally_affine_delay_and_log_amplitude_with_individual_ray_rain_phase"), indent=2))
        amplitude = 10 ** (-loss / 20)
        response=channel_response(NFFT,FS,loss,reflection) * 10 ** (-spectral_correction / 20)*np.exp(1j*rain_phase)
        response += (amplitude*10**(-spectral_correction/20)*np.exp(1j*rain_phase)
                     *additional_response(paper_paths, np.fft.fftfreq(NFFT, 1/FS), FS))
        if _array_context:
            response *= np.exp(-2j*np.pi*(carrier+np.fft.fftfreq(NFFT,1/FS))*_array_context['delay_s'])
        np.savez_compressed(directory / "Hprop.npz", Hprop=response,
                            taps=np.fft.ifft(response),impulse_response_kind="periodic_DFT_not_causal_FIR",net_loss_db=loss,
                            rain_effective_mm_h=float(row.get("rain_effective_mm_h",np.nan)) if case=="rain" else 0.,
                            rain_db=center_rain, rain_db_by_frequency=rain_spectrum,
                            rain_phase_rad_by_frequency=rain_phase,
                            rain_phase_rate_rad_s=rain_rate,
                            frequency_offset_hz=np.fft.fftfreq(NFFT, 1 / FS))
        for old in ("csi_pilots_corrected.npz", "csi.png", "rain_estimate.npz", "rain_estimate.png"):
            (directory / old).unlink(missing_ok=True)
        print(f"Saved {case} Starlink public-profile capture: {directory}", flush=True)


def process_starlink_capture(directory, rx, known, agc_gain, plot=False):
    if np.asarray(rx).ndim == 2:
        from src.receive_array import process_array_capture
        return process_array_capture(directory,rx,known,agc_gain,plot)
    directory = Path(directory)
    with np.load(directory / "orbit_prediction.npz") as prediction:
        doppler = float(prediction["predicted_doppler_hz"])
        scale = float(prediction["predicted_time_scale"])
    if not np.isfinite(agc_gain) or agc_gain <= 0:
        raise ValueError("AGC gain must be finite and positive")
    result = receive_frame(rx / agc_gain, known, doppler, scale)
    np.savez_compressed(directory / "csi_pilots_corrected.npz", **result, waveform_profile=PROFILE,
                        frequency_offset_hz=np.fft.fftfreq(NFFT, 1 / FS),
                        symbol_center_s=(np.arange(SYMBOLS) * BLOCK + CP + (NFFT - 1) / 2) / FS,
                        agc_removed=True, agc_gain_used=agc_gain, sample_rate_hz=FS, nfft=NFFT,
                        cp_samples=CP, predicted_doppler_hz=doppler, predicted_time_scale=scale,
                        csi_definition="effective_channel_at_known_SSS_and_edge_pilots_only")
    quality = {key: float(result[key]) for key in ("capture_score", "estimated_residual_cfo_hz",
               "estimated_residual_sfo_ppm", "synchronization_coherence")}
    print("Starlink pilot-only CSI:", quality, flush=True)
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for band, label in ((slice(488, 496), "+edge"), (slice(528, 536), "-edge")):
            csi = result["pilot_csi"][2:, band]
            axes[0].plot(np.arange(2, SYMBOLS), 10 * np.log10(np.mean(np.abs(csi) ** 2, axis=1)), label=label)
            axes[1].plot(np.arange(2, SYMBOLS), result["cpe_rad"][2:], alpha=.5)
        axes[0].set_ylabel("Effective CSI power (dB)")
        axes[0].legend()
        axes[1].set(xlabel="Symbol slot", ylabel="Estimated common phase (rad)")
        fig.tight_layout()
        fig.savefig(directory / "csi.png", dpi=160)
        plt.close(fig)
    return quality
