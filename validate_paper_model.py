"""Current-paper moving-LEO S/X/R channel check; analytical CSI and one separate IQ pair.

This verifies the supplied manuscript's component model, not its reference
estimator or rainfall accuracy. All oracle paths and controls are labelled.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.time import Time, TimeDelta
from sgp4.api import SatrecArray

from src.models import TLEBatch
from src.orbit import load_orbit_batch, _propagate_itrs
from src.earth_orientation import ensure_earth_orientation
from src.paper_channel import prepare_frame_paths, render_components
from src.propagation import propagation_table
from src.weather import clear_weather, KEYS
from src.clocks import receive_chain_phase, held_receiver_gain
from src.antenna import path_antenna_gains
from src.receive_array import channel_count, array_layout, render_array_components
import hashlib
from src.starlink_waveform import NFFT, FS, GUTTER, SYMBOLS, BLOCK, CP
from src.starlink_link import generate_starlink_pair, draw_clock
from src.provenance import software_provenance
from run_csi import process_capture

ROOT = Path(__file__).resolve().parent


def run(config, geometry_dir, output, norad_id=100520, snapshots=192, snapshot_rate=750.):
    channels = channel_count(config)
    positions, phases = array_layout(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not 16 <= snapshots <= 4096 or not 0 < snapshot_rate <= 750:
        raise ValueError("Use 16--4096 SSS snapshots at at most 750 nominal frames/s")
    geometry = pd.read_parquet(Path(geometry_dir)/'geometry.parquet')
    geometry = geometry[geometry.norad_id == norad_id].sort_values('time_utc').reset_index(drop=True)
    if geometry.empty:
        raise ValueError("Requested satellite has no geometry")
    weather = pd.read_parquet(Path(geometry_dir)/'weather_truth.parquet')
    settings = json.loads((Path(geometry_dir)/'geometry_metadata.json').read_text())
    if settings['config'] != json.loads(json.dumps({k: config[k] for k in ['geometry', 'station', 'weather']})):
        raise ValueError("Geometry/weather configuration mismatch")
    ensure_earth_orientation(geometry.time_utc.iloc[0].to_pydatetime(), ROOT/config['geometry']['eop_path'])
    carrier = float(config['link']['carrier_hz'])
    prop = propagation_table(geometry, weather, carrier, config['propagation'])
    dry = propagation_table(geometry, clear_weather(weather), carrier, config['propagation'])
    records = geometry.merge(prop, on=KEYS, validate='one_to_one').merge(
        weather[KEYS+[c for c in ['rain_rate_mm_h', 'rain_rate_samples_mm_h'] if c in weather]],
        on=KEYS, validate='one_to_one')
    records = records.merge(dry[KEYS+['net_loss_db']].rename(columns={'net_loss_db': 'reference_loss_db'}), on=KEYS)
    anchor = int(records.elevation_deg.argmax())
    row = records.iloc[anchor]
    batch = load_orbit_batch(ROOT/config['geometry']['orbit_path'])
    index = int(row.satellite_index)
    single = TLEBatch(np.array([batch.names[index]]), [batch.models[index]], SatrecArray([batch.models[index]]))
    times = np.arange(snapshots)/snapshot_rate
    epochs = Time(row.time_utc.to_pydatetime(), scale='utc')+TimeDelta(times, format='sec')
    valid, itrs = _propagate_itrs(single, epochs)
    if not valid.all():
        raise ValueError("Invalid SGP4 state within CPI")
    satellites = np.moveaxis(itrs.cartesian.xyz.to_value('m')[:, 0, :], 0, 1)
    frequency = np.fft.fftfreq(NFFT, 1/FS)
    mask = np.ones((snapshots, NFFT), bool)
    mask[:, GUTTER] = False
    paths, frame_truth = prepare_frame_paths(row, config)
    dry_paths, _ = prepare_frame_paths(row, config, 'reference')
    tx_clock = draw_clock(config['tx_clock'], 711)
    rx_clock = draw_clock(config['clock'], 712)
    rng = np.random.default_rng(713)
    # Each selected SSS belongs to a different frame. Initial frame phases are
    # unknown, but every symbol inside that frame follows one CFO ramp.
    cfo = tx_clock['lo_offset_hz']-rx_clock['lo_offset_hz']
    symbol_time = (np.arange(SYMBOLS)*BLOCK+CP+(NFFT-1)/2)/FS
    initial_phase_parameter = rng.uniform(-np.pi, np.pi, snapshots)
    frame_phase_parameter = initial_phase_parameter[:, None]-2*np.pi*cfo*symbol_time
    timing = ((tx_clock['timing_offset_samples']+rx_clock['timing_offset_samples'])/FS
              +(rx_clock['sfo_ppm']-tx_clock['sfo_ppm'])*1e-6*times)
    po = receive_chain_phase(config['receiver']) if channels == 1 else phases
    common = np.exp(-1j*frame_phase_parameter[:, 1, None]
                    -2j*np.pi*frequency[None, :]*timing[:, None])
    common = common*np.exp(-1j*po) if channels == 1 else common[...,None]*np.exp(-1j*po)
    if channels > 1:
        mask = np.broadcast_to(mask[...,None],(snapshots,NFFT,channels))
    components = {}
    cancellation_error = 0.
    for case, rays in [('rain', paths), ('reference', dry_paths)]:
        loss = float(row.net_loss_db if case == 'rain' else row.reference_loss_db)
        renderer = render_components if channels == 1 else render_array_components
        channel = renderer(row, rays, satellites, times, frequency, carrier, loss,
                                    config=config, case=case)
        total = channel['S']+channel['X']+channel['R']
        measured = total*common
        gains = np.array([held_receiver_gain(value, config['receiver']) for value in total]) if channels == 1 else np.array([
            [held_receiver_gain(value[:,m],config['receiver']) for m in range(channels)] for value in total])
        raw_measured = gains[:, None]*measured
        oracle_reference = channel['LoS']*common
        oracle_product = measured*oracle_reference.conj()
        cancellation_error = max(cancellation_error, float(np.max(abs(
            oracle_product-total*channel['LoS'].conj()))/np.max(abs(total*channel['LoS'].conj()))))
        values = dict(H_S=channel['S'], H_X=channel['X'], H_R=channel['R'], H_LoS=channel['LoS'],
            C_model=np.where(mask, raw_measured, np.nan),
            C_model_agc_calibrated=np.where(mask, measured, np.nan), known_SSS_mask=mask,
            agc_gain_linear=gains, rf_chain_phase_offset_rad=po, oscillator_CFO_hz=cfo,
            antenna_positions_enu_m=positions,receive_channels=channels,
            frame_initial_phase_parameter_rad=initial_phase_parameter,
            within_frame_phase_parameter_rad=frame_phase_parameter, symbol_center_s=symbol_time,
            TO_at_SSS_s=timing,
            oracle_LoS_reference=np.where(mask, oracle_reference, np.nan),
            oracle_reference_product=np.where(mask, oracle_product, np.nan),
            observation_kind='analytical_diagonal_CSI_not_extracted_from_IQ',
            agc_calibrated=False, calibrated_field='C_model_agc_calibrated', measurement_noise_added=False,
            intercarrier_interference='omitted_in_this_analytical_CSI_check_retained_in_separate_IQ_pair',
            oracle_label='known_LoS_used_only_for_model_validation_not_an_estimator',
            frequency_offset_hz=frequency, time_s=times, sample_rate_hz=snapshot_rate,
            orbit_source='SGP4_at_each_snapshot_TEME_to_ITRS',
            frame_phase_assumption='one_initial_phase_and_one_CFO_ramp_per_frame_shared_across_all_symbols',
            antenna_and_rain_evolution='per_snapshot_angular_patterns_and_individual_ray_complex_rain')
        np.savez_compressed(output/f'{case}_cpi_model.npz', **values)
        components[case] = channel
    frame_seed = 20261018
    generate_starlink_pair(row, records.iloc[max(anchor-1, 0)], records.iloc[min(anchor+1, len(records)-1)],
                          config, batch, output, Path('iq_example'), frame_seed)
    iq_quality = {case: process_capture(output/case/'iq_example', plot=False) for case in ['rain', 'reference']}
    # The analytical renderer and IQ transfer use independent construction
    # paths. Compare the physical epoch transfer, not a receiver's unknown CPE.
    transfer_errors = {}
    for case in ['rain', 'reference']:
        with np.load(output/case/'iq_example/Hprop.npz') as truth:
            expected = truth['Hprop']
        actual = sum(components[case][key][0] for key in ['S', 'X', 'R'])
        transfer_errors[case] = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        if transfer_errors[case] > 3e-6:
            raise AssertionError(f'{case} analytical / IQ channel mismatch: {transfer_errors[case]}')
    gain_columns = [c for c in ['antenna_gain_db','tx_gain_dbi','rx_gain_dbi','tx_offaxis_deg','rx_offaxis_deg'] if c in prop]
    diagnostic = geometry[KEYS+['elevation_deg']].merge(prop[KEYS+gain_columns], on=KEYS)
    diagnostic.to_csv(output/'antenna_geometry_diagnostics.csv', index=False)
    description = ROOT / "MODEL.md"
    model_source = {
        "path": str(description.relative_to(ROOT)),
        "sha256": hashlib.sha256(description.read_bytes()).hexdigest(),
    }
    (output/'configuration_as_run.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (output/'path_truth.json').write_text(json.dumps(paths, indent=2))
    pd.DataFrame([{k: v for k, v in p.items() if not isinstance(v, list)} for p in paths]).to_csv(output/'path_doppler_truth.csv', index=False)
    if channels > 1:
        # Existing summary plots show RX0; full multi-channel tensors remain in NPZ.
        components = {case:{k:v[...,0] for k,v in value.items()} for case,value in components.items()}
    rain_paths = [p for p in paths if p['group']=='R']
    velocities = np.array([p['relative_doppler_hz'] for p in rain_paths])
    def spectrum(x):
        window = np.hanning(len(x))
        return abs(np.fft.fftshift(np.fft.fft((x-x.mean())*window)))**2/(window@window)
    bins = np.fft.fftshift(np.fft.fftfreq(snapshots, 1/snapshot_rate))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for group, color in [('S', 'tab:blue'), ('X', 'tab:green'), ('R', 'tab:orange')]:
        signal = components['rain'][group][:, 0]/components['rain']['LoS'][:, 0]
        axes[0, 0].plot(bins, 10*np.log10(np.maximum(spectrum(signal), 1e-30)), label=group, color=color)
    axes[0, 0].set(xlabel='LoS-relative slow-time frequency (Hz)', ylabel='Component power (dB relative to LoS)', title='Oracle components, mean removed')
    axes[0, 0].legend()
    for key, name in [('relative_doppler_satellite_hz', 'Satellite geometry'), ('relative_doppler_scatterer_hz', 'Drop motion'), ('relative_doppler_hz', 'Total')]:
        axes[0, 1].plot(np.arange(len(rain_paths)), [p[key] for p in rain_paths], 'o', markersize=3, label=name)
    axes[0, 1].set(xlabel='Representative rain component', ylabel='LoS-relative Doppler at CPI start (Hz)', title='Per-path Doppler decomposition')
    axes[0, 1].legend(fontsize=8)
    for case, color in [('reference', 'tab:blue'), ('rain', 'tab:orange')]:
        ch = components[case]
        normalized = (ch['S'][:, 0]+ch['X'][:, 0]+ch['R'][:, 0])/ch['LoS'][:, 0]
        axes[1, 0].plot(bins, 10*np.log10(np.maximum(spectrum(normalized), 1e-30)), label=case, color=color)
    axes[1, 0].set(xlabel='LoS-relative slow-time frequency (Hz)', ylabel='Power (dB relative to LoS)', title='Oracle-normalized total channel')
    axes[1, 0].legend()
    cloud = components['rain']['R'][:, 0]/components['rain']['LoS'][:, 0]
    axes[1, 1].plot(times, cloud.real, label='Real')
    axes[1, 1].plot(times, cloud.imag, label='Imaginary', alpha=.7)
    axes[1, 1].set(xlabel='Time within CPI (s)', ylabel='Relative complex amplitude', title='Rain-scattering fluctuations')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.suptitle('Current paper S/X/R model with moving LEO transmitter\nSynthetic geometry/microphysics check; oracle plots, not rainfall retrieval', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output/'paper_model_components.png', dpi=160)
    plt.close(fig)
    summary = dict(model='H_S_plus_H_X_plus_H_R_with_moving_LEO_Tx',
        scope='geometry and numerical consistency, not measured rain calibration or algorithm performance',
        snapshots=snapshots, nominal_SSS_snapshot_rate_hz=snapshot_rate,
        CPI_support_s=snapshots/snapshot_rate, unpadded_frequency_bin_hz=snapshot_rate/snapshots,
        path_counts={group: sum(p['group']==group for p in paths)+(1 if group=='S' else 0) for group in ['S', 'X', 'R']},
        rain_relative_doppler_hz_range=[float(velocities.min()), float(velocities.max())] if velocities.size else [],
        sampled_geometric_paths_within_nyquist=bool(all(abs(p['relative_doppler_hz']) < snapshot_rate/2 for p in paths)),
        common_clock_oracle_cancellation_relative_error=cancellation_error,
        analytical_vs_IQ_physical_channel_relative_error=transfer_errors,
        paper_model_source=model_source,
        model_alignment=dict(components=['S','X','R'],
            ground_reflection=any(p.get('path_model')=='ground_specular' for p in paths),
            angular_gain_applies_to='direct_environment_and_rain_paths',
            rain_transfer='individual_ray_integrated_amplitude_and_phase',
            CFO='one_frequency_per_frame_with_symbol_phase_accumulation',
            AGC='one_held_voltage_gain_per_frame',
            TO='affine_SFO_drift_or_constant_when_SFO_zero',
            PO='fixed_receive_RF_chain_parameter_across_frames',
            CSI='known_nonzero_SSS_and_edge_pilots_only_in_IQ_receiver',
            implemented_receive_channels=channels,
            antenna_positions_enu_m=positions.tolist(),shared_LO_and_sample_clock=True),
        IQ_pair_receiver_results=iq_quality,
        assumptions=['individual ray integrated Mie rain amplitude and phase', 'short-CPI frozen point-scattering strengths/DSD weights',
                     'scalar circular port; spherical drops', 'finite rain representatives, not all physical drops',
                     'X point strengths and rain volume/wind are declared scenario inputs',
                     'fixed RX pointing; idealized TX serving-beam tracking',
                     'capped parabolic antenna pattern, not measured Starlink/SDR patterns'],
        configuration=config, software=software_provenance(ROOT))
    (output/'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ['model', 'path_counts', 'rain_relative_doppler_hz_range',
        'sampled_geometric_paths_within_nyquist', 'common_clock_oracle_cancellation_relative_error']}, indent=2))
    if cancellation_error > 1e-12:
        raise AssertionError('Common phase/TO cancellation failed')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/paper_moving_leo.yaml')
    parser.add_argument('--geometry-dir', type=Path, default=ROOT/'outputs/research_audit/temporal')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/current_paper_model')
    parser.add_argument('--norad-id', type=int, default=100520)
    parser.add_argument('--snapshots', type=int, default=192)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.geometry_dir, args.output, args.norad_id, args.snapshots)
