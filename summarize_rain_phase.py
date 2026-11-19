"""Summarize coherent-rain numerical checks and an explicitly synthetic pass."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rain_complex import (C, water_permittivity, forward_amplitude_m,
                              complex_rain_spectrum)


def summarize(root):
    root = Path(root)
    metadata = json.loads((root/'run_metadata.json').read_text())
    config = metadata['configuration']
    propagation = pd.read_parquet(root/'propagation.parquet').sort_values('time_utc')
    manifests = list((root/'passes').glob('*/rain_timeseries.csv'))
    if not manifests:
        raise ValueError("No completed pass results")
    results = pd.concat([pd.read_csv(p) for p in manifests], ignore_index=True)
    success = results.status == 'ok'
    failed = int((~success).sum())
    error = results.loc[success, 'error_db'].to_numpy()
    f = np.linspace(11.95e9, 12.2e9, 17)
    a = np.array(complex_rain_spectrum(f, 12., [1., 10., 100.], quadrature_order=96))
    b = np.array(complex_rain_spectrum(f, 12., [1., 10., 100.], quadrature_order=192))
    carrier = float(config['link']['carrier_hz'])
    k = 2*np.pi*carrier/C
    radius = 4e-6
    eps = water_permittivity(carrier)
    rayleigh = k*k*radius**3*(eps-1)/(eps+2)
    rayleigh_error = abs(forward_amplitude_m(carrier, radius*2000)/rayleigh-1)
    tests = ET.parse(root/'tests.xml').getroot()
    cases = tests.findall('.//testcase')
    failures = tests.findall('.//failure')+tests.findall('.//error')
    retained_fields = True
    max_response_error = 0.
    for manifest in (root/'passes').glob('*/captures.csv'):
        for _, row in pd.read_csv(manifest).iterrows():
            for case in ['rain', 'reference']:
                directory = root/row[case+'_capture']
                with np.load(directory/'csi_pilots_corrected.npz') as data:
                    retained_fields &= all(key in data for key in ['pilot_csi_orbit_only', 'pilot_csi_before_cpe', 'pilot_csi'])
                with np.load(directory/'Hprop.npz') as data:
                    # This particular validation config is LOS, without reflection.
                    expected = (10**(-(float(data['net_loss_db'])-float(data['rain_db'])
                                      +data['rain_db_by_frequency'])/20)
                                *np.exp(1j*data['rain_phase_rad_by_frequency']))
                    max_response_error = max(max_response_error,
                        float(np.max(abs(data['Hprop']-expected)/np.maximum(abs(expected), 1e-300))))
                    if case == 'reference' and np.any(data['rain_phase_rad_by_frequency'] != 0):
                        raise AssertionError("Clear control contains nonzero rain phase")
    summary = dict(
        report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_generation_provenance="run_metadata.json: original source hashes and configuration as run",
        interpretation="Numerical simulator consistency only; no measured Starlink rain-phase validation or rainfall algorithm accuracy claim.",
        amplitude_model=config['propagation']['rain_frequency_model'],
        phase_model=config['propagation']['rain_phase_model'],
        test_cases=len(cases), test_failures=len(failures), attempted_pairs=len(results),
        successful_pairs=int(success.sum()), failed_pairs=failed,
        attenuation_rmse_db=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        attenuation_evaluation="Ideal simultaneous counterfactual clear link, calibrated gain; amplitude check only, not phase estimator validation.",
        center_rain_phase_rad_range=[float(propagation.rain_phase_rad.min()), float(propagation.rain_phase_rad.max())],
        rain_excess_doppler_hz_range=[float(propagation.rain_phase_rate_rad_s.min()/(2*np.pi)), float(propagation.rain_phase_rate_rad_s.max()/(2*np.pi))],
        geometry_doppler_hz_range=[float((-carrier*propagation.slant_range_rate_km_s*1000/C).min()),
                                   float((-carrier*propagation.slant_range_rate_km_s*1000/C).max())],
        diameter_quadrature_max_abs_difference_loss_db=float(np.max(abs(a[0]-b[0]))),
        diameter_quadrature_max_abs_difference_phase_rad=float(np.max(abs(a[1]-b[1]))),
        rayleigh_small_drop_relative_error=float(rayleigh_error),
        los_complex_truth_relative_error=max_response_error,
        pre_correction_csi_retained=bool(retained_fields),
        clock_identifiability="Common LoS rain phase is canceled by same-path conjugate referencing; fitted CFO can absorb rain propagation phase rate.",
        interframe_carrier_coherence=False,
    )
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    seconds = (propagation.time_utc-propagation.time_utc.iloc[0]).dt.total_seconds()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    axes[0, 0].plot(seconds, propagation.rain_db, 'o-')
    axes[0, 0].set(ylabel='Rain attenuation at carrier (dB)')
    axes[0, 1].plot(seconds, propagation.rain_phase_rad, 'o-', color='tab:purple')
    axes[0, 1].set(ylabel='Rain propagation phase (rad, unwrapped)')
    axes[1, 0].plot(seconds, -carrier*propagation.slant_range_rate_km_s*1000/C/1000, 'o-')
    axes[1, 0].set(ylabel='Geometric satellite Doppler (kHz)')
    axes[1, 1].plot(seconds, propagation.rain_phase_rate_rad_s/(2*np.pi), 'o-', color='tab:orange')
    axes[1, 1].set(ylabel='Rain propagation excess Doppler (Hz)')
    for ax in axes.flat:
        ax.set_xlabel('Seconds since first pass sample')
        ax.grid(alpha=.25)
    fig.suptitle('Synthetic coherent-rain pass: separate geometry and rain phase\n'
                 'Spherical Mie / Marshall-Palmer; 10 s weather grid; independent IQ bursts', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(root/'rain_phase_components.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    if failed or failures or not retained_fields or max_response_error > 1e-12:
        raise SystemExit(2)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent/'outputs/rain_phase_validation')
    summarize(parser.parse_args().root)
