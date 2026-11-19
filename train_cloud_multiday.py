"""Prepare contiguous daily cloud windows and evaluate ridge on later held-out dates."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.cloud_regression import FEATURE_NAMES, extract_features, fit_ridge, predict

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help="Directory containing the Himawari input files")
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/cloud_multiday')
    parser.add_argument('--test-days', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    daily = {}
    for path in sorted(args.source.glob('NC_H09_*_R21_FLDK.07001_06001.nc')):
        parts = path.name.split('_')
        if '0800' <= parts[3] <= '1000':
            daily.setdefault(parts[2], []).append(pd.to_datetime(parts[2]+parts[3], format='%Y%m%d%H%M', utc=True))
    tables, inventory = [], []
    for day, times in sorted(daily.items()):
        groups = []
        for t in times:
            if not groups or t-groups[-1][-1] != pd.Timedelta(minutes=10):
                groups.append([])
            groups[-1].append(t)
        expected = pd.date_range(times[0].normalize()+pd.Timedelta(hours=8), periods=13, freq='10min')
        inventory.append(dict(date=day, missing_cloud_times=[t.isoformat() for t in expected if t not in times]))
        for group in groups:
            if len(group) < 7:
                continue
            start, end = group[0], group[-1]
            command = [sys.executable, str(ROOT/'prepare_cloud_samples.py'), '--source', str(args.source),
                       '--start', start.isoformat(), '--end', end.isoformat()]
            print(f'Preparing {day}: {start:%H:%M}-{end:%H:%M} UTC', flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
            sample = ROOT/'data/cloud_rain_samples'/f'batch_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}_lead10m'/'samples.npz'
            with np.load(sample, allow_pickle=False) as data:
                x = extract_features(data['brightness_temperature_k'], data['valid_mask'], data['window_indices'])
                table = pd.DataFrame(x, columns=FEATURE_NAMES)
                table['target_time_utc'] = data['target_time_utc'].astype(str)
                table['bom_rain_rate_mm_h'] = data['rain_rate_mm_h']
                table['date'] = day
                table['source_samples'] = str(sample)
                tables.append(table)
    if not tables:
        raise ValueError('No complete one-hour cloud windows available')
    table = pd.concat(tables, ignore_index=True).sort_values('target_time_utc').reset_index(drop=True)
    if table['target_time_utc'].duplicated().any():
        raise ValueError('Duplicate target timestamps')
    days = sorted(table['date'].unique())
    if args.test_days < 1 or len(days) <= args.test_days:
        raise ValueError('Need training dates and at least one held-out test date')
    test_days = days[-args.test_days:]
    train = ~table['date'].isin(test_days).to_numpy()
    x, y = table[FEATURE_NAMES].to_numpy(), table['bom_rain_rate_mm_h'].to_numpy()
    # Fit scaler and coefficients using training dates only. Test dates never tune alpha.
    mean, scale, coefficients, intercept = fit_ridge(x[train], y[train], 1.)
    estimate = predict(x, mean, scale, coefficients, intercept)
    table['split'] = np.where(train, 'train', 'test')
    table['raw_model_output_mm_h'] = estimate
    table['training_mean_baseline_mm_h'] = intercept
    table.to_csv(args.output/'predictions.csv', index=False)
    np.savez_compressed(args.output/'ridge_model.npz', feature_names=np.array(FEATURE_NAMES),
                        feature_mean=mean, feature_scale=scale, coefficients=coefficients,
                        intercept=intercept, alpha=1., train_dates=np.array(days[:-args.test_days]),
                        test_dates=np.array(test_days), output_units='mm/h')
    report = dict(train_dates=days[:-args.test_days], test_dates=test_days, daily_inventory=inventory,
                  notes=['Chronological held-out dates; overlapping windows within a date are not independent.',
                         'Fixed alpha=1; preprocessing fitted on training dates only.',
                         'Cloud-only baseline; no CSI. Raw negative regression outputs are retained.',
                         'Missing cloud frames are not interpolated. Only contiguous one-hour histories are used.',
                         'Small offline experiment; no operational forecasting accuracy claim.'])
    for name, mask in [('train', train), ('test', ~train)]:
        report[name] = dict(samples=int(mask.sum()), mae_mm_h=float(np.mean(abs(estimate[mask]-y[mask]))),
                            rmse_mm_h=float(np.sqrt(np.mean((estimate[mask]-y[mask])**2))),
                            mean_baseline_mae_mm_h=float(np.mean(abs(intercept-y[mask]))),
                            mean_baseline_rmse_mm_h=float(np.sqrt(np.mean((intercept-y[mask])**2))),
                            negative_predictions=int(np.sum(estimate[mask]<0)))
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, name, mask in zip(axes, ['Training fit', 'Held-out later dates'], [train, ~train]):
        ids = np.flatnonzero(mask)
        ax.plot(range(len(ids)), y[mask], 'o-', label='BOM reference')
        ax.plot(range(len(ids)), estimate[mask], '.--', label='Ridge output')
        ax.axhline(intercept, color='gray', linestyle=':', label='Training mean baseline')
        ticks = list(range(0, len(ids), max(1, len(ids)//8)))
        ax.set_xticks(ticks, [pd.Timestamp(table.iloc[ids[i]]['target_time_utc']).strftime('%m-%d %H:%M') for i in ticks], rotation=25)
        ax.set(title=name, xlabel='Target time (UTC; gaps between dates)', ylabel='Rain rate (mm/h)')
        ax.grid(alpha=.3)
        ax.legend()
    fig.savefig(args.output/'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
