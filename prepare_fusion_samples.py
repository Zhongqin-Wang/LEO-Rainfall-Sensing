"""Align cloud histories with receiver-only CSI; never manufacture missing pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from torch.nn.functional import interpolate
import torch

ROOT = Path(__file__).resolve().parent


def utc(value):
    t = pd.Timestamp(str(value))
    if t.tzinfo is None:
        raise ValueError('All timestamps must include timezone')
    return t.tz_convert('UTC')


def receiver_sequence(path):
    """Calibrated log magnitude by antenna/frequency band; no truth/clear channel."""
    with np.load(path, allow_pickle=False) as d:
        if not bool(d['agc_removed']):
            raise ValueError('CSI requires calibrated gain / agc_removed')
        h = d['pilot_csi']
        f = d['frequency_offset_hz']
        if h.ndim != 3 or h.shape[1] != len(f) or not np.isfinite(h).all():
            raise ValueError('Expected finite [frame,pilot,antenna] CSI')
        # Retain absolute calibrated power: do not normalize each capture separately.
        edges = np.linspace(f.min(), f.max(), 5)
        features = []
        for j in range(4):
            selected = (f >= edges[j]) & ((f <= edges[j+1]) if j == 3 else (f < edges[j+1]))
            if not selected.any():
                raise ValueError('Empty CSI frequency band')
            power = np.mean(abs(h[:, selected, :])**2, axis=1)
            features.append(10*np.log10(np.maximum(power, 1e-30)))
        x = np.concatenate(features, axis=1).astype('float32')
        # Require whole capture to precede issue time (conservative completion time).
        return x, np.asarray(d['frame_start_s'], dtype=float)


def window_specs(data, times, schedule):
    windows = data['window_indices']
    if schedule is None:
        for i, indices in enumerate(windows):
            yield i, indices, times[indices[-1]], utc(data['target_time_utc'][i]), float(data['rain_rate_mm_h'][i]), None, None
        return
    for row in schedule.itertuples(index=False):
        i = int(row.cloud_window_index)
        if not 0 <= i < len(windows):
            raise ValueError(f'Cloud window index out of range: {i}')
        indices = windows[i]
        if times[indices[-1]] != utc(row.last_cloud_observation_utc):
            raise ValueError(f'Schedule does not match the last cloud frame of window {i}')
        issue, target, created = map(utc, (row.issue_time_utc, row.target_time_utc, row.latest_cloud_file_created_utc))
        if not times[indices[-1]] <= created <= issue < target:
            raise ValueError(f'Cloud file creation and forecast timestamps are inconsistent for window {i}')
        source = Path(row.bom71_file)
        if not source.name.startswith('71_') or not source.is_file():
            raise ValueError(f'Missing Terrey Hills rain label source: {source}')
        yield i, indices, issue, target, float(row.rain_rate_mm_h), created, str(source)


def prepare(cloud_root, csi_root, output, steps=32, size=32, max_age=600, *, schedule_path=None, cloud_samples=None):
    if steps < 1 or size < 4 or max_age <= 0:
        raise ValueError('Invalid steps, image size or maximum age')
    if (schedule_path is None) != (cloud_samples is None):
        raise ValueError('--schedule and --cloud-samples must be supplied together')
    schedule = pd.read_csv(schedule_path, dtype={'cloud_window_index': 'int64'}) if schedule_path is not None else None
    if schedule is not None and (schedule.empty or schedule['cloud_window_index'].duplicated().any()):
        raise ValueError('Schedule must contain unique cloud window indices')
    output.mkdir(parents=True, exist_ok=True)
    captures, skipped, rows, clouds, sequences, rates = [], [], [], [], [], []
    for path in sorted(csi_root.rglob('csi_continuous.npz')):
        meta_path = path.parent/'capture_metadata.json'
        if not meta_path.exists():
            skipped.append(dict(source=str(path), reason='missing capture_metadata.json'))
            continue
        meta = json.loads(meta_path.read_text())
        config_path = path.parent/'configuration_as_run.yaml'
        if not config_path.exists():
            skipped.append(dict(source=str(path), reason='missing station configuration'))
            continue
        station = yaml.safe_load(config_path.read_text())['station']
        location = np.array([float(station['lat_deg']), float(station['lon_deg'])])
        if meta.get('case') != 'rain':
            skipped.append(dict(source=str(path), reason='not rainy/live-equivalent receiver'))
            continue
        try:
            x, offsets = receiver_sequence(path)
            start = utc(meta['time_utc'])
            duration = float(meta['duration_s'])
            if not np.isfinite(duration) or duration <= 0 or not len(x) or len(offsets) != len(x) or not np.isfinite(offsets).all() or np.any(offsets < 0) or np.any(offsets >= duration) or np.any(np.diff(offsets) <= 0):
                raise ValueError('Invalid capture frame timing')
            end = start+pd.Timedelta(seconds=duration)
            # Bin observed frames only; mask appended below distinguishes absent bins.
            seq = np.zeros((steps, x.shape[1]+1), dtype='float32')
            bins = np.minimum((offsets/duration*steps).astype(int), steps-1)
            for b in np.unique(bins):
                seq[b, :-1] = x[bins == b].mean(axis=0)
                seq[b, -1] = 1
            captures.append(dict(path=str(path.resolve()), start=start, end=end, seq=seq,
                                 duration_s=duration, location=location))
        except (ValueError, KeyError) as error:
            skipped.append(dict(source=str(path), reason=str(error)))
    seen = set()
    expected_lead = None
    for path in ([cloud_samples] if schedule is not None else sorted(cloud_root.glob('*/samples.npz'))):
        with np.load(path, allow_pickle=False) as d:
            times = [utc(t) for t in d['time_utc']]
            for i, indices, issue, target, y, created, bom_source in window_specs(d, times, schedule):
                if len(indices) < 2 or np.any(np.diff(indices) != 1):
                    raise ValueError('Cloud history must have consecutive frame indices')
                if any(times[b]-times[a] != pd.Timedelta(minutes=10) for a,b in zip(indices,indices[1:])):
                    raise ValueError('Cloud history must be contiguous ten-minute frames')
                location = np.array([float(d['station_lat_deg']), float(d['station_lon_deg'])])
                history_start = times[indices[0]]
                lead = (target-issue).total_seconds()
                if expected_lead is not None and lead != expected_lead:
                    raise ValueError('Mixed lead times require separate datasets/models')
                expected_lead = lead
                if target <= issue:
                    raise ValueError('Target must be strictly after cloud history')
                if any(times[j] > issue for j in indices):
                    raise ValueError('Future cloud in history')
                candidates = [c for c in captures if np.allclose(c['location'], location, atol=1e-5, rtol=0) and c['start'] >= history_start and c['end'] <= issue
                              and (issue-c['end']).total_seconds() <= max_age]
                if not candidates:
                    skipped.append(dict(source=str(path), sample=i, issue_time_utc=issue.isoformat(),
                                        reason='no completed CSI capture in history within max age'))
                    continue
                c = max(candidates, key=lambda c: c['end'])
                identity = (issue.isoformat(), target.isoformat())
                if identity in seen:
                    raise ValueError('Duplicate forecast issue/target pair')
                seen.add(identity)
                cloud = d['brightness_temperature_k'][indices]
                mask = d['valid_mask'][indices] & np.isfinite(cloud)
                if not mask.reshape(len(indices), -1).any(axis=1).all():
                    raise ValueError('Cloud frame has no valid pixels')
                if not np.isfinite(y) or y < 0:
                    raise ValueError('Invalid station rain-rate label')
                # Fixed physical scaling, not a scaler fitted on future/test data.
                value = np.where(mask, (cloud-250.)/50., 0).astype('float32')
                image = torch.from_numpy(np.concatenate([value, mask.astype('float32')], axis=0))[None]
                image = interpolate(image, size=(size, size), mode='nearest')[0].numpy()
                clouds.append(image); sequences.append(c['seq']); rates.append(y)
                rows.append(dict(issue_time_utc=issue.isoformat(), target_time_utc=target.isoformat(),
                    history_start_utc=history_start.isoformat(), csi_source=c['path'],
                    last_cloud_observation_utc=times[indices[-1]].isoformat(),
                    latest_cloud_file_created_utc=created.isoformat() if created is not None else None,
                    bom71_file=bom_source, rain_rate_mm_h=y,
                    csi_start_utc=c['start'].isoformat(), csi_end_utc=c['end'].isoformat(),
                    csi_duration_s=c['duration_s'], cloud_source=str(path.resolve()), cloud_sample=i,
                    # Date groups are provisional: users must group multi-day storms together.
                    event_group=issue.strftime('%Y-%m-%d')))
    if rows and len({x.shape for x in sequences}) != 1:
        raise ValueError('Matched CSI captures have different channel counts; prepare consistent captures')
    report = dict(schedule=str(schedule_path) if schedule_path is not None else None, paired_samples=len(rows), receiver_captures=len(captures), skipped=skipped,
                  operational_availability_verified=False,
                  label_semantics='future station-grid instantaneous rain rate, mm/h; not accumulated mm',
                  notes=['Cloud creation timestamps constrain issue time when using a schedule; arrival latency unverified.' if schedule is not None else 'Cloud nominal times only; latency unverified: offline experiment.',
                         'CSI is the most recent complete capture, not a continuous hour of observations.',
                         'Date groups must be reviewed for storms spanning midnight.',
                         'Simulated receiver CSI is not measured RF data.'])
    (output/'pairing_report.json').write_text(json.dumps(report, indent=2))
    if not rows:
        print('No valid CSI/cloud pairs. See '+str(output/'pairing_report.json'))
        return report
    np.savez_compressed(output/'samples.npz', csi=np.stack(sequences), cloud=np.stack(clouds),
        rain_rate_mm_h=np.array(rates, dtype='float32'),
        **{k:np.array([r[k] for r in rows]) for k in ('issue_time_utc','target_time_utc','history_start_utc','event_group')},
        synthetic_demo=False)
    pd.DataFrame(rows).to_csv(output/'sample_index.csv', index=False)
    print(json.dumps({k:v for k,v in report.items() if k != 'skipped'}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud-root', type=Path, default=ROOT/'data/cloud_rain_samples')
    p.add_argument('--csi-root', type=Path, default=ROOT/'outputs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--image-size', type=int, default=32)
    p.add_argument('--max-csi-age-seconds', type=float, default=600)
    p.add_argument('--schedule', type=Path, help='Sydney forecast schedule CSV with Terrey Hills labels')
    p.add_argument('--cloud-samples', type=Path, help='Cloud NPZ indexed by the schedule')
    a = p.parse_args()
    prepare(a.cloud_root, a.csi_root, a.output, a.steps, a.image_size, a.max_csi_age_seconds,
            schedule_path=a.schedule, cloud_samples=a.cloud_samples)
