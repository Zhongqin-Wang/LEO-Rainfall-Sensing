"""Prepare sliding B13 windows and future station rain-rate labels."""
import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import yaml
from pyproj import Geod

from src.bom_rain import BOMRain

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help="Directory containing the Himawari input files")
    parser.add_argument('--config', type=Path, default=ROOT/'configs/continuous_4rx_era5.yaml')
    parser.add_argument('--start', default='2026-10-12T08:00:00Z')
    parser.add_argument('--end', default='2026-10-12T10:00:00Z')
    parser.add_argument('--history-minutes', type=int, default=60)
    parser.add_argument('--lead-minutes', type=int, default=10)
    parser.add_argument('--size-km', type=float, default=100.)
    parser.add_argument('--grid', default='07001_06001')
    parser.add_argument('--output', type=Path, default=ROOT/'data/cloud_rain_samples')
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    if start.tzinfo is None:
        raise ValueError('--start must include a timezone, such as Z for UTC')
    start = start.tz_convert('UTC')
    if args.history_minutes < 10 or args.history_minutes % 10 or args.lead_minutes <= 0 or args.size_km <= 0:
        raise ValueError('History must be a positive multiple of 10; lead and size must be positive')
    end = pd.Timestamp(args.end)
    if end.tzinfo is None:
        raise ValueError('--end must include a timezone')
    end = end.tz_convert('UTC')
    if end-start < pd.Timedelta(minutes=args.history_minutes) or (end-start).total_seconds() % 600:
        raise ValueError('Time range must cover the history and align to 10-minute steps')
    times = pd.date_range(start, end, freq='10min')
    window_size = args.history_minutes//10+1
    config = yaml.safe_load(args.config.read_text())
    station = config['station']
    lat0, lon0 = float(station['lat_deg']), float(station['lon_deg'])
    geod = Geod(ellps='WGS84')
    half = args.size_km*500
    west = geod.fwd(lon0, lat0, 270, half)[0]
    east = geod.fwd(lon0, lat0, 90, half)[0]
    south = geod.fwd(lon0, lat0, 180, half)[1]
    north = geod.fwd(lon0, lat0, 0, half)[1]
    frames, records = [], []
    latitude = longitude = None
    for time in times:
        name = f'NC_H09_{time:%Y%m%d_%H%M}_R21_FLDK.{args.grid}.nc'
        path = args.source/name
        with netCDF4.Dataset(path) as ds:
            lat, lon = ds['latitude'][:], ds['longitude'][:]
            ys = np.flatnonzero((lat >= south) & (lat <= north))
            xs = np.flatnonzero((lon >= west) & (lon <= east))
            if not len(ys) or not len(xs):
                raise ValueError(f'Station region outside {path}')
            y, x = slice(ys[0], ys[-1]+1), slice(xs[0], xs[-1]+1)
            current_lat, current_lon = np.asarray(lat[y]), np.asarray(lon[x])
            if latitude is not None and (not np.array_equal(latitude, current_lat) or not np.array_equal(longitude, current_lon)):
                raise ValueError('Cloud grids differ between frames')
            latitude, longitude = current_lat, current_lon
            var = ds['tbb_13']
            if var.units != 'K':
                raise ValueError('Expected B13 brightness temperature in kelvin')
            # netCDF4 automatically applies scale_factor, add_offset and fill masks.
            frame = np.ma.asarray(var[y, x], dtype=np.float32).filled(np.nan)
            if not np.isfinite(frame).any():
                raise ValueError(f'No valid B13 pixels in {path}')
            frames.append(frame)
            records.append(dict(file=str(path.resolve()), bytes=path.stat().st_size,
                                nominal_time_utc=time.isoformat(),
                                date_created=getattr(ds, 'date_created', None)))
    rain_directory = Path(config['weather']['bom']['directory'])
    if not rain_directory.is_absolute():
        rain_directory = ROOT/rain_directory
    reader = BOMRain(rain_directory)
    cloud = np.stack(frames)
    windows, labels, entries, label_records = [], [], [], []
    for i in range(len(times)-window_size+1):
        last = i+window_size-1
        target = times[last]+pd.Timedelta(minutes=args.lead_minutes)
        rain = reader.query_rain(station, target)
        if not rain['valid_mask'] or pd.Timestamp(rain['source_time_utc']) != target:
            raise ValueError(f'An exact valid BOM frame is required at {target}')
        windows.append(np.arange(i, last+1))
        labels.append(rain['rain_rate_mm_h'])
        label_records.append(rain)
        entries.append(dict(sample_index=i, input_start_utc=times[i].isoformat(),
                            input_end_utc=times[last].isoformat(), target_time_utc=target.isoformat(),
                            rain_rate_mm_h=rain['rain_rate_mm_h']))
    indices = np.asarray(windows)
    tag = f'batch_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}_lead{args.lead_minutes}m'
    out = args.output/tag
    out.mkdir(parents=True, exist_ok=True)
    # Each source frame is cropped once; indices describe overlapping windows.
    np.savez_compressed(out/'samples.npz', brightness_temperature_k=cloud,
                        valid_mask=np.isfinite(cloud), latitude_deg=latitude,
                        longitude_deg=longitude, time_utc=np.array([t.isoformat() for t in times]),
                        window_indices=indices,
                        target_time_utc=np.array([e['target_time_utc'] for e in entries]),
                        rain_rate_mm_h=np.array(labels),
                        station_lat_deg=lat0, station_lon_deg=lon0)
    pd.DataFrame(entries).to_csv(out/'sample_index.csv', index=False)
    metadata = dict(config=str(args.config.resolve()), station=dict(lat_deg=lat0, lon_deg=lon0),
                    requested_region_size_km=args.size_km, grid=args.grid, variable='tbb_13',
                    units='K', input_files=records, labels=label_records,
                    history_minutes=args.history_minutes, lead_minutes=args.lead_minutes,
                    task='Future station-grid rain rate, not path-mean rain or a rain map',
                    notes=['Native cloud grid retained; no temporal interpolation or amplitude normalization.',
                           'Geographic bounding box approximately matches the requested size.',
                           'Nominal scan times used; pixel scan times and cloud parallax are not corrected.',
                           'Operational availability latency is not verified; this is an offline sample.',
                           'Overlapping windows are for pipeline validation; do not randomly split them for evaluation.'])
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = (len(times)+3)//4
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5*rows), squeeze=False, constrained_layout=True)
    lo, hi = float(np.nanmin(cloud)), float(np.nanmax(cloud))
    for ax, time, frame in zip(axes.flat, times, cloud):
        plot = ax.pcolormesh(longitude, latitude, frame, cmap='viridis', vmin=lo, vmax=hi, shading='auto')
        ax.plot(lon0, lat0, 'r+', markersize=9)
        ax.set(title=f'{time:%H:%M} UTC', xlabel='Longitude', ylabel='Latitude')
    for ax in list(axes.flat)[len(times):]:
        ax.set_axis_off()
    fig.colorbar(plot, ax=list(axes.flat), label='B13 brightness temperature (K)')
    fig.suptitle(f'{start:%Y-%m-%d}: B13 cloud sequence; red crosses mark the station')
    fig.savefig(out/'cloud_sequence.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot([pd.Timestamp(e['target_time_utc']).strftime('%H:%M') for e in entries], labels, 'o-')
    ax.set(xlabel='Target time (UTC)', ylabel='BOM rain rate (mm/h)',
           title='Reference labels at the station (not predictions)')
    ax.grid(alpha=0.3)
    fig.savefig(out/'rain_labels.png', dpi=150)
    plt.close(fig)
    print(f'Output: {out}\nCloud shape: {cloud.shape}\nSamples: {len(entries)}')


if __name__ == '__main__':
    main()
