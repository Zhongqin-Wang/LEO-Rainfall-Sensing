import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time

from src.weather import build_weather
from src.reflection import validate_polarization
from src.orbit import compute_geometry, load_orbit_batch, sample_times, OrbitHistory, add_beam_visibility
from src.models import TLEBatch
from src.earth_orientation import ensure_earth_orientation
from sgp4.api import SatrecArray
from src.orbit_data import (configured_orbit_path, eligible_orbits, read_provenance,
                            sha256_file, update_starlink)

PROJECT_ROOT = Path(__file__).resolve().parent

def _parse_start_time(value):
    if value in (None, "", "now"):
        return datetime.now(timezone.utc).replace(microsecond=0)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("geometry.start_time_utc must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _select_validation_pass(geometry: pd.DataFrame) -> pd.DataFrame:
    if geometry.empty:
        raise ValueError("No geometry samples available")
    candidates = geometry[geometry["complete_pass"]]
    if candidates.empty:
        candidates = geometry
    peak_rows = candidates.groupby("pass_id")["elevation_deg"].max()
    selected_pass = peak_rows.idxmax()
    return geometry[geometry["pass_id"] == selected_pass].sort_values("time_utc")


def validate_pass(pass_data: pd.DataFrame) -> dict:
    ranges = pass_data["slant_range_km"].to_numpy()
    rates = pass_data["slant_range_rate_km_s"].to_numpy()
    doppler = pass_data["doppler_hz"].to_numpy()
    closest_index = int(np.argmin(ranges))

    closest_is_interior = 0 < closest_index < len(pass_data) - 1
    rate_changes_sign = bool(
        np.any(rates[: closest_index + 1] < 0.0)
        and np.any(rates[closest_index:] > 0.0)
    )
    doppler_is_finite = bool(np.all(np.isfinite(doppler)))
    max_doppler_step_hz = float(np.max(np.abs(np.diff(doppler)))) if len(doppler) > 1 else 0.0

    return {
        "pass_id": str(pass_data["pass_id"].iloc[0]),
        "satellite_name": str(pass_data["satellite_name"].iloc[0]),
        "sample_count": int(len(pass_data)),
        "closest_approach_is_interior": bool(closest_is_interior),
        "range_rate_changes_sign": rate_changes_sign,
        "doppler_all_finite": doppler_is_finite,
        "max_adjacent_doppler_step_hz": max_doppler_step_hz,
        "minimum_slant_range_km": float(ranges[closest_index]),
        "maximum_elevation_deg": float(pass_data["elevation_deg"].max()),
    }


def plot_pass(pass_data: pd.DataFrame, output_path: Path) -> None:
    elapsed_s = (
        pass_data["time_utc"] - pass_data["time_utc"].iloc[0]
    ).dt.total_seconds().to_numpy()
    series = (
        ("elevation_deg", "Elevation (deg)"),
        ("slant_range_km", "Slant range (km)"),
        ("slant_range_rate_km_s", "Slant range rate (km/s)"),
        ("doppler_hz", "Doppler (Hz)"),
    )

    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for axis, (column, label) in zip(axes, series):
        axis.plot(elapsed_s, pass_data[column].to_numpy(), linewidth=1.7)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        if column in ("slant_range_rate_km_s", "doppler_hz"):
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

    axes[-1].set_xlabel("Seconds since AOS sample")
    figure.suptitle(
        f"{pass_data['satellite_name'].iloc[0]} pass {pass_data['pass_id'].iloc[0]}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "example.yaml",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    geometry_config = config["geometry"]
    station_config = config["station"]
    validate_polarization(config)

    start_time = _parse_start_time(geometry_config.get("start_time_utc"))
    datetimes, obstime = sample_times(
        start_time=start_time,
        duration_hours=float(geometry_config["duration_hours"]),
        step_seconds=float(geometry_config["step_seconds"]),
    )
    eop_path = PROJECT_ROOT / geometry_config["eop_path"] if geometry_config.get("eop_path") else None
    eop_metadata = ensure_earth_orientation(obstime, eop_path)
    orbit_path = configured_orbit_path(PROJECT_ROOT, geometry_config)
    # Fixed historical windows use the selected saved snapshot, never today's download.
    realtime = geometry_config.get("start_time_utc") in (None, "", "now")
    if realtime and geometry_config.get("auto_update_orbits", False):
        if orbit_path.suffix.lower() != ".json":
            raise ValueError("Automatic Starlink updates require a .json geometry.orbit_path")
        update = update_starlink(orbit_path)
        print(f"Starlink data: {update['update_status']}; {update['satellite_count']} satellites", flush=True)
    history = OrbitHistory(orbit_path) if geometry_config.get("orbit_selection") == "latest_available" else None
    batch = history.batch if history else load_orbit_batch(orbit_path)
    provenance = read_provenance(orbit_path)
    orbit_hash = sha256_file(orbit_path)
    if history:
        eligible = history.select(datetimes[0], geometry_config.get('max_epoch_distance_days', 3.))
        orbit_report = dict(policy='latest_available', publication_proxy='CREATION_DATE',
                            source_records=len(batch.models), eligible_count_at_start=len(eligible),
                            excluded_epoch_count=len(batch.models)-len(eligible))
    else:
        eligible, orbit_report = eligible_orbits(
            batch, datetimes[0], datetimes[-1], geometry_config.get("max_epoch_distance_days", 3.0))
    batch_size = int(geometry_config.get("satellite_batch_size", 128))
    if batch_size < 1:
        raise ValueError("geometry.satellite_batch_size must be positive")
    print(f"Epoch check: {len(eligible)} eligible; {orbit_report['excluded_epoch_count']} excluded", flush=True)
    station = EarthLocation.from_geodetic(
        lon=float(station_config["lon_deg"]) * u.deg,
        lat=float(station_config["lat_deg"]) * u.deg,
        height=(float(station_config["height_m"])
    + float(station_config["antenna_height_m"])) * u.m,
    )
#Ground receiver configuration
    frames = []
    if history:
        counts = []
        for ti, instant in enumerate(datetimes):
            selected = history.select(instant, geometry_config.get('max_epoch_distance_days', 3.))
            counts.append(len(selected))
            for offset in range(0, len(selected), batch_size):
                indices = selected[offset:offset+batch_size]
                models = [batch.models[int(i)] for i in indices]
                part = compute_geometry(TLEBatch(batch.names[indices], models, SatrecArray(models)),
                    station, [instant], Time([instant]), float(config['link']['carrier_hz']),
                    float(geometry_config['min_elevation_deg']), satellite_indices=indices, allow_empty=True)
                if not part.empty:
                    ri = part['satellite_index'].to_numpy(int)
                    part['orbit_epoch_utc'] = history.epochs[ri]
                    part['orbit_creation_utc'] = history.created[ri]
                    frames.append(part)
            if ti % 30 == 0 or ti == len(datetimes)-1:
                print(f'History geometry: {ti+1}/{len(datetimes)} times, {len(selected)} eligible satellites', flush=True)
        orbit_report.update(min_eligible_satellites=min(counts), max_eligible_satellites=max(counts))
    else:
        for offset in range(0, len(eligible), batch_size):
            indices = eligible[offset:offset + batch_size]
            models = [batch.models[int(index)] for index in indices]
            part = compute_geometry(
                batch=TLEBatch(batch.names[indices], models, SatrecArray(models)),
                station_location=station,
                datetimes=datetimes,
                obstime=obstime,
                carrier_hz=float(config["link"]["carrier_hz"]),
                min_elevation_deg=float(geometry_config["min_elevation_deg"]),
                rain_rate_mm_h=float(config["weather"]["rain_rate_mm_h"]),
                satellite_indices=indices,
                allow_empty=True,
            )
            if not part.empty:
                frames.append(part)
            print(f"Orbit propagation: {min(offset + batch_size, len(eligible))}/{len(eligible)} satellites", flush=True)
    if not frames:
        raise RuntimeError("No visible samples found; extend duration or lower min_elevation_deg")
    geometry = pd.concat(frames, ignore_index=True)
    geometry = add_beam_visibility(geometry, config['propagation'])
    if history:
        # A record update is not a new satellite or a new visible pass.
        parts = []
        for norad, group in geometry.groupby('norad_id'):
            group = group.sort_values('time_utc').copy()
            breaks = group.time_utc.diff().dt.total_seconds().gt(float(geometry_config['step_seconds'])*1.01)
            group['pass_id'] = [f'{norad}-{n:03d}' for n in breaks.cumsum()]
            for _, segment in group.groupby('pass_id'):
                group.loc[segment.index, 'complete_pass'] = (segment.time_utc.min() > pd.Timestamp(datetimes[0]) and segment.time_utc.max() < pd.Timestamp(datetimes[-1]))
            parts.append(group)
        geometry = pd.concat(parts, ignore_index=True)


    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "geometry.parquet"
    candidates = geometry[geometry.in_rx_beam].copy()
    candidates.to_csv(output_dir/'observation_candidates.csv', index=False)
    intervals = []
    for norad, group in candidates.groupby('norad_id'):
        group = group.sort_values('time_utc')
        split = group.time_utc.diff().dt.total_seconds().gt(float(geometry_config['step_seconds'])*1.01).cumsum()
        for _, segment in group.groupby(split):
            intervals.append(dict(norad_id=int(norad), first_sample_utc=segment.time_utc.min(),
                last_sample_utc=segment.time_utc.max(), sampled_points=len(segment),
                max_elevation_deg=float(segment.elevation_deg.max())))
    pd.DataFrame(intervals, columns=['norad_id','first_sample_utc','last_sample_utc','sampled_points','max_elevation_deg']).to_csv(output_dir/'observation_intervals.csv', index=False)
    print(f'Within RX half-power cone: {len(candidates)} sampled observations', flush=True)

    weather_config = dict(config["weather"])
    if not weather_config.get("reference_time_utc"):
        weather_config["reference_time_utc"] = start_time.isoformat()
    weather = build_weather(geometry, weather_config, station_config)
    geometry.to_parquet(parquet_path, index=False, engine="pyarrow")
    weather.to_parquet(output_dir / "weather_truth.parquet", index=False)

    metadata = {
        "config": {key: config[key] for key in ("geometry", "station", "weather")},
        "carrier_hz": float(config["link"]["carrier_hz"]),
        "resolved_start_time_utc": start_time.isoformat(),
        "tle_sha256": orbit_hash,  # Retained for compatibility with old results.
        "orbit_sha256": orbit_hash,
        "orbit_path": str(orbit_path.resolve()),
        "orbit_provenance": provenance,
        "orbit_epoch_check": orbit_report,
        "earth_orientation": eop_metadata,
        "geometry_contains_rainfall_truth": False,
        "beam_config": config['propagation']['beam'],
        "observation_selection_note": 'Sampled half-power cone only; not continuous boundaries, link budget, or receiver acquisition guarantee.',

        "weather_reference_time_utc": weather_config["reference_time_utc"],
    }
    if config["weather"].get("model") == "bom":
        from src.bom_rain import fingerprint
        metadata["bom_data_sha256"] = fingerprint(config["weather"])
    (output_dir / "geometry_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (output_dir / "orbit_epoch_check.json").write_text(json.dumps(orbit_report, indent=2))

    plot_path = output_dir / "geometry_pass.png"
    validation_pass = _select_validation_pass(geometry)
    validation = validate_pass(validation_pass)
    plot_pass(validation_pass, plot_path)
    summary_path = output_dir / "geometry_validation.json"
    summary_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Visible passes: {geometry['pass_id'].nunique()}")
    print(f"Geometry rows: {len(geometry)}")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
