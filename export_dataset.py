"""Export single-receiver observations separately from synthetic future targets.

This is a data contract, not a trained forecasting algorithm. It deliberately
does not import run_rain or read any simultaneous clear-control CSI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import capture_features


def event_group(configuration, metadata):
    # Changing noise seed, receiver or satellite does not create a new storm.
    weather = dict(configuration["weather"])
    reference = weather.get("reference_time_utc") or metadata["resolved_start_time_utc"]
    weather["reference_time_utc"] = reference
    identity = dict(weather=weather, station={k: configuration["station"][k]
                    for k in ("lat_deg", "lon_deg")})
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def make_future_labels(index, propagation, horizon_s):
    if not np.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("Forecast horizon must be finite and nonnegative")
    truth = propagation.copy()
    truth["time_utc"] = pd.to_datetime(truth.time_utc, utc=True, format="ISO8601")
    keys = ["norad_id", "pass_id", "time_utc"]
    if truth.duplicated(keys).any():
        raise ValueError("Duplicate truth keys")
    fields = {"rain_rate_mm_h": "station_rain_rate_mm_h",
              "rain_effective_mm_h": "path_equivalent_rain_rate_mm_h",
              "rain_db": "center_frequency_rain_attenuation_db"}
    missing = set(fields) - set(truth)
    if missing:
        raise ValueError("Missing explicit rainfall truth: " + str(sorted(missing)))
    labels = index[["sample_id", "norad_id", "pass_id", "time_utc"]].copy()
    labels["target_time_utc"] = pd.to_datetime(labels.time_utc, utc=True, format="ISO8601") + pd.to_timedelta(horizon_s, unit="s")
    duration = index.get("capture_duration_s", pd.Series(np.zeros(len(index)), index=index.index)).to_numpy(float)
    if horizon_s > 0 and np.any(horizon_s < duration):
        raise ValueError("Forecast target precedes completion of the input capture")
    labels["lead_after_capture_end_s"] = horizon_s - duration
    lookup = truth[keys + list(fields)].rename(columns={"time_utc": "target_time_utc", **fields})
    labels = labels.merge(lookup, on=["norad_id", "pass_id", "target_time_utc"], how="left", validate="many_to_one")
    labels["target_available"] = labels[list(fields.values())].notna().all(axis=1)
    labels["forecast_horizon_s"] = horizon_s
    # Never interpolate across pass gaps or manufacture future labels.
    return labels.drop(columns=["norad_id", "pass_id", "time_utc"])


def export_dataset(root, satellite, output, horizon_s):
    root, output = Path(root), Path(output)
    run = json.loads((root / "run_metadata.json").read_text())
    group = event_group(run["configuration"], run["geometry_metadata"])
    configuration_id = hashlib.sha256(json.dumps(run["configuration"], sort_keys=True, default=str).encode()).hexdigest()
    manifest = pd.read_csv(root / "passes" / satellite / "captures.csv", keep_default_na=False)
    c = run["configuration"]
    if c["ofdm"].get("profile") == "starlink_public_v1":
        duration = 1 / 750
    else:
        fs = float(c["link"]["bandwidth_hz"])
        duration = c["ofdm"]["symbols"] * (round(fs / c["link"]["subcarrier_spacing_hz"]) + c["ofdm"]["cp_samples"]) / fs
    observations, indices = [], []
    for _, row in manifest.iterrows():
        sample_id = hashlib.sha256((configuration_id + ":" + group + ":" + str(row.norad_id) + ":" + row.time_utc).encode()).hexdigest()
        indices.append(dict(sample_id=sample_id, time_utc=row.time_utc, norad_id=row.norad_id,
                            pass_id=row.pass_id, event_group=group, capture_relative_path=row.rain_capture,
                            capture_duration_s=duration,
                            observation_available_at_utc=(pd.Timestamp(row.time_utc)+pd.to_timedelta(duration,unit="s")).isoformat()))
        observed = dict(sample_id=sample_id, elevation_deg=row.elevation_deg,
                        slant_range_km=row.slant_range_km, receiver_status="unavailable", receiver_error="")
        # Only the rainy/live-equivalent receive channel is available here.
        if row.generation_status == "generated":
            try:
                observed.update(capture_features(root / row.rain_capture))
                observed["receiver_status"] = "ok"
            except (OSError, ValueError, KeyError) as error:
                observed["receiver_status"] = "failed"
                observed["receiver_error"] = str(error)
        else:
            observed["receiver_error"] = str(row.error)
        observations.append(observed)
    index = pd.DataFrame(indices)
    if index.sample_id.duplicated().any():
        raise ValueError("Duplicate observations in manifest")
    labels = make_future_labels(index, pd.read_parquet(root / "propagation.parquet"), horizon_s)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(observations).to_parquet(output / "observations.parquet", index=False)
    index.to_parquet(output / "index.parquet", index=False)
    labels.to_parquet(output / "targets.parquet", index=False)
    info = dict(sample_count=len(index), receiver_success_count=sum(r["receiver_status"] == "ok" for r in observations),
        target_available_count=int(labels.target_available.sum()), forecast_horizon_s=horizon_s,
        task_type="forecasting" if horizon_s > 0 else "contemporaneous_estimation",
        event_group=group, independent_weather_event_groups=1,
        validation_split_ready=False,
        split_rule="Hold out entire independent weather events; all satellites, seeds and overlapping windows of an event stay together. One run cannot establish generalization.",
        feature_sources=["rain_channel_receiver_CSI", "orbit_geometry"],
        excluded_from_features=["simultaneous_clear_control", "true_rain", "net_loss", "true_clocks", "future_observations"],
        calibration_assumption="Exact logged AGC and stable nominal pilot scale; uncalibrated SDR counts require separate calibration.",
        time_convention="Horizon is measured from frame start. Features are available after frame end; targets also record remaining lead time. Processing latency is not included.",
        labels="Synthetic truth at exact requested future timestamp within same pass; unavailable targets retained, no gap interpolation.")
    (output / "dataset_contract.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--satellite", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizon-seconds", type=float, required=True)
    args = p.parse_args()
    print(json.dumps(export_dataset(args.root, args.satellite, args.output, args.horizon_seconds), indent=2))


if __name__ == "__main__":
    main()
