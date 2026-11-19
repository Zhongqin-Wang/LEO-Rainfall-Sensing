import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.propagation import propagation_table, compute_ground_reflection, channel_taps, two_ray_doppler
from src.waveform import make_frame
from src.clocks import apply_motion, receiver
from src.orbit import load_orbit_batch
from src.orbit_data import configured_orbit_path, sha256_file
from src.earth_orientation import ensure_earth_orientation
from src.orbit_prediction import predict_orbit
from src.weather import KEYS, clear_weather, retrieval_config
from src.bandlimited import sample_at
from src.reflection import validate_polarization
from src.receiver_noise import case_receiver_settings
from src.rain_spectrum import row_rain_spectrum, row_rain_phase, apply_spectral_loss
from src.provenance import software_provenance

ROOT = Path(__file__).resolve().parent
def reflection_geometry_from_row(row):
    sat_xyz = row[
        ["sat_x_km", "sat_y_km", "sat_z_km"]
    ].to_numpy(dtype=float)

    station_xyz = row[
        ["station_x_km", "station_y_km", "station_z_km"]
    ].to_numpy(dtype=float)

    result = {
        "satellite_xyz_km": sat_xyz,
        "station_xyz_km": station_xyz,
    }
    if all(key in row for key in ("sat_vx_km_s", "sat_vy_km_s", "sat_vz_km_s")):
        result["satellite_velocity_km_s"] = row[["sat_vx_km_s", "sat_vy_km_s", "sat_vz_km_s"]].to_numpy(dtype=float)
    return result

def generate_frame_pair(row, before_row, after_row, config, prediction_batch,
                        out, relative_dir, frame_seed):
    profile = config["ofdm"].get("profile", "custom_ofdm")
    if profile == "starlink_public_v1":
        from src.starlink_link import generate_starlink_pair
        return generate_starlink_pair(row, before_row, after_row, config, prediction_batch,
                                      out, relative_dir, frame_seed)
    if profile != "custom_ofdm":
        raise ValueError(f"Unknown waveform profile: {profile}")
    if config['propagation'].get('paper_channel', {}).get('enabled', False):
        raise ValueError("The moving-paper ray channel currently requires starlink_public_v1")
    from src.receive_array import channel_count
    if channel_count(config) != 1:
        raise ValueError('Multi-channel reception requires the public Starlink S/X/R model')
    validate_polarization(config)
    satellite_index = int(row["satellite_index"])
    fs = float(config["link"]["bandwidth_hz"])
    spacing = float(config["link"]["subcarrier_spacing_hz"])
    nfft = round(fs/spacing)
    if not np.isclose(nfft*spacing,fs):
        raise ValueError("Sample rate must be an integer multiple of subcarrier spacing")
    cp = int(config["ofdm"].get("cp_samples",32))
    if not 0 <= cp < nfft:
        raise ValueError("CP must be fixed in [0,nfft)")
    reflection = None
    carrier = float(config["link"]["carrier_hz"])
    direct_rate = float(row["slant_range_rate_km_s"])*1000
    reflected_rate = direct_rate
    if config["propagation"]["enabled"].get("ground_reflection",False):
        reflection = compute_ground_reflection(reflection_geometry_from_row(row),config,fs)
        if reflection["delay_samples"] > cp:
            raise ValueError("Reflection exceeds the fixed waveform CP; configure an ISI experiment explicitly")
        if "reflected_range_rate_m_s" in reflection:
            reflected_rate = reflection["reflected_range_rate_m_s"]
        else:
            dt = (after_row.time_utc-before_row.time_utc).total_seconds()
            if dt > 0:
                a = compute_ground_reflection(reflection_geometry_from_row(before_row),config,fs)
                b = compute_ground_reflection(reflection_geometry_from_row(after_row),config,fs)
                reflected_rate += (b["excess_path_m"]-a["excess_path_m"])/dt
    motion = config["ofdm"].get("motion_enabled",True)
    fd,fr = two_ray_doppler(direct_rate,reflected_rate,carrier) if motion else (0.,0.)
    grid,mask,tx = make_frame(nfft,cp,config["ofdm"]["symbols"],config["ofdm"]["pilot_spacing"],frame_seed)
    power = float(config["ofdm"]["tx_power_w"])
    if not np.isfinite(power) or power <= 0:
        raise ValueError("TX power must be positive")
    scale = np.sqrt(power/np.mean(abs(tx)**2))
    grid *= scale;tx *= scale
    known = np.zeros_like(grid);known[mask] = grid[mask]
    prediction = predict_orbit(prediction_batch.models[satellite_index],row.time_utc.to_pydatetime(),
                               config["station"],carrier,config.get("orbit_prediction")) if motion else {
                                   "predicted_doppler_hz":0.,"predicted_time_scale":1.}
    # Preserve enough leading samples for guarded pre-frame noise estimation.
    minimum_prefix = max(nfft, 384)
    prefix = int(np.random.default_rng(frame_seed+100).integers(minimum_prefix, minimum_prefix+nfft))
    for case_index,(case,loss) in enumerate([("rain",float(row.net_loss_db)),("reference",float(row.reference_loss_db))]):
        rain_spectrum = row_rain_spectrum(row, config, np.fft.fftfreq(nfft, 1 / fs), case)
        rain_phase = row_rain_phase(row, config, np.fft.fftfreq(nfft, 1 / fs), case)
        rain_rate = float(row.get("rain_phase_rate_rad_s", 0.)) if case == "rain" else 0.
        center_rain = float(row.rain_db) if case == "rain" else 0.
        spectral_correction = rain_spectrum - center_rain
        amplitude = 10**(-loss/20)
        channel_tx = apply_spectral_loss(tx, fs, np.fft.fftfreq(nfft, 1 / fs), spectral_correction,
                                         phase_rad=rain_phase)
        direct = amplitude*channel_tx
        moving = apply_motion(direct,fs,carrier,fd) if motion else direct.copy()
        if reflection is not None:
            delay = reflection["delay_samples"]
            coefficient = 10**(reflection["gain_db"]/20)*np.exp(1j*np.deg2rad(reflection["phase_deg"]))
            reflected = amplitude*coefficient*sample_at(channel_tx,np.arange(len(tx))-delay)
            moving += apply_motion(reflected,fs,carrier,fr,delay) if motion else reflected
        moving *= np.exp(1j*rain_rate*np.arange(len(moving))/fs)
        capture = np.pad(moving,(prefix,nfft))
        noise_seed = np.random.SeedSequence([int(frame_seed),case_index,91])
        clock_mode=config["experiment"].get("reference_clock_mode","independent")
        if clock_mode not in ("shared","independent"):
            raise ValueError("reference_clock_mode must be shared or independent")
        rx,truth = receiver(capture,fs,config["clock"],case_receiver_settings(row, config["receiver"], case),
                            frame_seed+1+(case_index*131 if clock_mode=="independent" else 0),noise_seed=noise_seed)
        directory = Path(out)/case/relative_dir;directory.mkdir(parents=True,exist_ok=True)
        metadata = dict(satellite_index=satellite_index,satellite_name=str(row.satellite_name),
                        norad_id=int(prediction_batch.models[satellite_index].satnum),pass_id=str(row.pass_id),
                        time_utc=row.time_utc.isoformat(),elevation_deg=float(row.elevation_deg),
                        slant_range_km=float(row.slant_range_km),complete_pass=bool(row.complete_pass),
                        case=case,frame_seed=int(frame_seed),frame_duration_s=len(tx)/fs,
                        configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest(),
                        paired_clocks=clock_mode,
                        sampling_mode="independent_custom_OFDM_frames",waveform_profile="custom_ofdm",
                        paired_noise="independent_thermal_noise",ground_reflection=reflection is not None)
        metadata.update(reference_role="simultaneous_synthetic_counterfactual_not_a_field_observation",
                        rain_frequency_model=config["propagation"].get("rain_frequency_model", "center_frequency_flat"),
                        rain_phase_model=config["propagation"].get("rain_phase_model", "none"),
                        rain_phase_scope="coherent_forward_propagation_no_droplet_microdoppler",
                        rain_path_assumption="shared_rain_transfer_for_LOS_and_local_ground_reflection",
                        noise_model=truth["noise_model"])
        (directory/"capture_metadata.json").write_text(json.dumps(metadata,indent=2))
        (directory/"simulation_truth.json").write_text(json.dumps(dict(truth,doppler_hz=fd,reflection=reflection,
            rain_phase_center_rad=float(row.get("rain_phase_rad", 0.)) if case == "rain" else 0.,
            rain_phase_rate_rad_s=rain_rate, rain_excess_doppler_hz=rain_rate/(2*np.pi),
            rain_phase_rate_basis="finite_difference_of_unwrapped_phase_at_geometry_UTC_samples"),indent=2))
        (directory/"receiver_log.json").write_text(json.dumps({"agc_gain_linear":truth["agc_gain_linear"]}))
        (directory/"retrieval_config.json").write_text(json.dumps(retrieval_config(row,config),indent=2))
        np.save(directory/"rx_iq.npy",rx)
        np.savez_compressed(directory/"frame_config.npz",waveform_profile="custom_ofdm",sample_rate_hz=fs,
                            nfft=nfft,cp_samples=cp,symbols=len(grid),pilot_mask=mask,pilot_values=known)
        np.savez_compressed(directory/"orbit_prediction.npz",**prediction)
        taps,h,_,_ = channel_taps(nfft,cp,loss,reflection)
        h *= 10 ** (-spectral_correction / 20)*np.exp(1j*rain_phase)
        taps = np.fft.ifft(h)
        np.savez_compressed(directory/"Hprop.npz",Hprop=h,taps=taps,net_loss_db=loss,
                            impulse_response_kind="periodic_DFT_not_causal_FIR",
                            rain_effective_mm_h=float(row.get("rain_effective_mm_h",np.nan)) if case=="rain" else 0.,
                            rain_db=center_rain, rain_db_by_frequency=rain_spectrum,
                            rain_phase_rad_by_frequency=rain_phase,
                            rain_phase_rate_rad_s=rain_rate,
                            frequency_offset_hz=np.fft.fftfreq(nfft, 1 / fs))
        for name in ("csi_pilots_corrected.npz","csi.png","rain_estimate.npz","rain_estimate.png"):
            (directory/name).unlink(missing_ok=True)
        print(f"Saved {case} custom OFDM capture: {directory}",flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/example.yaml')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--satellite", type=int, help="Generate only this snapshot satellite index")
    selection.add_argument("--norad-id", type=int, help="Generate only this stable NORAD catalog ID")
    selection.add_argument("--all-satellites", action="store_true", help="Generate the full visible constellation")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--geometry-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("propagation") and config.get("station"):
        validate_polarization(config)
    if config.get("ofdm", {}).get("profile") == "starlink_public_v1":
        from src.starlink_waveform import validate_config
        validate_config(config)
    orbit_path = configured_orbit_path(ROOT, config["geometry"])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    geometry_dir = args.geometry_dir

    metadata = json.loads((geometry_dir/'geometry_metadata.json').read_text())
    expected = json.loads(json.dumps({key: config[key] for key in ('geometry', 'station', 'weather')}, default=str))
    if metadata['config'] != expected or metadata['carrier_hz'] != float(config['link']['carrier_hz']):
        raise ValueError('Geometry/weather config changed; rerun run_geometry.py first')
    recorded_hash = metadata.get("orbit_sha256", metadata.get("tle_sha256"))
    if not recorded_hash or recorded_hash != sha256_file(orbit_path):
        raise ValueError("Orbit snapshot changed after geometry generation; rerun run_geometry.py first")
    eop_config = config["geometry"].get("eop_path")
    eop = ensure_earth_orientation(metadata["resolved_start_time_utc"], ROOT / eop_config if eop_config else None)
    if metadata.get("earth_orientation", {}).get("sha256") != eop["sha256"]:
        raise ValueError("Earth-orientation snapshot changed or is unrecorded; rerun run_geometry.py first")
    prediction_batch = load_orbit_batch(orbit_path)
    (out/"run_metadata.json").write_text(json.dumps(dict(configuration=config,geometry_metadata=metadata,
        software=software_provenance(ROOT)),indent=2,default=str))

    geo = pd.read_parquet(geometry_dir/'geometry.parquet')
    weather = pd.read_parquet(geometry_dir/'weather_truth.parquet')
    # Older geometry files mixed truth into geometry. Keep it only in the
    # explicitly truth-bearing propagation table used by the signal generator.
    geo = geo.drop(columns=["rain_rate_mm_h"], errors="ignore")
    # Selection depends only on geometry. Avoid computing the new drop physics
    # for every visible satellite when the requested experiment uses one link.
    if args.satellite is not None:
        geo = geo[geo["satellite_index"] == args.satellite]
    elif args.norad_id is not None:
        if "norad_id" not in geo:
            raise ValueError("NORAD selection requires regenerated geometry with norad_id")
        geo = geo[geo["norad_id"] == args.norad_id]
    elif not args.all_satellites and not geo.empty:
        candidates = geo[geo["complete_pass"]]
        if candidates.empty:
            candidates = geo
        selected = candidates.loc[candidates["elevation_deg"].idxmax()]
        geo = geo[geo["satellite_index"] == int(selected["satellite_index"])]
    if geo.empty:
        raise ValueError("No geometry samples for the selected satellite")

    prop = propagation_table(geo, weather, float(config['link']['carrier_hz']),
                             config['propagation'], config['experiment']['seed'])

    reference_weather = clear_weather(weather)

    reference_prop = propagation_table(
        geo,
        reference_weather,
        float(config["link"]["carrier_hz"]),
        config["propagation"],
        config["experiment"]["seed"],
    )

    if geo.empty:
        raise ValueError('No valid geometry samples available')
    all_links = geo.merge(
        prop,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    truth_columns = KEYS + [k for k in ("rain_rate_mm_h", "rain_rate_samples_mm_h") if k in weather]
    all_links = all_links.merge(weather[truth_columns], on=KEYS, how="left", validate="one_to_one")
    reference_loss = reference_prop[
        KEYS + ["net_loss_db"]
        ].rename(columns={"net_loss_db": "reference_loss_db"})

    all_links = all_links.merge(
        reference_loss,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if args.satellite is not None:
        all_links = all_links[all_links["satellite_index"] == args.satellite]
    elif args.norad_id is not None:
        if "norad_id" not in all_links:
            raise ValueError("NORAD selection requires regenerated geometry with norad_id")
        all_links = all_links[all_links["norad_id"] == args.norad_id]
    elif not args.all_satellites:
        candidates = all_links[all_links["complete_pass"]]
        if candidates.empty:
            candidates = all_links
        selected = candidates.loc[candidates["elevation_deg"].idxmax()]
        selected_index = int(selected["satellite_index"])
        print(f"Selected satellite_{selected_index:04d}: {selected['satellite_name']} "
              "(highest-elevation eligible pass; use --all-satellites for bulk IQ)")
        all_links = all_links[all_links["satellite_index"] == selected_index]
    if all_links.empty:
        raise ValueError("No geometry samples for the selected satellite")
    all_links.to_parquet(
        out / 'propagation.parquet',
        index=False
    )
    failure_count = 0
    for satellite_index, satellite_geo in all_links.groupby("satellite_index"):
        records = []
        satellite_name = f"satellite_{int(satellite_index):04d}"
        manifest_dir = out / "passes" / satellite_name
        manifest_dir.mkdir(parents=True,exist_ok=True)
        (manifest_dir/"rain_timeseries.csv").unlink(missing_ok=True)
        for old_plot in manifest_dir.glob("rain_pass_*.png"):
            old_plot.unlink()
        for pass_id, pass_geo in satellite_geo.groupby("pass_id", sort=False):
            pass_geo = pass_geo.sort_values("time_utc").reset_index(drop=True)
            pass_start = pass_geo.iloc[0]["time_utc"]
            pass_dir = "pass_" + pass_start.strftime("%Y%m%dT%H%M%S%fZ")
            for position, row in pass_geo.iterrows():
                time_dir = "time_" + row["time_utc"].strftime("%Y%m%dT%H%M%S%fZ")
                relative_dir = Path(satellite_name) / pass_dir / time_dir
                record = {
                    "satellite_index": int(satellite_index),
                    "satellite_name": str(row["satellite_name"]),
                    "norad_id": int(prediction_batch.models[int(satellite_index)].satnum),
                    "pass_id": str(pass_id),
                    "time_utc": row["time_utc"].isoformat(),
                    "elapsed_s": (row["time_utc"] - pass_start).total_seconds(),
                    "elevation_deg": float(row["elevation_deg"]),
                    "slant_range_km": float(row["slant_range_km"]),
                    "complete_pass": bool(row["complete_pass"]),
                    "true_rain_db": float(row["rain_db"]),
                    "rain_capture": str(Path("rain") / relative_dir),
                    "reference_capture": str(Path("reference") / relative_dir),
                    "generation_status": "pending",
                    "error": "",
                }
                timestamp = int(row["time_utc"].value)
                frame_seed = int(np.random.SeedSequence([
                    int(config["experiment"]["seed"]),int(satellite_index),
                    timestamp & 0xffffffff,timestamp >> 32,
                ]).generate_state(1)[0])
                try:
                    generate_frame_pair(
                        row,pass_geo.iloc[max(position-1,0)],pass_geo.iloc[min(position+1,len(pass_geo)-1)],
                        config,prediction_batch,out,relative_dir,frame_seed,
                    )
                    record["generation_status"] = "generated"
                except (ValueError,RuntimeError,FloatingPointError,OSError) as error:
                    failure_count += 1
                    record["generation_status"] = "generation_failed"
                    record["error"] = str(error)
                    print(f"Frame failed at {row['time_utc']}: {error}")
                records.append(record)
                pd.DataFrame(records).to_csv(manifest_dir / "captures.csv",index=False)
        manifest_dir = out / "passes" / satellite_name
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "rain_timeseries.csv").unlink(missing_ok=True)
        for old_plot in manifest_dir.glob("rain_pass_*.png"):
            old_plot.unlink()
        manifest = pd.DataFrame(records)
        manifest.to_csv(manifest_dir / "captures.csv", index=False)
        print(manifest["generation_status"].value_counts().to_string())
        print(f"Capture manifest: {manifest_dir / 'captures.csv'}")
    if failure_count:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
