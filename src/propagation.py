import numpy as np
from src.reflection import compute_ground_reflection, channel_response
from itur.models import itu676, itu838, itu840
from src.weather import KEYS
from src.antenna import path_antenna_gains
from src.cloud_liquid_water import era5_column

itu676.change_version(12)
itu838.change_version(3)
itu840.change_version(7)
C = 299792458.0 #Speed of light (m/s)

def two_ray_doppler(vd_m_s, vr_m_s, carrier_hz, direction="downlink"):
    f0 = float(carrier_hz)
    if direction == "downlink":  # Satellite transmitter and ground receiver
        fd_hz = -f0 * vd_m_s / C
        fr_hz = -f0 * vr_m_s / C
    elif direction == "uplink":
        fd_hz = -f0 * vd_m_s / C
        fr_hz = -f0 * vr_m_s / C
    else:
        raise ValueError("direction must be downlink or uplink")
    return fd_hz, fr_hz

def propagation_table(geometry, weather, carrier_hz, config, seed=0):
    data = geometry.drop(columns=["rain_rate_mm_h"], errors="ignore").merge(
        weather, on=KEYS, validate="one_to_one", how="left")
    if data.rain_path_km.isna().any():
        raise ValueError("Missing weather truth for geometry samples")
    if carrier_hz <= 0 or np.any(data.slant_range_km <= 0):
        raise ValueError("Frequency and range must be positive")
    el = data.elevation_deg.to_numpy()
    if not np.isfinite(el).all() or np.any((el <= 0) | (el > 90)):
        raise ValueError("Propagation elevations must be in (0,90]")
    f = carrier_hz / 1e9  #GHz
    switches = config["enabled"]
    out = data[KEYS].copy()

    out["fspl_db"] = 20*np.log10(4*np.pi*data.slant_range_km*1000*carrier_hz/C) if switches["fspl"] else 0.

    gains = path_antenna_gains(
        data[['sat_x_km', 'sat_y_km', 'sat_z_km']].to_numpy(float)*1000,
        data[['station_x_km', 'station_y_km', 'station_z_km']].to_numpy(float)*1000, config)
    out['antenna_gain_db'] = gains['combined_gain_dbi']
    for key in ('tx_gain_dbi', 'rx_gain_dbi', 'tx_offaxis_deg', 'rx_offaxis_deg'):
        out[key] = gains[key]

    gas = config["gas"]
    gas_loss = np.zeros(len(data))
    if switches["gas"]:
        mode = gas.get("mode","auto")
        if mode not in ("auto","approx","exact"):
            raise ValueError("gas.mode must be auto, approx or exact")
        normal = (el>=5)&(el<90)
        if mode=="approx" and not normal.all():
            raise ValueError("Gas approximation is outside its elevation domain; use auto or exact")
        approximate = normal if mode=="auto" else np.full(len(data),mode=="approx")
        for selected,method in ((approximate,"approx"),(~approximate,"exact")):
            if selected.any():
                gas_loss[selected] = itu676.gaseous_attenuation_slant_path(
                    f,el[selected],gas["water_vapour_density_g_m3"],gas["pressure_hpa"],
                    gas["temperature_k"],mode=method).value
    out["gas_db"] = gas_loss

    cloud = config["cloud"]
    out["cloud_db"] = 0.
    if switches["cloud"]:
        if "era5_file" in cloud:
            liquid, stamp, lat, lon = era5_column(data, cloud)
            out["cloud_source_time_utc"] = stamp
            out["cloud_grid_lat_deg"] = lat
            out["cloud_grid_lon_deg"] = lon
        out["cloud_liquid_water_kg_m2"] = liquid
        out["cloud_db"] = (itu840.specific_attenuation_coefficients(f, cloud["temperature_c"])*
                           liquid/np.sin(np.deg2rad(el)))

    k, alpha = itu838.rain_specific_attenuation_coefficients(f,el,config["polarization_tilt_deg"])
    k,alpha = np.asarray(k),np.asarray(alpha)
    rates = data.rain_rate_mm_h.to_numpy(dtype=float)
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Rain rates must be finite and nonnegative")
    path = data.rain_path_km.to_numpy(dtype=float)
    if not np.isfinite(path).all() or np.any(path < 0):
        raise ValueError("Rain lengths must be finite and nonnegative")
    gamma = k*rates**alpha if switches["rain"] else np.zeros(len(data))
    out["rain_specific_db_km"] = gamma
    rain_loss = gamma*path
    if switches["rain"] and "rain_rate_samples_mm_h" in data:
        samples = np.stack(data.rain_rate_samples_mm_h.to_numpy())
        if not np.isfinite(samples).all() or np.any(samples<0):
            raise ValueError("Spatial rain samples must be finite and nonnegative")
        rain_loss = np.mean(k[:,None]*samples**alpha[:,None],axis=1)*path
    out["station_rain_specific_db_km"] = gamma
    out["rain_specific_db_km"] = np.divide(rain_loss,path,out=np.zeros_like(rain_loss),where=path>0)
    out["rain_db"] = rain_loss
    out["rain_path_km"] = path
    out["rain_effective_mm_h"] = np.where(path>0,(rain_loss/np.maximum(k*path,1e-300))**(1/alpha),0.)
    amplitude_model = config.get("rain_frequency_model", "center_frequency_flat")
    phase_model = config.get("rain_phase_model", "none")
    if amplitude_model not in ("center_frequency_flat", "itu838_frequency_dependent", "mie_spherical_mp"):
        raise ValueError("Unknown propagation.rain_frequency_model")
    if phase_model not in ("none", "mie_spherical_mp"):
        raise ValueError("Unknown propagation.rain_phase_model")
    if amplitude_model == "mie_spherical_mp" and phase_model != "mie_spherical_mp":
        raise ValueError("Full complex Mie rain requires rain_phase_model: mie_spherical_mp")
    out["rain_phase_rad"] = 0.
    if phase_model == "mie_spherical_mp":
        from src.rain_complex import complex_rain_spectrum, settings_from_config
        settings = settings_from_config(config)
        # Compute even clear controls, so invalid physics settings cannot hide.
        for i, row in data.iterrows():
            samples = row.get("rain_rate_samples_mm_h", row["rain_rate_mm_h"])
            loss_mie, phase_mie = complex_rain_spectrum([carrier_hz], float(row["rain_path_km"]),
                samples if switches["rain"] else [0.], **settings)
            out.loc[i, "rain_phase_rad"] = phase_mie[0]
            if amplitude_model == "mie_spherical_mp":
                out.loc[i, "rain_db"] = loss_mie[0]
                out.loc[i, "rain_specific_db_km"] = loss_mie[0]/path[i] if path[i] > 0 else 0.
                station_loss, _ = complex_rain_spectrum([carrier_hz], 1.,
                    [row["rain_rate_mm_h"]] if switches["rain"] else [0.], **settings)
                out.loc[i, "station_rain_specific_db_km"] = station_loss[0]
        if amplitude_model == "mie_spherical_mp":
            out["rain_effective_mm_h"] = np.where(path > 0, data.rain_rate_mm_h.to_numpy(), 0.) if switches["rain"] else 0.
            if "rain_rate_samples_mm_h" in data and switches["rain"]:
                from scipy.optimize import brentq
                for i, row in data.iterrows():
                    samples = np.asarray(row["rain_rate_samples_mm_h"], float)
                    low, high = float(samples.min()), float(samples.max())
                    if path[i] == 0:
                        value = 0.
                    elif high == low:
                        value = low
                    else:
                        target = float(out.loc[i, "rain_db"])
                        value = brentq(lambda r: complex_rain_spectrum([carrier_hz], path[i], [r],
                            **settings)[0][0]-target, low, high)
                    out.loc[i, "rain_effective_mm_h"] = value
    # Local phase slope is estimated from actual UTC-spaced propagation samples;
    # it is a coarse-weather derivative, not raindrop micro-Doppler.
    out["rain_phase_rate_rad_s"] = 0.
    for _, group in data.groupby(["satellite_index", "pass_id"], sort=False):
        group = group.sort_values("time_utc")
        if len(group) > 1:
            times = (group.time_utc-group.time_utc.iloc[0]).dt.total_seconds().to_numpy()
            if np.any(np.diff(times) <= 0):
                raise ValueError("Rain phase samples must have increasing UTC times")
            out.loc[group.index, "rain_phase_rate_rad_s"] = np.gradient(
                out.loc[group.index, "rain_phase_rad"].to_numpy(), times)

    scint = np.zeros(len(data))
    if switches["scintillation"]:
        rng = np.random.default_rng(seed)
        tau = float(config["scintillation"]["correlation_s"])
        sigma = float(config["scintillation"]["sigma_db"])
        if tau <= 0 or sigma < 0:
            raise ValueError("Invalid scintillation parameters")
        for _, group in data.groupby("pass_id", sort=False):
            group = group.sort_values("time_utc")
            state = rng.normal(0, sigma)
            times = group.time_utc.astype("int64").to_numpy()/1e9
            for j, idx in enumerate(group.index):
                if j:
                    a = np.exp(-(times[j]-times[j-1])/tau)
                    state = a*state + np.sqrt(1-a*a)*rng.normal(0, sigma)
                scint[idx] = state
    out["scintillation_db"] = scint

    out["net_loss_db"] = out[["fspl_db", "gas_db", "cloud_db", "rain_db", "scintillation_db"]].sum(axis=1)-out.antenna_gain_db
    if not np.isfinite(out.select_dtypes("number").to_numpy()).all():
        raise ValueError("Nonfinite propagation result")
    return out

def channel_taps(nfft, cp, net_loss_db, refl_path=None):
    """Exact sampled frequency response and its periodic DFT impulse response.

    These taps are a periodic representation for inspection, not a causal
    fractional-delay FIR. Waveform propagation uses bandlimited sample delays.
    """
    if nfft < 1 or not 0 <= cp < nfft:
        raise ValueError("Invalid FFT or CP length")
    if refl_path is not None and not 0 <= refl_path["delay_samples"] <= cp:
        raise ValueError("Physical path delay exceeds the configured CP")
    direct = np.full(nfft, 10**(-float(net_loss_db)/20), dtype=complex)
    # Frequency in cycles/sample makes fs=1 exact for delays in samples.
    response = channel_response(nfft, 1., net_loss_db, refl_path)
    direct_taps = np.fft.ifft(direct)
    reflected_taps = np.fft.ifft(response-direct)
    return direct_taps+reflected_taps, response, direct_taps, reflected_taps
