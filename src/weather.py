"""Deterministic uniform, time-varying and advecting synthetic rainfall fields."""
import numpy as np
import pandas as pd
from src.reflection import station_axes

KEYS = ["time_utc", "satellite_index", "pass_id"]


def rain_paths(geometry, rain_top_km, station_height_m=0.):
    """Intersect the ITRS ray with an expanded WGS84 rain-top ellipsoid.

    The expanded semi-axes approximate a constant geodetic-height rain surface;
    receiver coordinates already include antenna height. No spherical-earth
    replacement of the station or satellite position is made.
    """
    top = float(rain_top_km)
    if not np.isfinite(top) or top < 0:
        raise ValueError("Rain top must be finite and nonnegative")
    el = geometry.elevation_deg.to_numpy(dtype=float)
    distance = geometry.slant_range_km.to_numpy(dtype=float)
    if not np.isfinite(el).all() or np.any((el <= 0) | (el > 90)) or np.any(distance <= 0):
        raise ValueError("Rain paths require positive ranges and elevations in (0,90]")
    station_columns = ["station_x_km", "station_y_km", "station_z_km"]
    satellite_columns = ["sat_x_km", "sat_y_km", "sat_z_km"]
    if all(key in geometry for key in station_columns+satellite_columns):
        station = geometry[station_columns].to_numpy(dtype=float)
        satellite = geometry[satellite_columns].to_numpy(dtype=float)
        direction = satellite-station
        norm = np.linalg.norm(direction,axis=1)
        if not np.allclose(norm,distance,rtol=1e-8,atol=1e-7):
            raise ValueError("Rain geometry range disagrees with the ITRS positions")
        direction /= norm[:,None]
        axes = np.array([6378.137+top,6378.137+top,6356.752314245+top])
        a = np.sum((direction/axes)**2,axis=1)
        b = 2*np.sum(station*direction/axes**2,axis=1)
        c = np.sum((station/axes)**2,axis=1)-1
        discriminant = np.maximum(b*b-4*a*c,0.)
        path = np.where(c < 0,-2*c/np.maximum(b+np.sqrt(discriminant),1e-300),0.)
    else:
        # Compatibility for elevation-only callers. Main geometry always uses ITRS.
        thickness = max(top-float(station_height_m)/1000,0.)
        radius = 6371.+float(station_height_m)/1000
        vertical = radius*np.sin(np.deg2rad(el))
        path = (2*radius*thickness+thickness**2)/(np.sqrt(vertical**2+2*radius*thickness+thickness**2)+vertical)
    return np.minimum(path,distance)


def _elapsed(geometry, config):
    times = pd.to_datetime(geometry.time_utc,utc=True)
    reference = pd.Timestamp(config.get("reference_time_utc") or times.min())
    if reference.tzinfo is None:
        raise ValueError("Weather reference_time_utc must include a timezone")
    return (times-reference).dt.total_seconds().to_numpy(),reference.isoformat()


def build_weather(geometry, config, station_config):
    model = config.get("model","uniform")
    rate = float(config.get("rain_rate_mm_h",0.))
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("Rain rate must be finite and nonnegative")
    top = float(config["rain_top_km"])
    height = float(station_config.get("height_m",0.))+float(station_config.get("antenna_height_m",0.))
    path = rain_paths(geometry,top,height)
    if model == "bom":
        from src.bom_rain import build_bom_weather
        return build_bom_weather(geometry, config, station_config, path)
    out = geometry[KEYS].copy()
    elapsed,reference = _elapsed(geometry,config)
    if model == "uniform":
        rates = np.full(len(geometry),rate)
    elif model == "temporal_uniform":
        series = config["time_series"]
        knots = np.asarray(series["elapsed_s"],dtype=float)
        values = np.asarray(series["rain_rate_mm_h"],dtype=float)
        if knots.ndim != 1 or len(knots)<2 or values.shape!=knots.shape or not np.isfinite([knots,values]).all() or np.any(np.diff(knots)<=0) or np.any(values<0):
            raise ValueError("Rain time-series knots must be finite, ordered and nonnegative")
        if elapsed.min()<knots[0]-1e-6 or elapsed.max()>knots[-1]+1e-6:
            raise ValueError("Rain time series does not cover the requested observation window")
        rates = np.interp(elapsed,knots,values)
    elif model == "advecting_cells":
        count = int(config.get("integration_segments",64))
        if not 8 <= count <= 1024:
            raise ValueError("Rain integration_segments must be in [8,1024]")
        columns = ["station_x_km","station_y_km","station_z_km"]
        if not all(key in geometry for key in columns):
            raise ValueError("Spatial rainfall requires ITRS station geometry")
        station = geometry[columns].to_numpy(dtype=float)
        if not np.allclose(station,station[0],atol=1e-8):
            raise ValueError("One rainfall field requires a fixed ground station")
        east,north,_ = station_axes(station[0]*1000)
        satellite = geometry[["sat_x_km","sat_y_km","sat_z_km"]].to_numpy(dtype=float)
        direction = (satellite-station)/geometry.slant_range_km.to_numpy()[:,None]
        positions = path[:,None]*(np.arange(count)[None,:]+.5)/count
        east_km = positions*(direction@east)[:,None]
        north_km = positions*(direction@north)[:,None]
        samples = np.full((len(geometry),count),rate)
        rates = np.full(len(geometry),rate)
        cells = config.get("cells",[])
        if not cells:
            raise ValueError("advecting_cells requires at least one cell")
        for cell in cells:
            peak = float(cell["peak_mm_h"])
            sx,sy = float(cell["sigma_east_km"]),float(cell["sigma_north_km"])
            e = float(cell["east_km"])+float(cell.get("velocity_east_m_s",0.))*elapsed/1000
            n = float(cell["north_km"])+float(cell.get("velocity_north_m_s",0.))*elapsed/1000
            if not np.isfinite([peak,sx,sy]).all() or peak<0 or min(sx,sy)<=0 or not np.isfinite([e,n]).all():
                raise ValueError("Invalid moving rain cell")
            samples += peak*np.exp(-.5*(((east_km-e[:,None])/sx)**2+((north_km-n[:,None])/sy)**2))
            rates += peak*np.exp(-.5*((e/sx)**2+(n/sy)**2))
        out["rain_rate_samples_mm_h"] = list(samples)
        out["rain_segment_length_km"] = path/count
        out["rain_rate_path_mean_mm_h"] = samples.mean(axis=1)
    else:
        raise ValueError(f"Unknown weather.model: {model}")
    out["rain_rate_mm_h"] = rates
    out["rain_path_km"] = path
    out["rain_top_km"] = top
    out["truth_source"] = "synthetic_"+model
    out["weather_reference_time_utc"] = reference
    return out


def uniform_rain(geometry, config, station_height_m):
    """Compatibility wrapper, dispatching to the configured rain model."""
    return build_weather(geometry,config,{"height_m":station_height_m})


def clear_weather(weather):
    result = weather.copy()
    result["rain_rate_mm_h"] = 0.
    if "rain_rate_samples_mm_h" in result:
        result["rain_rate_samples_mm_h"] = [np.zeros_like(x) for x in result.rain_rate_samples_mm_h]
        result["rain_rate_path_mean_mm_h"] = 0.
    result["truth_source"] = "synthetic_zero_rain_control"
    return result


def retrieval_config(row, config):
    settings = config.get("retrieval",{})
    top = float(settings.get("assumed_rain_top_km",config["weather"]["rain_top_km"]))
    length = float(rain_paths(pd.DataFrame([row]),top)[0])
    return dict(carrier_hz=float(config["link"]["carrier_hz"]),elevation_deg=float(row.elevation_deg),
                polarization_tilt_deg=float(config["propagation"]["polarization_tilt_deg"]),
                rain_frequency_model=config["propagation"].get("rain_frequency_model", "center_frequency_flat"),
                rain_microphysics=config["propagation"].get("rain_microphysics", {}),
                assumed_rain_top_km=top,assumed_rain_path_km=length,
                interpretation="uniform-equivalent path rain rate, not a point rain gauge or a reconstructed rain field")
