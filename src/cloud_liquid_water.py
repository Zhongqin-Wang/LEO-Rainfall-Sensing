"""Select historical ERA5T cloud water for a fixed receiver's local column."""
from pathlib import Path

from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
ECEF_TO_WGS84 = Transformer.from_crs(4978, 4979, always_xy=True)


def era5_column(geometry, config):
    """Nearest spatial cell, newest timestamp at or before the simulation time."""
    path = ROOT / config["era5_file"]
    max_age = float(config.get("max_age_seconds", 3600))
    if not np.isfinite(max_age) or max_age <= 0:
        raise ValueError("Cloud max_age_seconds must be finite and positive")
    xyz = geometry[["station_x_km", "station_y_km", "station_z_km"]].to_numpy(float)*1000
    if not np.isfinite(xyz).all() or not np.allclose(xyz, xyz[0], atol=1e-3, rtol=0):
        raise ValueError("ERA5 cloud mode requires a fixed receiver with finite coordinates")
    lon, lat, _ = ECEF_TO_WGS84.transform(*xyz[0])
    with Dataset(path) as ds:
        v = ds["tclw"]
        if v.units not in ("kg m**-2", "kg m-2", "kg m^-2", "kg/m²", "kg/m2"):
            raise ValueError("Expected ERA5 tclw in kg/m²")
        time_name = v.dimensions[0]
        if time_name not in ("valid_time", "time") or v.dimensions != (time_name, "latitude", "longitude"):
            raise ValueError("Expected tclw(time, latitude, longitude)")
        t = ds[time_name]
        dates = num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"),
                         only_use_cftime_datetimes=False)
        times = pd.DatetimeIndex(dates, tz="UTC").as_unit("ns")
        if not len(times) or times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
            raise ValueError("ERA5 times must be strictly increasing")
        ys, xs = [np.asarray(ds[key][:], dtype=float) for key in ("latitude", "longitude")]
        if not all(len(a) and np.isfinite(a).all() for a in (ys, xs)):
            raise ValueError("Invalid ERA5 coordinate axes")
        if xs.min() >= 0:
            lon %= 360
        if not (ys.min() <= lat <= ys.max() and xs.min() <= lon <= xs.max()):
            raise ValueError("Receiver outside ERA5 cloud grid")
        y, x = np.argmin(abs(ys-lat)), np.argmin(abs(xs-lon))
        values = np.ma.asarray(v[:, y, x], dtype=float).filled(np.nan)
    if any(pd.Timestamp(t).tzinfo is None for t in geometry.time_utc):
        raise ValueError("Cloud simulation times must have a timezone")
    requested = pd.DatetimeIndex(pd.to_datetime(geometry.time_utc, utc=True)).as_unit("ns")
    idx = np.searchsorted(times.asi8, requested.asi8, side="right")-1
    if np.any(idx < 0):
        raise ValueError("No earlier ERA5 tclw observation")
    age = (requested.asi8-times.asi8[idx])/1e9
    if np.any(age >= max_age):
        raise ValueError("ERA5 tclw stale or missing at a simulation time")
    selected = values[idx]
    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("Invalid ERA5 tclw at receiver")
    return selected, times[idx], ys[y], xs[x]
