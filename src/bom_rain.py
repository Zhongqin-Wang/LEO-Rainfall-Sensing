"""BOM Rainfields 2-D rain-rate queries, without invented weather resolution.
Positions are WGS84 longitude/latitude. Time is timezone-aware UTC. The most
recent observation is held for less than max_age_seconds; no future frame or
temporal interpolation is used. Operational publication latency is unknown.
"""
from bisect import bisect_right
from collections import OrderedDict
from functools import lru_cache
import hashlib
from pathlib import Path
from threading import RLock
import netCDF4
import numpy as np
import pandas as pd
from src.weather import KEYS
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
_LOCK = RLock()  # netCDF/HDF5 calls are serialized, including worker threads.
_GEODETIC = Transformer.from_crs(4978, 4979, always_xy=True)


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None or pd.isna(value):
        raise ValueError('Rain query time must include a timezone')
    return value.tz_convert('UTC')


class BOMRain:
    def __init__(self, directory, max_age_seconds=300.):
        self.directory = Path(directory).resolve()
        self.max_age = float(max_age_seconds)
        if not np.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError('max_age_seconds must be positive')
        records = []
        self.cache = OrderedDict()
        with _LOCK:
            for path in sorted(self.directory.rglob('*.nc')):
                with netCDF4.Dataset(path) as ds:
                    value = ds['valid_time']
                    date = netCDF4.num2date(value[()], value.units,
                        calendar=getattr(value, 'calendar', 'standard'),
                        only_use_cftime_datetimes=False)
                    stamp = pd.Timestamp(date).tz_localize('UTC')
                    records.append((stamp.value, path))
        if not records:
            raise ValueError(f'No NetCDF rain files in {self.directory}')
        records.sort()
        self.times, self.paths = zip(*records)
        if len(set(self.times)) != len(self.times):
            raise ValueError('Duplicate BOM valid_time values')

    def _frame(self, index):
        with _LOCK:
            if index not in self.cache:
                with netCDF4.Dataset(self.paths[index]) as ds:
                    var = ds['rain_rate']
                    if var.units not in ('mm hr-1', 'mm/h', 'mm h-1'):
                        raise ValueError('Expected rain rate in mm/h, not accumulated mm')
                    if tuple(var.dimensions) != ('y', 'x'):
                        raise ValueError('Expected rain_rate(y,x)')
                    proj = ds[var.grid_mapping]
                    crs = CRS.from_cf({k: proj.getncattr(k) for k in proj.ncattrs()})
                    scale = {'km': 1000., 'm': 1.}
                    x = np.asarray(ds['x'][:])*scale[ds['x'].units]
                    y = np.asarray(ds['y'][:])*scale[ds['y'].units]
                    xb = np.asarray(ds[ds['x'].bounds][:])*scale[ds['x'].units]
                    yb = np.asarray(ds[ds['y'].bounds][:])*scale[ds['y'].units]
                    rain = np.ma.asarray(var[:], dtype=float).filled(np.nan)
                    rain[(rain < 0) | ~np.isfinite(rain)] = np.nan
                    self.cache[index] = (x, y, xb, yb, rain,
                        Transformer.from_crs(4326, crs, always_xy=True))
                if len(self.cache) > 3:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            return self.cache[index]

    def query_rain(self, position, time):
        """Accept scalar or broadcastable arrays in lat_deg/lon_deg; height ignored."""
        stamp = utc(time)
        lon, lat = np.broadcast_arrays(np.asarray(position['lon_deg'], float),
                                       np.asarray(position['lat_deg'], float))
        if not np.isfinite([lon, lat]).all() or np.any(abs(lat) > 90) or np.any(abs(lon) > 180):
            raise ValueError('Invalid WGS84 longitude/latitude')
        rate = np.full(lon.shape, np.nan)
        status = np.full(lon.shape, 'no_prior_frame', dtype='U24')
        index = bisect_right(self.times, stamp.value)-1
        source_time, source_file, age = None, None, None
        if index >= 0:
            source_time = pd.Timestamp(self.times[index], tz='UTC').isoformat()
            source_file = str(self.paths[index])
            age = (stamp.value-self.times[index])/1e9
            if age >= self.max_age:
                status[...] = 'stale'
            else:
                x, y, xb, yb, rain, transform = self._frame(index)
                px, py = transform.transform(lon, lat)
                inside = (px >= xb.min()) & (px <= xb.max()) & (py >= yb.min()) & (py <= yb.max())
                ix = np.abs(np.asarray(px)[..., None]-x).argmin(axis=-1)
                iy = np.abs(np.asarray(py)[..., None]-y).argmin(axis=-1)
                rate = np.where(inside, rain[iy, ix], np.nan)
                status = np.where(~inside, 'outside_grid', np.where(np.isfinite(rate), 'ok', 'missing'))
        valid = np.isfinite(rate)
        scalar = lon.ndim == 0
        return dict(rain_rate_mm_h=float(rate) if scalar else rate,
            valid_mask=bool(valid) if scalar else valid,
            source_time_utc=source_time, source_file=source_file, age_seconds=age,
            quality={'query_status': str(status.item()) if scalar else status,
                     'radar_quality': None}, uncertainty_mm_h=None,
            actual_available_time_utc=None)


@lru_cache(maxsize=4)
def _reader(directory, age):
    return BOMRain(directory, age)


def reader(weather):
    path = Path(weather['bom']['directory'])
    if not path.is_absolute():
        path = ROOT/path
    return _reader(str(path.resolve()), float(weather['bom'].get('max_age_seconds', 300.)))


def fingerprint(weather):
    """Content hash for archive reuse checks; computed at pipeline boundaries."""
    _reader.cache_clear()
    source = reader(weather)
    digest = hashlib.sha256()
    for path in sorted(source.directory.rglob('*.nc')):
        digest.update(path.relative_to(source.directory).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def query_itrs(points_m, time, weather):
    """Extrude the horizontal field vertically; layer clipping is caller-owned."""
    points = np.asarray(points_m, float)
    lon, lat, _ = _GEODETIC.transform(points[..., 0], points[..., 1], points[..., 2])
    return reader(weather).query_rain({'lon_deg': lon, 'lat_deg': lat}, time)


def require_rates(result):
    if not np.all(result['valid_mask']):
        raise ValueError(f"Unavailable BOM rain: {result['quality']['query_status']}; "
                         f"source={result['source_time_utc']}")
    rates = np.asarray(result['rain_rate_mm_h'], float)
    if np.any(rates > 500):
        raise ValueError('BOM rain exceeds the propagation model limit of 500 mm/h')
    return rates


def build_bom_weather(geometry, weather, station_config, path):
    """Supply the same segment rain samples used by the existing loss integrator."""

    count = int(weather.get('integration_segments', 64))
    if not 8 <= count <= 1024:
        raise ValueError('Rain integration_segments must be in [8,1024]')
    out = geometry[KEYS].copy()
    stations = geometry[['station_x_km', 'station_y_km', 'station_z_km']].to_numpy(float)*1000
    satellites = geometry[['sat_x_km', 'sat_y_km', 'sat_z_km']].to_numpy(float)*1000
    samples, rates, sources, files, ages = [], [], [], [], []
    for i, (_, row) in enumerate(geometry.iterrows()):
        direction = satellites[i]-stations[i]
        direction /= np.linalg.norm(direction)
        points = stations[i]+(np.arange(count)+.5)[:, None]/count*path[i]*1000*direction
        result = query_itrs(points, row.time_utc, weather)
        samples.append(require_rates(result))
        station = reader(weather).query_rain(station_config, row.time_utc)
        rates.append(float(require_rates(station)))
        sources.append(result['source_time_utc']); files.append(result['source_file']); ages.append(result['age_seconds'])
    out['rain_rate_mm_h'] = rates
    out['rain_rate_samples_mm_h'] = samples
    out['rain_segment_length_km'] = path/count
    out['rain_rate_path_mean_mm_h'] = np.mean(samples, axis=1)
    out['rain_path_km'] = path
    out['rain_top_km'] = float(weather['rain_top_km'])
    out['truth_source'] = 'bom_rainfields_2d_vertical_extrusion'
    out['weather_reference_time_utc'] = sources
    out['weather_source_file'] = files
    out['weather_age_seconds'] = ages
    out['weather_valid'] = True
    return out
