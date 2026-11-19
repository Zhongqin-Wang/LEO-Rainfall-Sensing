"""Rain transfer integrated separately along each straight segment of a ray.

The same expanded-WGS84 upper layer as weather.py is used. The lower boundary
is the local ground-height ellipsoid. No independent random rain phase is used.
"""
import numpy as np
import pandas as pd
from src.reflection import station_axes
from src.rain_complex import complex_rain_spectrum, settings_from_config
from src.rain_spectrum import attenuation_spectrum


def _inside_interval(start, end, axes):
    direction = end-start
    a = float(np.sum((direction/axes)**2))
    b = float(2*np.sum(start*direction/axes**2))
    c = float(np.sum((start/axes)**2)-1)
    if a == 0:
        return (0., 1.) if c < 0 else None
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return None
    root = np.sqrt(max(discriminant, 0.))
    q = -.5*(b+np.copysign(root, b))
    roots = sorted([q/a, c/q]) if q != 0 else [0., 0.]
    low, high = max(0., roots[0]), min(1., roots[1])
    return (low, high) if high > low else None


def rain_segments(points_m, config):
    """Return equal-length quadrature cells for each rain-intersecting leg."""
    points = np.asarray(points_m, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Rain ray requires finite 3D endpoints')
    top = float(config['weather']['rain_top_km'])*1000
    bottom = float(config['station']['height_m'])
    if top < 0 or not np.isfinite([top, bottom]).all():
        raise ValueError('Invalid rain layer heights')
    if top <= bottom:
        return []
    base = np.array([6378137., 6378137., 6356752.314245])
    count = config['weather'].get('integration_segments', 64)
    if not isinstance(count, int) or not 8 <= count <= 1024:
        raise ValueError('Rain integration_segments must be in [8,1024]')
    cells = []
    for start, end in zip(points[:-1], points[1:]):
        inside = _inside_interval(start, end, base+top)
        if inside is None:
            continue
        ground = _inside_interval(start, end, base+bottom)
        spans = [inside]
        if ground is not None:
            spans = []
            if inside[0] < min(inside[1], ground[0]):
                spans.append((inside[0], min(inside[1], ground[0])))
            if max(inside[0], ground[1]) < inside[1]:
                spans.append((max(inside[0], ground[1]), inside[1]))
        for low, high in spans:
            length_km = np.linalg.norm(end-start)*(high-low)/1000
            if length_km <= 1e-12:
                continue
            fractions = low+(high-low)*(np.arange(count)+.5)/count
            cells.append((start+fractions[:, None]*(end-start), float(length_km)))
    return cells


def ray_rain_spectrum(points_m, receiver_m, time_utc, config, frequency_offsets_hz, case='rain'):
    """Return excess rain loss (dB) and unwrapped phase (rad), never geometric phase."""
    offsets = np.asarray(frequency_offsets_hz, float)
    if offsets.ndim != 1 or offsets.size == 0 or not np.isfinite(offsets).all():
        raise ValueError('Rain frequency offsets must be a finite vector')
    target = offsets
    if case not in ('rain', 'reference'):
        raise ValueError('Unknown rain case')
    if case == 'reference' or not config['propagation']['enabled']['rain']:
        return np.zeros_like(target), np.zeros_like(target)
    # Late import avoids duplicating the declared temporal/spatial weather law.
    from src.paper_channel import rate_at_enu
    # All array channels query one world-fixed weather field, not a copy
    # translated independently to each antenna's phase centre.
    origin = np.asarray(config.get('_weather_origin_itrs_m', receiver_m), float)
    basis = np.column_stack(station_axes(origin))
    prop = config['propagation']
    count = int(prop.get('rain_microphysics', {}).get('spectral_nodes', 33))
    if not 9 <= count <= 257:
        raise ValueError('Rain spectral_nodes must be in [9,257]')
    if offsets.size > count:
        offsets = np.unique(np.r_[np.linspace(offsets.min(), offsets.max(), count), 0.])
    loss, phase = np.zeros_like(offsets), np.zeros_like(offsets)
    f = float(config['link']['carrier_hz'])+offsets
    mode = prop.get('rain_frequency_model', 'center_frequency_flat')
    phase_mode = prop.get('rain_phase_model', 'none')
    if mode not in ('center_frequency_flat', 'itu838_frequency_dependent', 'mie_spherical_mp'):
        raise ValueError('Unknown ray rain attenuation model')
    if phase_mode not in ('none', 'mie_spherical_mp'):
        raise ValueError('Unknown ray rain phase model')
    if mode == 'mie_spherical_mp' and phase_mode != 'mie_spherical_mp':
        raise ValueError('Complex Mie rain requires Mie rain phase')
    receiver = origin
    for positions, length in rain_segments(points_m, config):
        enu = (positions-receiver)@basis
        rates = rate_at_enu(enu, time_utc, config['weather'],
                           config['geometry'].get('start_time_utc') or time_utc,
                           origin_itrs_m=origin)
        if mode == 'mie_spherical_mp' or phase_mode == 'mie_spherical_mp':
            amplitude, angle = complex_rain_spectrum(f, length, rates, **settings_from_config(prop))
            if mode == 'mie_spherical_mp':
                loss += amplitude
            if phase_mode == 'mie_spherical_mp':
                phase += angle
        if mode != 'mie_spherical_mp':
            start, end = np.asarray(points_m[0]), np.asarray(points_m[-1])
            up = basis[:, 2]
            direction = start-end
            elevation = np.degrees(np.arcsin(np.clip(direction@up/np.linalg.norm(direction), 0., 1.)))
            elevation = max(float(elevation), .01)
            frequencies = f if mode == 'itu838_frequency_dependent' else np.full_like(f, config['link']['carrier_hz'])
            loss += attenuation_spectrum(frequencies, elevation, prop['polarization_tilt_deg'], length, rates)
    order = np.argsort(offsets)
    return np.interp(target, offsets[order], loss[order]), np.interp(target, offsets[order], phase[order])


def derivative_times(time_utc, config, step_s=.01):
    """Two nearby valid UTC epochs; one-sided at a temporal-field endpoint."""
    epoch = pd.Timestamp(time_utc)
    offsets = np.array([-step_s, step_s])
    weather = config['weather']
    if weather.get('model') == 'temporal_uniform':
        reference = pd.Timestamp(weather.get('reference_time_utc') or config['geometry']['start_time_utc'])
        elapsed = (epoch-reference).total_seconds()
        knots = weather['time_series']['elapsed_s']
        offsets = np.clip(offsets, float(knots[0])-elapsed, float(knots[-1])-elapsed)
    return offsets
