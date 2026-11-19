"""Current paper H_S + H_X + H_R, with moving satellite geometry on all rays.

Geometry determines delay and Doppler; no Doppler band is hand-assigned to rain.
Rain uses a finite zero-mean single-scatter Monte Carlo representation. Its
power weights use a stated DSD, Mie amplitudes and volume, not measured Starlink
rain calibration. See MODEL.md for assumptions and scope.
"""
import numpy as np
import pandas as pd
import miepython

from src.reflection import station_axes, compute_ground_reflection
from src.antenna import path_antenna_gains, path_to_los_gain
from src.rain_complex import C, water_permittivity


def _vector(value, name):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite 3-vector")
    return result


def path_geometry(satellite_m, satellite_velocity_m_s, receiver_m,
                  receiver_velocity_m_s, scatterer_m, scatterer_velocity_m_s, carrier_hz):
    """Instantaneous common-epoch bistatic geometry, exp(+jwt) convention."""
    s, vs, r, vr, b, vb = [_vector(x, name) for x, name in zip(
        [satellite_m, satellite_velocity_m_s, receiver_m, receiver_velocity_m_s,
         scatterer_m, scatterer_velocity_m_s], ['satellite', 'vs', 'receiver', 'vr', 'scatterer', 'vb'])]
    fc = float(carrier_hz)
    if not np.isfinite(fc) or fc <= 0:
        raise ValueError("Carrier frequency must be positive")
    d0, dt, dr = np.linalg.norm(s-r), np.linalg.norm(s-b), np.linalg.norm(b-r)
    if min(d0, dt, dr) <= 0:
        raise ValueError("Degenerate direct or scattering path")
    u0, ut, ur = (s-r)/d0, (s-b)/dt, (b-r)/dr
    direct_rate = float(u0 @ (vs-vr))
    satellite_term = float((ut-u0) @ vs)
    scatterer_term = float((ur-ut) @ vb)
    receiver_term = float((u0-ur) @ vr)
    # Stable subtraction of two satellite ranges, followed by the local leg.
    difference = float(((s-b)+(s-r)) @ (r-b)/(dt+d0)+dr)
    difference = max(difference, 0.)  # roundoff at the exactly collinear baseline
    relative_rate = satellite_term+scatterer_term+receiver_term
    return dict(path_length_m=dt+dr, direct_length_m=d0, excess_path_m=difference,
        delay_s=(dt+dr)/C, excess_delay_s=difference/C,
        direct_range_rate_m_s=direct_rate, range_rate_m_s=direct_rate+relative_rate,
        relative_doppler_hz=-fc/C*relative_rate,
        relative_doppler_satellite_hz=-fc/C*satellite_term,
        relative_doppler_scatterer_hz=-fc/C*scatterer_term,
        relative_doppler_receiver_hz=-fc/C*receiver_term,
        tx_leg_m=dt, rx_leg_m=dr, arrival_direction_itrs=ur.tolist(),
        scattering_cosine=float(np.clip(ut @ ur, -1., 1.)))


def terminal_fall_speed(diameter_mm):
    """Atlas et al. (1973) still-air, near-sea-level fit, D in mm, speed m/s.

    Only use its stated liquid-rain size interval; do not extrapolate the
    negative-speed small-drop end or invent high-altitude density corrections.
    """
    d = np.asarray(diameter_mm, float)
    if not np.isfinite(d).all() or np.any((d < .2) | (d > 5.8)):
        raise ValueError("Fall-speed fit is restricted to 0.2--5.8 mm drops")
    return 9.65-10.3*np.exp(-.6*d)


def scattering_amplitude_m(frequency_hz, diameter_mm, cosine, temperature_c=20., port='co'):
    """Scalar circular-polarization Mie amplitude in the scattering-plane basis.

    Orientation transport across a wide aperture is not a polarimetric model.
    In the co-circular forward limit this equals rain_complex.forward_amplitude_m.
    """
    if port not in ('co', 'cross') or not np.isfinite(cosine) or not -1 <= cosine <= 1:
        raise ValueError("Invalid scattering angle or circular polarization port")
    k = 2*np.pi*float(frequency_hz)/C
    m = np.sqrt(water_permittivity(frequency_hz, temperature_c))
    a, b = miepython.S1_S2(m, k*float(diameter_mm)*.001/2, float(cosine), norm='wiscombe')
    amplitude = (a[0]+b[0])/2 if port == 'co' else (b[0]-a[0])/2
    return complex(-1j*amplitude/k)


def rate_at_enu(enu_m, time_utc, weather, fallback_reference, origin_itrs_m=None):
    """Same horizontal synthetic weather law as weather.build_weather."""
    points = np.asarray(enu_m, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Weather query points must be finite ENU vectors")
    if weather.get('model') == 'bom':
        from src.bom_rain import query_itrs, require_rates
        if origin_itrs_m is None:
            raise ValueError('BOM ENU queries require an ITRS origin')
        origin = np.asarray(origin_itrs_m, float)
        world = origin+points@np.column_stack(station_axes(origin)).T
        return require_rates(query_itrs(world, time_utc, weather))
    rate = np.full(len(points), float(weather.get('rain_rate_mm_h', 0.)))
    model = weather.get('model', 'uniform')
    reference = pd.Timestamp(weather.get('reference_time_utc') or fallback_reference)
    time = pd.Timestamp(time_utc)
    if reference.tzinfo is None or time.tzinfo is None:
        raise ValueError("Weather times must include UTC offsets")
    elapsed = (time-reference).total_seconds()
    if model == 'temporal_uniform':
        series = weather['time_series']
        knots, values = np.asarray(series['elapsed_s'], float), np.asarray(series['rain_rate_mm_h'], float)
        if (len(knots) < 2 or knots.shape != values.shape or not np.isfinite([knots, values]).all()
                or np.any(np.diff(knots) <= 0) or np.any(values < 0) or not knots[0] <= elapsed <= knots[-1]):
            raise ValueError("Invalid or uncovered temporal rain field")
        rate[:] = np.interp(elapsed, knots, values)
    elif model == 'advecting_cells':
        if not weather.get('cells'):
            raise ValueError("Moving rain field has no cells")
        for cell in weather['cells']:
            e = float(cell['east_km'])+float(cell.get('velocity_east_m_s', 0.))*elapsed/1000
            n = float(cell['north_km'])+float(cell.get('velocity_north_m_s', 0.))*elapsed/1000
            sx, sy = float(cell['sigma_east_km']), float(cell['sigma_north_km'])
            if min(sx, sy) <= 0 or float(cell['peak_mm_h']) < 0:
                raise ValueError("Invalid rain cell")
            rate += float(cell['peak_mm_h'])*np.exp(-.5*((points[:, 0]/1000-e)**2/sx**2
                                                    +(points[:, 1]/1000-n)**2/sy**2))
    elif model != 'uniform':
        raise ValueError("Unknown rain field")
    if not np.isfinite(rate).all() or np.any((rate < 0) | (rate > 500)):
        raise ValueError("Rain rates outside the supported 0--500 mm/h interval")
    return rate



def ground_path(row, config):
    """Smooth-facet Fresnel ground ray, classified as X, never static CSI."""
    satellite = row[['sat_x_km', 'sat_y_km', 'sat_z_km']].to_numpy(float)
    receiver = row[['station_x_km', 'station_y_km', 'station_z_km']].to_numpy(float)
    velocity = row[['sat_vx_km_s', 'sat_vy_km_s', 'sat_vz_km_s']].to_numpy(float)
    if not config['ofdm'].get('motion_enabled', True):
        velocity = np.zeros(3)
    reflection = compute_ground_reflection(dict(satellite_xyz_km=satellite,
        station_xyz_km=receiver, satellite_velocity_km_s=velocity), config,
        float(config['link']['bandwidth_hz']))
    fc = float(config['link']['carrier_hz'])
    coefficient = complex(reflection['relative_coefficient_real'], reflection['relative_coefficient_imag'])
    relative = coefficient*np.exp(-2j*np.pi*fc*reflection['excess_path_m']/C)
    direct_rate, rate = reflection['direct_range_rate_m_s'], reflection['reflected_range_rate_m_s']
    return dict(group='X', path_role='ground_reflection', path_model='ground_specular',
        position_itrs_m=reflection['ground_point_itrs_m'], velocity_itrs_m_s=[0., 0., 0.],
        coefficient_real=coefficient.real, coefficient_imag=coefficient.imag,
        relative_coefficient_real=relative.real, relative_coefficient_imag=relative.imag,
        delay_samples=reflection['delay_samples'], delay_s=reflection['lr_m']/C,
        excess_delay_s=reflection['excess_path_m']/C, excess_path_m=reflection['excess_path_m'],
        path_length_m=reflection['lr_m'], direct_length_m=reflection['ld_m'],
        direct_range_rate_m_s=direct_rate, range_rate_m_s=rate,
        doppler_difference_hz=-fc/C*(rate-direct_rate), relative_doppler_hz=-fc/C*(rate-direct_rate),
        absolute_doppler_hz=-fc/C*rate, relative_doppler_satellite_hz=-fc/C*(rate-direct_rate),
        relative_doppler_scatterer_hz=0., relative_doppler_receiver_hz=0.,
        arrival_direction_itrs=reflection['arrival_direction_itrs'],
        antenna_relative_amplitude=reflection['antenna_relative_amplitude'],
        tx_gain_dbi=reflection['tx_gain_dbi'], rx_gain_dbi=reflection['rx_gain_dbi'],
        coefficient_basis='Fresnel_local_smooth_facet_with_roughness_and_angular_antenna_gains')


def build_paper_paths(row, config, case='rain'):
    """Construct environmental X and rain R rays relative to direct path S.

    Environmental point-scatterer strengths are explicit scenario inputs, not inferred from a
    point's position. Rain strengths instead use physical second-moment weights
    sqrt[N(D,x) V/(Q p(D))] of a zero-mean single-scattering population.
    Independent burst realizations are not a continuous drop-replenishment model.
    """
    settings = config['propagation'].get('paper_channel', {})
    if not settings.get('enabled', False):
        return []
    if case not in ('rain', 'reference'):
        raise ValueError("Unknown capture case")
    if config['station'].get('polarization', 'circular') != 'circular':
        raise ValueError("Current paper-channel scattering implementation uses a scalar circular port")
    satellite = row[['sat_x_km', 'sat_y_km', 'sat_z_km']].to_numpy(dtype=float)*1000
    receiver = row[['station_x_km', 'station_y_km', 'station_z_km']].to_numpy(dtype=float)*1000
    velocity = row[['sat_vx_km_s', 'sat_vy_km_s', 'sat_vz_km_s']].to_numpy(dtype=float)*1000
    if not config['ofdm'].get('motion_enabled', True):
        velocity[:] = 0.
    basis = np.column_stack(station_axes(receiver))
    carrier = float(config['link']['carrier_hz'])
    seed = int(settings.get('seed', 20261018))
    rng = np.random.default_rng(seed)
    paths = []

    def append(position_enu, velocity_enu, coefficient, group, **extra):
        point, point_velocity = receiver+basis@position_enu, basis@velocity_enu
        geometry = path_geometry(satellite, velocity, receiver, np.zeros(3), point, point_velocity, carrier)
        intrinsic = complex(coefficient)
        antenna_ratio, antenna = path_to_los_gain(satellite, receiver, point, config['propagation'])
        coefficient = intrinsic*float(antenna_ratio)
        if not np.isfinite([coefficient.real, coefficient.imag]).all():
            raise ValueError("Nonfinite path coefficient")
        # Coefficients exclude geometric carrier phase; never add Doppler twice.
        phase = -2*np.pi*carrier*geometry['excess_delay_s']
        relative = coefficient*np.exp(1j*phase)
        paths.append(dict(group=group, position_itrs_m=point.tolist(), velocity_itrs_m_s=point_velocity.tolist(),
            coefficient_real=coefficient.real, coefficient_imag=coefficient.imag,
            intrinsic_coefficient_real=intrinsic.real, intrinsic_coefficient_imag=intrinsic.imag,
            antenna_relative_amplitude=float(antenna_ratio),
            tx_gain_dbi=float(antenna['tx_gain_dbi']), rx_gain_dbi=float(antenna['rx_gain_dbi']),
            path_model='point_scatterer', absolute_doppler_hz=-carrier/C*geometry['range_rate_m_s'],
            relative_coefficient_real=relative.real, relative_coefficient_imag=relative.imag,
            delay_samples=geometry['excess_delay_s']*float(config['link']['bandwidth_hz']),
            doppler_difference_hz=geometry['relative_doppler_hz'], **geometry, **extra))

    for role, key in [('fixed_environment', 'fixed_scatterers'), ('moving_environment', 'moving_scatterers')]:
        for item in settings.get(key, []):
            position = _vector(item['position_enu_m'], 'scatterer position')
            speed = np.zeros(3) if role == 'fixed_environment' else _vector(item['velocity_enu_m_s'], 'scatterer velocity')
            if item.get('position_epoch_utc') is not None:
                elapsed = (pd.Timestamp(row['time_utc'])-pd.Timestamp(item['position_epoch_utc'])).total_seconds()
                position = position+speed*elapsed
            physical = 'scattering_coefficient_m' in item
            if physical and 'relative_coefficient' in item:
                raise ValueError('Specify scattering_coefficient_m or relative_coefficient, not both')
            coefficient = np.asarray(item['scattering_coefficient_m'] if physical else item['relative_coefficient'], float)
            if coefficient.shape != (2,):
                raise ValueError("Environmental coefficient must contain real and imaginary parts")
            value = complex(*coefficient)
            if physical:
                point = receiver+basis@position
                geometry = path_geometry(satellite, velocity, receiver, np.zeros(3), point, basis@speed, carrier)
                value *= geometry['direct_length_m']/(geometry['tx_leg_m']*geometry['rx_leg_m'])
            append(position, speed, value, 'X', path_role=role,
                   coefficient_basis=('declared_effective_scattering_amplitude_m_with_both_leg_spreading'
                                      if physical else 'legacy_per_epoch_relative_scenario_amplitude'))
    if config['propagation']['enabled'].get('ground_reflection', False):
        paths.append(ground_path(row, config))
    rain = settings.get('rain_scattering', {})
    if case == 'reference' or not rain.get('enabled', False) or not config['propagation']['enabled']['rain']:
        return paths
    count = rain.get('representatives', 32)
    if not isinstance(count, int) or not 8 <= count <= 4096:
        raise ValueError("rain_scattering.representatives must be in [8,4096]")
    size = _vector(rain['volume_size_enu_m'], 'rain volume size')
    if np.any(size <= 0):
        raise ValueError("Rain volume dimensions must be positive")
    direction_enu = basis.T@(satellite-receiver)/np.linalg.norm(satellite-receiver)
    center = _vector(rain.get('volume_center_enu_m', direction_enu*1000), 'rain volume center')
    station_height = float(config['station']['height_m'])+float(config['station']['antenna_height_m'])
    if center[2]-size[2]/2 < 0 or center[2]+size[2]/2 > config['weather']['rain_top_km']*1000-station_height:
        raise ValueError("The whole local rain-scattering box must lie inside the liquid rain layer")
    positions = center+(rng.random((count, 3))-.5)*size
    rates = rate_at_enu(positions, row['time_utc'], config['weather'],
                        config['geometry']['start_time_utc'] or row['time_utc'],
                        origin_itrs_m=receiver)
    bounds = np.asarray(rain.get('diameter_bounds_mm', [.2, 5.8]), float)
    if bounds.shape != (2,) or not .2 <= bounds[0] < bounds[1] <= 5.8:
        raise ValueError("Rain scattering diameters must lie in 0.2--5.8 mm")
    proposal_rate = float(rain.get('proposal_rate_mm_h', 10.))
    if not 0 < proposal_rate <= 500:
        raise ValueError("Invalid Monte Carlo proposal rate")
    slope = 4.1*proposal_rate**(-.21)
    low, high = np.exp(-slope*bounds)
    diameter = -np.log(low-rng.random(count)*(low-high))/slope
    proposal = slope*np.exp(-slope*diameter)/(low-high)
    density = np.zeros(count)
    active = rates > 0
    density[active] = 8000*np.exp(-4.1*rates[active]**(-.21)*diameter[active])
    multiplicity = density*np.prod(size)/(count*proposal)
    wind = _vector(rain.get('wind_enu_m_s', [0., 0., 0.]), 'wind')
    phases = rng.uniform(-np.pi, np.pi, count)
    temperature = float(config['propagation'].get('rain_microphysics', {}).get('temperature_c', 20.))
    for i in range(count):
        if multiplicity[i] == 0:
            continue
        point = receiver+basis@positions[i]
        speed = wind+np.array([0., 0., -float(terminal_fall_speed(diameter[i]))])
        geometry = path_geometry(satellite, velocity, receiver, np.zeros(3), point, basis@speed, carrier)
        amplitude = scattering_amplitude_m(carrier, diameter[i], geometry['scattering_cosine'], temperature)
        relative = (amplitude*geometry['direct_length_m']/(geometry['tx_leg_m']*geometry['rx_leg_m'])
                    *np.sqrt(multiplicity[i])*np.exp(1j*phases[i]))
        append(positions[i], speed, relative, 'R', diameter_mm=float(diameter[i]),
            represented_number_for_power=float(multiplicity[i]), local_rain_rate_mm_h=float(rates[i]),
            coefficient_basis='zero_mean_Mie_MP_volume_second_moment_not_measured_calibration')
    return paths


def additional_response(paths, frequency_offsets_hz, fs):
    """Additional relative transfer at the burst epoch (not a receiver estimate)."""
    frequency = np.asarray(frequency_offsets_hz, float)
    response = np.zeros_like(frequency, complex)
    for path in paths:
        coefficient = complex(path['relative_coefficient_real'], path['relative_coefficient_imag'])
        if 'rain_frequency_offset_hz' in path:
            nodes = path['rain_frequency_offset_hz']
            coefficient = coefficient*10**(-np.interp(frequency, nodes, path['rain_delta_loss_db'])/20)
            coefficient = coefficient*np.exp(1j*np.interp(frequency, nodes, path['rain_delta_phase_rad']))
        response += coefficient*np.exp(-2j*np.pi*frequency*float(path['delay_samples'])/fs)
    return response


def _ground_at(satellite, receiver, config):
    return compute_ground_reflection(dict(satellite_xyz_km=np.asarray(satellite)/1000,
        station_xyz_km=np.asarray(receiver)/1000), config, float(config['link']['bandwidth_hz']))


def _ray_points(path, satellite, receiver, elapsed, config):
    if path is None:
        return np.asarray([satellite, receiver])
    if path.get('path_model') == 'ground_specular':
        point = _ground_at(satellite, receiver, config)['ground_point_itrs_m']
    else:
        point = np.asarray(path['position_itrs_m'])+elapsed*np.asarray(path['velocity_itrs_m_s'])
    return np.asarray([satellite, point, receiver])


def _geometric_log_amplitude(points, path, config):
    satellite, receiver = points[0], points[-1]
    if path is None:
        gains = path_antenna_gains(satellite, receiver, config['propagation'])
        return -np.log(np.linalg.norm(satellite-receiver))+np.log(10)/20*gains['combined_gain_dbi']
    if path.get('path_model') == 'ground_specular':
        reflected = _ground_at(satellite, receiver, config)
        coefficient = complex(reflected['relative_coefficient_real'], reflected['relative_coefficient_imag'])
        return _geometric_log_amplitude(np.asarray([satellite, receiver]), None, config)+np.log(max(abs(coefficient), 1e-300))
    gains = path_antenna_gains(satellite, receiver, config['propagation'], points[1]-satellite, points[1]-receiver)
    distance = np.linalg.norm(satellite-points[1])*np.linalg.norm(points[1]-receiver)
    return -np.log(distance)+np.log(10)/20*gains['combined_gain_dbi']


def prepare_frame_paths(row, config, case='rain', paths=None):
    """Freeze spectra at a frame epoch and retain their local time derivatives.

    Every additional ray is relative to the direct ray. The IQ renderer applies
    each rain spectrum separately, with geometric Doppler counted only once.
    """
    from src.path_rain import ray_rain_spectrum, derivative_times
    paths = build_paper_paths(row, config, case) if paths is None else paths
    satellite = row[['sat_x_km', 'sat_y_km', 'sat_z_km']].to_numpy(float)*1000
    velocity = row[['sat_vx_km_s', 'sat_vy_km_s', 'sat_vz_km_s']].to_numpy(float)*1000
    if not config['ofdm'].get('motion_enabled', True):
        velocity = np.zeros(3)
    receiver = row[['station_x_km', 'station_y_km', 'station_z_km']].to_numpy(float)*1000
    fs = float(config['link']['bandwidth_hz'])
    count = int(config['propagation'].get('rain_microphysics', {}).get('spectral_nodes', 33))
    nfft = int(round(fs/float(config['link']['subcarrier_spacing_hz'])))
    nodes = np.unique(np.r_[np.linspace(-fs/2, fs/2-fs/nfft, count), 0.])
    epoch = pd.Timestamp(row.time_utc)
    offsets = derivative_times(epoch, config)
    span = offsets[1]-offsets[0]
    points0 = _ray_points(None, satellite, receiver, 0., config)
    direct_loss, direct_phase = ray_rain_spectrum(points0, receiver, epoch, config, nodes, case)
    def slopes(path):
        values = []
        for offset in offsets:
            points = _ray_points(path, satellite+offset*velocity, receiver, offset, config)
            loss, phase = ray_rain_spectrum(points, receiver, epoch+pd.Timedelta(seconds=float(offset)),
                                           config, [0.], case)
            values.append((_geometric_log_amplitude(points, path, config)-np.log(10)/20*loss[0], phase[0]))
        return (np.asarray(values[1])-values[0])/span if span > 0 else np.zeros(2)
    direct_slopes = slopes(None)
    direct = dict(frequency_offset_hz=nodes.tolist(), rain_loss_db=direct_loss.tolist(),
        rain_phase_rad=direct_phase.tolist(), rain_phase_rate_rad_s=float(direct_slopes[1]),
        log_amplitude_rate_per_s=float(direct_slopes[0]),
        rain_time_model='individual_ray_complex_transfer_with_local_frame_derivative')
    for path in paths:
        points = _ray_points(path, satellite, receiver, 0., config)
        loss, phase = ray_rain_spectrum(points, receiver, epoch, config, nodes, case)
        rates = slopes(path)
        path.update(rain_frequency_offset_hz=nodes.tolist(),
            rain_delta_loss_db=(loss-direct_loss).tolist(), rain_delta_phase_rad=(phase-direct_phase).tolist(),
            rain_phase_rate_rad_s=float(rates[1]),
            rain_phase_rate_difference_rad_s=float(rates[1]-direct_slopes[1]),
            log_amplitude_rate_difference_per_s=float(rates[0]-direct_slopes[0]),
            rain_path_model='integrated_along_this_ray_not_shared_LoS_transfer')
    return paths, direct


def render_components(row, paths, satellite_positions_m, times_s, frequency_offsets_hz,
                      carrier_hz, net_loss_db=0., config=None, case='rain'):
    """Analytical physical CSI, separately labelled from IQ-derived pilot CSI.

    Satellite positions are evaluated at each snapshot. With config, every ray
    uses its current antenna gain and its own integrated complex rain transfer.
    Point-scatter strengths/DSD weights are frozen over the declared short CPI.
    """
    times = np.asarray(times_s, float)
    satellite = np.asarray(satellite_positions_m, float)
    frequency = np.asarray(frequency_offsets_hz, float)
    if (times.ndim != 1 or not times.size or satellite.shape != (times.size, 3)
            or frequency.ndim != 1 or not np.isfinite(satellite).all()
            or not np.isfinite(times).all() or not np.isfinite(frequency).all()):
        raise ValueError('Invalid CPI channel coordinates')
    if config is None and any(p.get('path_model') == 'ground_specular' for p in paths):
        raise ValueError('A specular ground CPI requires its physical configuration')
    receiver = row[['station_x_km', 'station_y_km', 'station_z_km']].to_numpy(dtype=float)*1000
    direct = np.linalg.norm(satellite-receiver, axis=1)
    anchor = direct[0]
    center_rain = float(row.get('rain_db', 0.)) if config is not None and case == 'rain' else 0.
    amplitude = 10**(-(float(net_loss_db)-center_rain)/20)
    direct_phase = np.exp(-2j*np.pi*(carrier_hz+frequency[None, :])*(direct[:, None]-anchor)/C)
    antenna_scale = np.ones(times.size)
    if config is not None:
        gains = path_antenna_gains(satellite, receiver, config['propagation'])['combined_gain_dbi']
        antenna_scale = 10**((gains-gains[0])/20)
    los = amplitude*anchor/direct[:, None]*antenna_scale[:, None]*direct_phase
    result = {key: np.zeros_like(los) for key in ['S', 'X', 'R']}
    result['S'] = los.copy()
    for path in paths:
        if path.get('path_model') == 'ground_specular':
            for j, (position, time) in enumerate(zip(satellite, times)):
                reflection = _ground_at(position, receiver, config)
                coefficient = complex(reflection['relative_coefficient_real'], reflection['relative_coefficient_imag'])
                phase = np.exp(-2j*np.pi*(carrier_hz+frequency)*reflection['excess_path_m']/C)
                value = los[j]*coefficient*phase
                if config is not None:
                    from src.path_rain import ray_rain_spectrum
                    rain_loss, rain_phase = ray_rain_spectrum(_ray_points(path, position, receiver, time, config),
                        receiver, pd.Timestamp(row.time_utc)+pd.Timedelta(seconds=float(time)), config, frequency, case)
                    value *= 10**(-rain_loss/20)*np.exp(1j*rain_phase)
                result['X'][j] += value
            continue
        point = np.asarray(path['position_itrs_m'])+times[:, None]*np.asarray(path['velocity_itrs_m_s'])
        tx_leg = np.linalg.norm(satellite-point, axis=1)
        rx_leg = np.linalg.norm(point-receiver, axis=1)
        length = tx_leg+rx_leg
        coefficient = complex(path['coefficient_real'], path['coefficient_imag'])
        spreading = path['tx_leg_m']*path['rx_leg_m']/(tx_leg*rx_leg)
        pattern = np.ones(times.size)
        if config is not None:
            gains = path_antenna_gains(satellite, receiver, config['propagation'], point-satellite, point-receiver)
            anchor_gain = float(path['tx_gain_dbi'])+float(path['rx_gain_dbi'])
            pattern = 10**((gains['combined_gain_dbi']-anchor_gain)/20)
        phase = np.exp(-2j*np.pi*(carrier_hz+frequency[None, :])*(length[:, None]-anchor)/C)
        values = amplitude*coefficient*(spreading*pattern)[:, None]*phase
        if config is not None:
            from src.path_rain import ray_rain_spectrum
            for j, time in enumerate(times):
                rain_loss, rain_phase = ray_rain_spectrum([satellite[j], point[j], receiver], receiver,
                    pd.Timestamp(row.time_utc)+pd.Timedelta(seconds=float(time)), config, frequency, case)
                values[j] *= 10**(-rain_loss/20)*np.exp(1j*rain_phase)
        result[path['group']] += values
    if config is not None:
        from src.path_rain import ray_rain_spectrum
        for j, time in enumerate(times):
            rain_loss, rain_phase = ray_rain_spectrum([satellite[j], receiver], receiver,
                pd.Timestamp(row.time_utc)+pd.Timedelta(seconds=float(time)), config, frequency, case)
            result['S'][j] *= 10**(-rain_loss/20)*np.exp(1j*rain_phase)
    result['LoS'] = result['S']  # explicitly labelled reference truth, not a measured pilot
    return result
