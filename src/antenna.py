"""Declared angular antenna patterns, distinct from receiver AGC.

The capped parabolic pattern is an engineering scenario, not a measured
Starlink or SDR antenna pattern. Angles use the actual departure/arrival ray.
Historical configurations without angular_model retain their scalar baseline.
"""
import numpy as np


def _unit(value):
    value = np.asarray(value, float)
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("Antenna directions must be finite three-vectors")
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(length <= 0):
        raise ValueError("Antenna direction cannot be zero")
    return value/length


def _angle(a, b):
    a, b = _unit(a), _unit(b)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                np.sum(a*b, axis=-1)))


def path_antenna_gains(satellite_m, receiver_m, propagation,
                       departure_direction=None, arrival_direction=None):
    """Return per-ray absolute TX/RX power gains in dBi (scalar or vector).

    Departure points away from the transmitter; arrival points from receiver
    toward the apparent source. Identical array elements use the same RX
    element pattern; no array beamforming gain is added here.
    """
    satellite, receiver = np.asarray(satellite_m, float), np.asarray(receiver_m, float)
    tx_los, rx_los = _unit(receiver-satellite), _unit(satellite-receiver)
    tx_ray = tx_los if departure_direction is None else _unit(departure_direction)
    rx_ray = rx_los if arrival_direction is None else _unit(arrival_direction)
    beam = propagation['beam']
    model = beam.get('angular_model', 'legacy_constant_pointing')
    if model not in ('legacy_constant_pointing', 'capped_parabolic'):
        raise ValueError("Unknown beam.angular_model")
    if model == 'capped_parabolic':
        tx_mode = beam.get('tx_pointing_mode', 'track_receiver')
        rx_mode = beam.get('rx_pointing_mode', 'fixed_enu')
        if tx_mode == 'track_receiver':
            tx_bore = _unit(np.asarray(beam['tx_reference_receiver_itrs_m'])-satellite) if 'tx_reference_receiver_itrs_m' in beam else tx_los
        elif tx_mode == 'fixed_itrs':
            tx_bore = _unit(beam['tx_boresight_itrs'])
        else:
            raise ValueError("TX pointing mode must be track_receiver or fixed_itrs")
        if rx_mode == 'fixed_enu':
            # Import lazily: reflection also uses this common antenna model.
            from src.reflection import station_axes
            az = float(beam['rx_boresight_az_deg'])
            el = float(beam['rx_boresight_el_deg'])
            if not np.isfinite([az, el]).all() or not -90 <= el <= 90:
                raise ValueError("Invalid fixed RX antenna boresight")
            station = receiver.reshape(-1, 3)[0]
            origin = np.asarray(beam.get('rx_frame_origin_itrs_m', station), float)
            if not np.allclose(receiver, station, atol=1e-6, rtol=0):
                raise ValueError("fixed_enu antenna requires a fixed receiver position")
            az, el = np.radians([az, el])
            local = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
            rx_bore = np.column_stack(station_axes(origin))@local
        elif rx_mode == 'track_los':
            rx_bore = rx_los
        else:
            raise ValueError("RX pointing mode must be fixed_enu or track_los")
    else:
        tx_mode = rx_mode = 'legacy_los_relative'
        tx_bore, rx_bore = tx_los, rx_los
    tx_angle, rx_angle = _angle(tx_ray, tx_bore), _angle(rx_ray, rx_bore)
    width = float(beam['half_power_beamwidth_deg'])
    tx_width = float(beam.get('tx_half_power_beamwidth_deg', width))
    rx_width = float(beam.get('rx_half_power_beamwidth_deg', width))
    rejection = float(beam.get('max_rejection_db',
        propagation.get('reflection', {}).get('max_antenna_rejection_db', 40.)))
    pointing_error = float(beam.get('pointing_error_deg', 0.))
    peaks = [float(beam['tx_gain_dbi']), float(beam['rx_gain_dbi'])]
    if (not np.isfinite([width, tx_width, rx_width, rejection, pointing_error, *peaks]).all()
            or min(width, tx_width, rx_width) <= 0 or rejection < 0):
        raise ValueError("Invalid antenna power gains, beam widths or rejection")
    tx_loss = np.minimum(12*(tx_angle/tx_width)**2, rejection)
    rx_loss = np.minimum(12*(rx_angle/rx_width)**2, rejection)
    if model == 'legacy_constant_pointing':
        # Retain the historical one common pointing loss for old audit configs.
        rx_loss = rx_loss + 12*(pointing_error/width)**2
    if not propagation['enabled']['beam']:
        tx_gain, rx_gain = np.zeros_like(tx_angle), np.zeros_like(rx_angle)
    else:
        tx_gain, rx_gain = peaks[0]-tx_loss, peaks[1]-rx_loss
    return dict(tx_gain_dbi=tx_gain, rx_gain_dbi=rx_gain,
                combined_gain_dbi=tx_gain+rx_gain,
                tx_offaxis_deg=tx_angle, rx_offaxis_deg=rx_angle,
                antenna_model=model, tx_pointing_mode=tx_mode, rx_pointing_mode=rx_mode)


def path_to_los_gain(satellite_m, receiver_m, scatterer_m, propagation):
    """Voltage ratio sqrt(GT_path GR_path / (GT_LoS GR_LoS))."""
    ray = path_antenna_gains(satellite_m, receiver_m, propagation,
        np.asarray(scatterer_m)-satellite_m, np.asarray(scatterer_m)-receiver_m)
    direct = path_antenna_gains(satellite_m, receiver_m, propagation)
    ratio = 10**((ray['combined_gain_dbi']-direct['combined_gain_dbi'])/20)
    return ratio, ray
