"""Local tangent-plane two-ray geometry in the same WGS84 ITRS frame as orbit.

The specular footprint is local to a low ground antenna. Terrain, shadowing and
measured antenna sidelobes are outside this smooth-facet scenario model.
"""
import numpy as np
from src.antenna import path_antenna_gains
from astropy.coordinates import EarthLocation
from astropy import units as u

C = 299792458.0


def station_axes(station_xyz_m):
    location = EarthLocation.from_geocentric(*np.asarray(station_xyz_m), unit=u.m)
    lon, lat = location.lon.rad, location.lat.rad
    east = np.array([-np.sin(lon), np.cos(lon), 0.])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    return east, north, up


def validate_polarization(config):
    polarization = config["station"].get("polarization", "circular").lower()
    expected = {"h": 0., "v": 90., "circular": 45.}
    if polarization not in expected:
        raise ValueError("station.polarization must be h, v or circular")
    tilt = float(config["propagation"]["polarization_tilt_deg"])
    if not np.isfinite(tilt) or not np.isclose(tilt, expected[polarization]):
        raise ValueError(f"{polarization} polarization requires polarization_tilt_deg={expected[polarization]:g}")


def compute_ground_reflection(geometry, config, fs):
    satellite = np.asarray(geometry["satellite_xyz_km"], dtype=float) * 1000
    station = np.asarray(geometry["station_xyz_km"], dtype=float) * 1000
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (satellite, station)):
        raise ValueError("Reflection coordinates must be finite three-dimensional ITRS vectors")
    carrier = float(config["link"]["carrier_hz"])
    height = float(config["station"]["antenna_height_m"])
    if not np.isfinite([carrier, height, fs]).all() or min(carrier, fs) <= 0 or height < 0:
        raise ValueError("Invalid reflection frequency, sample rate or antenna height")
    _, _, up = station_axes(station)
    up = np.asarray(config.get('_ground_up_itrs', up), float)
    direct_vector = satellite - station
    direct = np.linalg.norm(direct_vector)
    vertical = float(np.dot(direct_vector, up))
    if direct <= 0 or vertical < -1e-6:
        raise ValueError("Ground reflection requires an above-horizon satellite")
    image_vector = direct_vector + 2 * height * up
    reflected = np.linalg.norm(image_vector)
    # Stable range difference; no subtraction of nearly equal 500-km ranges.
    excess = (4 * height * vertical + 4 * height**2) / (reflected + direct)
    sin_el = np.clip((vertical + 2 * height) / reflected, 0., 1.)
    cos_el_sq = 1 - sin_el**2
    epsilon = float(config["station"]["epsilon_r"])
    conductivity = float(config["station"]["conductivity"])
    if not np.isfinite([epsilon, conductivity]).all() or epsilon <= 0 or conductivity < 0:
        raise ValueError("Ground permittivity must be positive and conductivity nonnegative")
    material = epsilon - 1j * conductivity / (2*np.pi*carrier*8.8541878128e-12)
    root = np.sqrt(material - cos_el_sq)
    horizontal = (sin_el-root) / (sin_el+root)
    vertical_coefficient = (material*sin_el-root) / (material*sin_el+root)
    polarization = config["station"]["polarization"].lower()
    reflection_config = config["propagation"].get("reflection", {})
    if polarization == "h":
        coefficient = horizontal
    elif polarization == "v":
        coefficient = vertical_coefficient
    elif polarization == "circular":
        port = reflection_config.get("circular_port", "co")
        if port not in ("co", "cross"):
            raise ValueError("reflection.circular_port must be co or cross")
        coefficient = (horizontal + vertical_coefficient) / 2 if port == "co" else (vertical_coefficient-horizontal) / 2
    else:
        raise ValueError("Reflection polarization must be h, v or circular")
    # Direction of the virtual transmitter beneath the local facet.
    reflected_arrival = direct_vector - 2 * (vertical + height) * up
    angle = np.rad2deg(np.arccos(np.clip(np.dot(direct_vector, reflected_arrival) / (direct*reflected), -1., 1.)))
    beam = config["propagation"]["beam"]
    width = float(beam["half_power_beamwidth_deg"])
    floor = float(reflection_config.get("max_antenna_rejection_db", 40.))
    roughness = float(reflection_config.get("surface_rms_height_m", .002))
    if not np.isfinite([width, floor, roughness]).all() or width <= 0 or min(floor, roughness) < 0:
        raise ValueError("Invalid reflection antenna pattern or surface roughness")
    # Intersection of the image-receiver ray with the local ground plane.
    fraction = height/(vertical+2*height) if vertical+2*height > 0 else 0.
    ground_point = station-2*height*up+fraction*image_vector
    direct_gains = path_antenna_gains(satellite, station, config['propagation'])
    reflected_gains = path_antenna_gains(satellite, station, config['propagation'],
        ground_point-satellite, reflected_arrival)
    if beam.get('angular_model', 'legacy_constant_pointing') == 'legacy_constant_pointing':
        discrimination = min(12 * (angle / width)**2, floor) if config["propagation"]["enabled"]["beam"] else 0.
    else:
        discrimination = float(direct_gains['combined_gain_dbi']-reflected_gains['combined_gain_dbi'])
    roughness_factor = np.exp(-.5 * (4*np.pi*roughness*sin_el / (C/carrier))**2)
    relative = coefficient * direct/reflected * roughness_factor * 10**(-discrimination/20)
    delay = excess / C
    phase = np.angle(relative) - 2*np.pi*carrier*delay
    result = dict(delay_samples=delay*fs, gain_db=20*np.log10(max(abs(relative), 1e-300)),
                  phase_deg=float(np.rad2deg(phase)), ld_m=float(direct), lr_m=float(reflected),
                  excess_path_m=float(excess), relative_coefficient_real=float(relative.real),
                  relative_coefficient_imag=float(relative.imag), antenna_discrimination_db=discrimination,
                  roughness_amplitude=roughness_factor, grazing_angle_deg=float(np.rad2deg(np.arcsin(sin_el))),
                  ground_point_itrs_m=ground_point.tolist(),
                  arrival_direction_itrs=(reflected_arrival/np.linalg.norm(reflected_arrival)).tolist(),
                  antenna_relative_amplitude=float(10**(-discrimination/20)),
                  tx_gain_dbi=float(reflected_gains['tx_gain_dbi']),
                  rx_gain_dbi=float(reflected_gains['rx_gain_dbi']),
                  geometry_model="WGS84_ITRS_local_tangent_specular_facet")
    if "satellite_velocity_km_s" in geometry:
        velocity = np.asarray(geometry["satellite_velocity_km_s"], dtype=float) * 1000
        result["direct_range_rate_m_s"] = float(np.dot(direct_vector/direct, velocity))
        result["reflected_range_rate_m_s"] = float(np.dot(image_vector/reflected, velocity))
    return result


def channel_response(nfft, fs, loss_db, reflection=None):
    frequency = np.fft.fftfreq(nfft, 1/float(fs))
    amplitude = 10**(-float(loss_db)/20)
    response = np.full(nfft, amplitude, dtype=complex)
    if reflection is not None:
        coefficient = 10**(reflection["gain_db"]/20) * np.exp(1j*np.deg2rad(reflection["phase_deg"]))
        response += amplitude * coefficient * np.exp(-2j*np.pi*frequency*reflection["delay_samples"]/fs)
    return response
