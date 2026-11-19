import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    ITRS,
    TEME,
)
from astropy.time import Time
from src.earth_orientation import ensure_earth_orientation


def predict_orbit(satellite, time_utc, station, carrier_hz, prediction_config=None):
    time = Time(time_utc, scale="utc")
    ensure_earth_orientation(time)

    error, position, velocity = satellite.sgp4(time.jd1, time.jd2)
    if error:
        raise ValueError(f"SGP4 prediction failed: {error}")

    state = CartesianRepresentation(
        np.asarray(position) * u.km,
        differentials=CartesianDifferential(
            np.asarray(velocity) * u.km / u.s
        ),
    )

    predicted = TEME(
        state, obstime=time
    ).transform_to(ITRS(obstime=time))

    location = EarthLocation.from_geodetic(
        lon=float(station["lon_deg"]) * u.deg,
        lat=float(station["lat_deg"]) * u.deg,
        height=(
            float(station["height_m"])
            + float(station["antenna_height_m"])
        ) * u.m,
    )

    # Controlled prediction errors in ITRS coordinates.
    settings = prediction_config or {}
    position_error_m = np.asarray(settings.get("position_error_m", [100., -50., 30.]))
    velocity_error_m_s = np.asarray(settings.get("velocity_error_m_s", [.1, -.05, .02]))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (position_error_m, velocity_error_m_s)):
        raise ValueError("Orbit prediction errors must be finite length-three vectors")

    satellite_position = (
        predicted.cartesian.xyz.to_value(u.m) + position_error_m
    )
    satellite_velocity = (
        predicted.velocity.d_xyz.to_value(u.m / u.s)
        + velocity_error_m_s
    )
    station_position = location.get_itrs(
        obstime=time
    ).cartesian.xyz.to_value(u.m)

    relative_position = satellite_position - station_position
    distance = np.linalg.norm(relative_position)
    direction = relative_position / distance
    range_rate = float(np.dot(direction, satellite_velocity))

    c = 299792458.0
    f0 = float(carrier_hz)
    doppler_hz = -f0 * range_rate / c

    return {
        "predicted_doppler_hz": doppler_hz,
        "predicted_time_scale": 1.0 + doppler_hz / f0,
    }
