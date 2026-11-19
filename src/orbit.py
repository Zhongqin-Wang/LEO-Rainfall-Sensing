from __future__ import annotations
from src.models import TLEBatch

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    ITRS,
    TEME,
)
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import Satrec, SatrecArray
from sgp4 import omm
from src.earth_orientation import ensure_earth_orientation


SPEED_OF_LIGHT_KM_S = 299_792.458
iers.conf.auto_download = False

def load_orbit_batch(path: str | Path, allow_history=False) -> TLEBatch:
    """Load named TLE records or CelesTrak GP JSON, including six-digit IDs."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        records = json.loads(text)
        if not isinstance(records, list) or not records:
            raise ValueError("GP JSON must contain a nonempty array of orbit records")
        names, models, identifiers = [], [], set()
        for index, record in enumerate(records):
            try:
                fields = dict(record)
                # CelesTrak omits these fixed OMM values from JSON.
                for key, expected in {"CENTER_NAME": "EARTH", "REF_FRAME": "TEME",
                                      "TIME_SYSTEM": "UTC", "MEAN_ELEMENT_THEORY": "SGP4"}.items():
                    if fields.get(key, expected) != expected:
                        raise ValueError(f"Unsupported {key}: {fields[key]}")
                identifier = int(fields["NORAD_CAT_ID"])
                if identifier <= 0 or (identifier in identifiers and not allow_history):
                    raise ValueError(f"Invalid or duplicate NORAD_CAT_ID: {identifier}")
                epoch = datetime.fromisoformat(str(fields["EPOCH"]).replace("Z", "+00:00"))
                if epoch.tzinfo is None:
                    epoch = epoch.replace(tzinfo=timezone.utc)
                fields["EPOCH"] = epoch.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
                name = str(fields["OBJECT_NAME"]).strip()
                if not name:
                    raise ValueError("Missing satellite name")
                numeric_keys = ("MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
                                "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR",
                                "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT")
                if not all(np.isfinite(float(fields[key])) for key in numeric_keys):
                    raise ValueError("Nonfinite orbital element")
                if float(fields["MEAN_MOTION"]) <= 0 or not 0 <= float(fields["ECCENTRICITY"]) < 1:
                    raise ValueError("Invalid mean motion or eccentricity")
                model = Satrec()
                omm.initialize(model, fields)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Invalid GP JSON record {index}: {error}") from error
            names.append(name)
            models.append(model)
            identifiers.add(identifier)
        return TLEBatch(np.asarray(names, dtype=object), models, SatrecArray(models))

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    if not lines or len(lines) % 3 != 0:
        raise ValueError(f"TLE file has {len(lines)} non-empty lines; expected groups of 3")

    names: List[str] = []
    models: List[Satrec] = []
    for index in range(0, len(lines), 3):
        name, line1, line2 = lines[index : index + 3]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise ValueError(f"Invalid TLE record near line {index + 1}: {name}")
        for line in (line1, line2):
            checksum = sum(int(char) if char.isdigit() else int(char == "-") for char in line[:68]) % 10
            if len(line) != 69 or not line[68].isdigit() or checksum != int(line[68]):
                raise ValueError(f"Invalid TLE checksum near line {index + 1}: {name}")
        if line1[2:7] != line2[2:7]:
            raise ValueError(f"TLE satellite IDs disagree near line {index + 1}")
        names.append(name[2:].strip() if name.startswith("0 ") else name)
        models.append(Satrec.twoline2rv(line1, line2))
    return TLEBatch(
        names=np.asarray(names, dtype=object),
        models=models,
        propagator=SatrecArray(models),
    )


def load_tle_batch(path: str | Path) -> TLEBatch:
    """Backward-compatible entry point; also accepts GP JSON."""
    return load_orbit_batch(path)

def sample_times(
    start_time: datetime,
    duration_hours: float,
    step_seconds: float,
) -> Tuple[List[datetime], Time]:
    if start_time.tzinfo is None:
        raise ValueError("start_time must include a timezone")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be positive")

    start_utc = start_time.astimezone(timezone.utc)
    duration_s = duration_hours * 3600.0
    offsets = np.arange(0.0, duration_s, float(step_seconds))
    if offsets.size == 0 or not np.isclose(offsets[-1], duration_s, rtol=0, atol=1e-9):
        offsets = np.append(offsets, duration_s)
    datetimes = [
        start_utc + timedelta(seconds=float(offset))
        for offset in offsets
    ]
    if any(b <= a for a, b in zip(datetimes, datetimes[1:])):
        raise ValueError("Sample times exceed datetime microsecond resolution")
    return datetimes, Time(datetimes, scale="utc")

def _propagate_itrs(batch: TLEBatch, obstime: Time):
    ensure_earth_orientation(obstime)
    errors, positions_km, velocities_km_s = batch.propagator.sgp4(
        np.asarray(obstime.jd1, dtype=np.float64),
        np.asarray(obstime.jd2, dtype=np.float64),
    ) #Earth-centred TEME position (km) and velocity
    errors = np.asarray(errors)
    positions_km = np.asarray(positions_km)
    velocities_km_s = np.asarray(velocities_km_s)
    min_perigee: float = 300
    perigee_altitude_km = np.asarray(
        [
            model.altp * model.radiusearthkm #Convert the perigee altitude to kilometres
            for model in batch.models
        ],
        dtype=np.float64,
    )
    perigee_valid = (perigee_altitude_km >= min_perigee)
    '''
    rejected_indices = np.flatnonzero(~perigee_valid)
    for index in rejected_indices:
        print(
            f"[Perigee filter] "
            f"index={index}, "
            f"name={batch.names[index]}, "
            f"perigee altitude={perigee_altitude_km[index]:.2f} km"
        )
    '''
    valid = (
        (errors == 0)
        & np.all(np.isfinite(positions_km), axis=-1)
        & np.all(np.isfinite(velocities_km_s), axis=-1)
        & perigee_valid[:, np.newaxis]
    )

    representation = CartesianRepresentation(
        positions_km[..., 0] * u.km,
        positions_km[..., 1] * u.km,
        positions_km[..., 2] * u.km,
        differentials=CartesianDifferential(
            velocities_km_s[..., 0] * u.km / u.s,
            velocities_km_s[..., 1] * u.km / u.s,
            velocities_km_s[..., 2] * u.km / u.s,
        ),
    ) #Attach physical units
    teme = TEME(representation, obstime=obstime)#SGP4 TEME frame
    return valid, teme.transform_to(ITRS(obstime=obstime)) #TEME → ITRS

def _label_visible_passes(visible: np.ndarray):
    padded = np.pad(visible.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    sample_count = visible.size
    return [
        (int(start), int(stop), start > 0 and stop < sample_count)
        for start, stop in zip(starts, stops)
    ]

def compute_geometry(
    batch: TLEBatch,
    station_location: EarthLocation,
    datetimes: Sequence[datetime],
    obstime: Time,
    carrier_hz: float,
    min_elevation_deg: float,
    rain_rate_mm_h: float = 0.0,
    satellite_indices: Sequence[int] | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if carrier_hz <= 0:
        raise ValueError("carrier_hz must be positive")
    if satellite_indices is None:
        satellite_indices = np.arange(len(batch.models))
    if len(satellite_indices) != len(batch.models):
        raise ValueError("Satellite indices must match the orbit batch")
    valid, satellite_itrs = _propagate_itrs(batch, obstime)
    station_itrs = station_location.get_itrs(obstime=obstime)#Ground receiver in the ITRS frame

    satellite_xyz_km = satellite_itrs.cartesian.xyz.to_value(u.km)
    station_xyz_km = station_itrs.cartesian.xyz.to_value(u.km)
    relative_xyz_km = satellite_xyz_km - station_xyz_km[:, np.newaxis, :]
    slant_range_km = np.linalg.norm(relative_xyz_km, axis=0) #Slant range\(r=\sqrt{x^2+y^2+z^2}\)

    los_unit = relative_xyz_km / slant_range_km[np.newaxis, ...]
    relative_velocity_km_s = satellite_itrs.velocity.d_xyz.to_value(u.km / u.s)
    slant_range_rate_km_s = np.sum(
        los_unit * relative_velocity_km_s,
        axis=0,
    )
    doppler_hz = -(
        slant_range_rate_km_s / SPEED_OF_LIGHT_KM_S
    ) * carrier_hz #Doppler equation

    topocentric_itrs = ITRS(
        CartesianRepresentation(
            relative_xyz_km[0] * u.km,
            relative_xyz_km[1] * u.km,
            relative_xyz_km[2] * u.km,
        ),
        obstime=obstime,
        location=station_location,
    )
    altaz = topocentric_itrs.transform_to(
        AltAz(
            obstime=obstime,
            location=station_location,
            pressure=0 * u.hPa,
        )
    )
    elevation_deg = altaz.alt.to_value(u.deg)
    azimuth_deg = altaz.az.to_value(u.deg)
    visible = valid & (elevation_deg >= min_elevation_deg)& (elevation_deg <= 90)

    frames = []
    time_array = np.asarray(datetimes, dtype=object)
    for local_index, row in enumerate(visible):
        satellite_index = int(satellite_indices[local_index])
        for pass_number, (start, stop, complete) in enumerate(
            _label_visible_passes(row),
            start=1,
        ):
            indices = np.arange(start, stop)
            if indices.size == 0:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "time_utc": pd.to_datetime(time_array[indices], utc=True),
                        "station_x_km": station_xyz_km[0, indices],
                        "station_y_km": station_xyz_km[1, indices],
                        "station_z_km": station_xyz_km[2, indices],
                        "satellite_index": satellite_index,
                        "satellite_name": str(batch.names[local_index]),
                        "norad_id": int(batch.models[local_index].satnum),
                        "pass_id": f"{satellite_index:03d}-{pass_number:03d}",
                        "sat_x_km": satellite_xyz_km[0, local_index, indices],
                        "sat_y_km": satellite_xyz_km[1, local_index, indices],
                        "sat_z_km": satellite_xyz_km[2, local_index, indices],
                        "sat_vx_km_s": relative_velocity_km_s[0, local_index, indices],
                        "sat_vy_km_s": relative_velocity_km_s[1, local_index, indices],
                        "sat_vz_km_s": relative_velocity_km_s[2, local_index, indices],
                        "complete_pass": complete, #Complete pass window
                        "elevation_deg": elevation_deg[local_index, indices],
                        "azimuth_deg": azimuth_deg[local_index, indices],
                        "slant_range_km": slant_range_km[local_index, indices],
                        "slant_range_rate_km_s": slant_range_rate_km_s[
                            local_index, indices
                        ],
                        "doppler_hz": doppler_hz[local_index, indices],
                    }
                )
            )

    if not frames:
        if allow_empty:
            return pd.DataFrame()
        raise RuntimeError(
            "No visible samples found; extend duration or lower min_elevation_deg"
        )
    return pd.concat(frames, ignore_index=True)



class OrbitHistory:
    """Select latest past epoch available by observation time; preserve source indices.

    CREATION_DATE is a publication proxy, not proof of delivery to a receiver.
    Missing creation dates are excluded rather than assumed available.
    """
    def __init__(self, path):
        records = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        self.batch = load_orbit_batch(path, allow_history=True)
        self.epochs = pd.to_datetime([r['EPOCH'] for r in records], utc=True, format='mixed')
        self.created = pd.to_datetime([r.get('CREATION_DATE') for r in records], utc=True, format='mixed')
        self.ids = np.array([int(r['NORAD_CAT_ID']) for r in records])
        # Sort low to high priority, then retain the last eligible record per satellite.
        self.order = np.lexsort((np.arange(len(records)), self.created.asi8, self.epochs.asi8, self.ids))

    def select(self, time, max_age_days=3.):
        t = pd.Timestamp(time)
        if t.tzinfo is None:
            raise ValueError('Observation time must have a timezone')
        if not np.isfinite(max_age_days) or max_age_days <= 0:
            raise ValueError('Maximum orbit age must be positive')
        age = t.value-self.epochs.asi8
        valid = (age >= 0) & (age <= max_age_days*86400e9)
        valid &= (~self.created.isna()) & (self.created.asi8 <= t.value)
        ordered = self.order[valid[self.order]]
        if not len(ordered):
            return ordered
        ids = self.ids[ordered]
        return ordered[np.r_[ids[1:] != ids[:-1], True]]


def add_beam_visibility(geometry, propagation):
    """Half-power cone eligibility; not an SNR or acquisition-success guarantee."""
    result = geometry.copy()
    beam = propagation['beam']
    mode = beam.get('rx_pointing_mode', 'fixed_enu')
    if mode == 'track_los':
        angle = np.zeros(len(result))
    elif mode == 'fixed_enu':
        az, el = np.radians(result[['azimuth_deg', 'elevation_deg']].to_numpy()).T
        ba, be = np.radians([beam['rx_boresight_az_deg'], beam['rx_boresight_el_deg']])
        cosine = np.sin(el)*np.sin(be)+np.cos(el)*np.cos(be)*np.cos(az-ba)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    else:
        raise ValueError('Unsupported RX pointing mode for observation selection')
    width = float(beam.get('rx_half_power_beamwidth_deg', beam['half_power_beamwidth_deg']))
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Beam width must be positive')
    result['rx_offaxis_deg'] = angle
    result['in_rx_beam'] = angle <= width/2+1e-9
    result['rx_pointing_mode'] = mode
    return result
