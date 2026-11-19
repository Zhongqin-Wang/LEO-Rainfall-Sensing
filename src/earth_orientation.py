"""Explicit, pinned IERS-A data; never silently substitute UT1=UTC or mean poles."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from astropy.time import Time
from astropy.utils import iers
import astropy_iers_data
from importlib.metadata import version
from datetime import datetime, timezone

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data/earth_orientation/finals2000A.all"
_active_path = DEFAULT_PATH
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "error"


def install_bundled(path=DEFAULT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(astropy_iers_data.__file__).parent / "data/finals2000A.all"
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    archive = path.parent / "archive" / f"finals2000A_{digest[:16]}.all"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_bytes(payload)
    if path.exists():
        old = path.read_bytes()
        old_archive = archive.parent / f"finals2000A_{hashlib.sha256(old).hexdigest()[:16]}.all"
        if not old_archive.exists():
            old_archive.write_bytes(old)
    shutil.copyfile(archive, path)
    info = dict(source="astropy-iers-data bundled IERS-A", package_version=version("astropy-iers-data"),
                source_documentation="https://docs.astropy.org/en/stable/utils/iers.html",
                sha256=digest, archive_path=str(archive.resolve()))
    path.with_name(path.name + ".metadata.json").write_text(json.dumps(info, indent=2))
    archive_metadata=archive.with_name(archive.name+".metadata.json")
    if not archive_metadata.exists():
        archive_metadata.write_text(json.dumps(info,indent=2))
    _read_table.cache_clear()
    return info


@lru_cache(maxsize=8)
def _read_table(path, digest):
    table = iers.IERS_A.open(str(path))
    if len(table) < 2:
        raise ValueError("Earth-orientation table is empty")
    return table


def ensure_earth_orientation(obstime, path=None):
    global _active_path
    if path is not None:
        _active_path = Path(path).resolve()
    path = _active_path
    if not path.exists():
        if path != DEFAULT_PATH:
            raise ValueError(f"Configured IERS-A file does not exist: {path}")
        install_bundled(path)
    payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    table = _read_table(str(path),payload_hash)
    if isinstance(obstime,str):
        try:
            parsed = datetime.fromisoformat(obstime.replace("Z","+00:00"))
        except ValueError:
            parsed = obstime
        if isinstance(parsed,datetime) and parsed.tzinfo is None:
            parsed=parsed.replace(tzinfo=timezone.utc)
        time=Time(parsed,scale="utc")
    else:
        time = obstime if isinstance(obstime, Time) else Time(obstime, scale="utc")
    _, status_ut1 = table.ut1_utc(time, return_status=True)
    _, _, status_pm = table.pm_xy(time, return_status=True)
    statuses = np.r_[np.atleast_1d(status_ut1), np.atleast_1d(status_pm)]
    if np.any(statuses < 0):
        raise ValueError("Observation time is outside the pinned IERS-A table. Update the "
                         "astropy-iers-data package and run update_eop.py, or select a matching saved table.")
    iers.earth_orientation_table.set(table)
    sidecar = path.with_name(path.name + ".metadata.json")
    provenance = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if provenance.get("sha256", payload_hash) != payload_hash:
        raise ValueError("Earth-orientation file does not match its provenance")
    return dict(path=str(path.resolve()), sha256=payload_hash,
                min_utc=Time(table["MJD"][0], format="mjd").isot + "Z",
                max_utc=Time(table["MJD"][-1], format="mjd").isot + "Z",
                uses_predictions=bool(np.any(statuses == iers.FROM_IERS_A_PREDICTION)),
                provenance=provenance)
