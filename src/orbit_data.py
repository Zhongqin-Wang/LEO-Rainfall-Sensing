"""Versioned CelesTrak snapshots and per-satellite epoch checks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
from astropy.time import Time

from src.orbit import load_orbit_batch


STARLINK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json"
MIN_DOWNLOAD_INTERVAL_S = 2 * 3600


def configured_orbit_path(project_root, geometry_config):
    value = geometry_config.get("orbit_path") or geometry_config.get("tle_path")
    if not value:
        raise ValueError("Set geometry.orbit_path to a GP JSON or named TLE file")
    return Path(project_root) / value


def metadata_path(path):
    path = Path(path)
    return path.with_name(path.name + ".metadata.json")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def epoch_summary(batch):
    epochs = Time([s.jdsatepoch for s in batch.models],
                  [s.jdsatepochF for s in batch.models], format="jd", scale="utc")
    return {
        "satellite_count": len(batch.models),
        "epoch_min_utc": epochs[int(np.argmin(epochs.jd))].isot + "Z",
        "epoch_max_utc": epochs[int(np.argmax(epochs.jd))].isot + "Z",
        "six_digit_catalog_count": sum(s.satnum >= 100000 for s in batch.models),
    }


def read_provenance(path):
    sidecar = metadata_path(path)
    if not sidecar.exists():
        return None
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if provenance.get("sha256") != sha256_file(path):
        raise ValueError(f"Orbit file does not match its provenance: {path}")
    return provenance


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def install_snapshot(path, payload, provenance):
    """Validate before replacing current data; keep the exact downloaded bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        batch = load_orbit_batch(temporary)
        if not all(str(name).startswith("STARLINK-") for name in batch.names):
            raise ValueError("The Starlink response contains unexpected satellite names")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    downloaded_at = _utc(provenance["downloaded_at_utc"])
    digest = hashlib.sha256(payload).hexdigest()
    if provenance.get("sha256", digest) != digest:
        raise ValueError("Download checksum mismatch")
    archive = path.parent / "archive" / (
        "starlink_" + downloaded_at.strftime("%Y%m%dT%H%M%SZ") + "_" + digest[:12] + ".json"
    )
    info = dict(provenance)
    info.update(epoch_summary(batch), sha256=digest, format="CelesTrak GP JSON",
                downloaded_at_utc=downloaded_at.isoformat(), archive_path=str(archive.resolve()))
    encoded = (json.dumps(info, indent=2) + "\n").encode("utf-8")
    if archive.exists() and archive.read_bytes() != payload:
        raise ValueError(f"Refusing to overwrite an existing snapshot: {archive}")
    _atomic_write(archive, payload)
    _atomic_write(metadata_path(archive), encoded)
    _atomic_write(path, payload)
    _atomic_write(metadata_path(path), encoded)
    return info


def update_starlink(path, now=None, timeout=60):
    """One HTTP request at most; reuse verified downloads for at least two hours."""
    path = Path(path)
    if path.parent.name == "archive":
        raise ValueError("A dated archive is read-only; disable auto_update_orbits or select data/starlink.json")
    now = _utc(now or datetime.now(timezone.utc))
    if path.exists():
        provenance = read_provenance(path)
        if provenance and provenance.get("source_url") == STARLINK_URL:
            age = (now - _utc(provenance["downloaded_at_utc"])).total_seconds()
            if age < 0:
                raise ValueError("Orbit download timestamp is in the future; check the system clock")
            if age < MIN_DOWNLOAD_INTERVAL_S:
                return dict(provenance, update_status="cached")
    request = urllib.request.Request(STARLINK_URL, headers={
        "User-Agent": "LEO-Rainfall-Research-Simulator/1.0", "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status: {response.status}")
            payload = response.read(32 * 1024 * 1024 + 1)
            if len(payload) > 32 * 1024 * 1024:
                raise ValueError("Orbit response is unexpectedly large")
            provenance = {
                "source_url": STARLINK_URL, "resolved_url": response.url,
                "http_date": response.headers.get("Date"),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        info = install_snapshot(path, payload, provenance)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Starlink update failed; existing files were not replaced: {error}. "
            "No automatic retry. To use a saved snapshot, disable auto_update_orbits; "
            "epoch validation still applies."
        ) from error
    return dict(info, update_status="downloaded")


def eligible_orbits(batch, start, end, max_epoch_distance_days=3.0):
    """Require each epoch to be close to both ends of the requested time window.

    Three days is a configurable research screening threshold, not a promise of
    positional accuracy. Indices always refer to the unchanged snapshot order.
    """
    limit = float(max_epoch_distance_days)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("max_epoch_distance_days must be finite and positive")
    start, end = _utc(start), _utc(end)
    if end < start:
        raise ValueError("Observation end must not precede start")
    epochs = Time([s.jdsatepoch for s in batch.models],
                  [s.jdsatepochF for s in batch.models], format="jd", scale="utc")
    distances = np.maximum(np.abs((Time(start) - epochs).to_value("day")),
                           np.abs((Time(end) - epochs).to_value("day")))
    eligible = np.flatnonzero(distances <= limit)
    rejected = np.flatnonzero(distances > limit)
    report = dict(epoch_summary(batch), max_epoch_distance_days=limit,
                  observation_start_utc=start.isoformat(), observation_end_utc=end.isoformat(),
                  eligible_count=len(eligible), excluded_epoch_count=len(rejected),
                  excluded_epochs=[{"satellite_index": int(i), "norad_id": int(batch.models[i].satnum),
                                    "name": str(batch.names[i]), "epoch_utc": epochs[i].isot + "Z",
                                    "max_epoch_distance_days": float(distances[i])} for i in rejected])
    if not len(eligible):
        raise ValueError(
            f"No orbits within {limit:g} days of the requested observation window. "
            f"Snapshot epochs: {report['epoch_min_utc']} to {report['epoch_max_utc']}. "
            "Update current data or select a historical snapshot matching the simulation date."
        )
    return eligible, report
