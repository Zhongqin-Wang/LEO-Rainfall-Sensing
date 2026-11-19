"""Versioned scientific scope and source fingerprints for reproducible runs."""
import hashlib
import platform
from pathlib import Path
from importlib.metadata import version


def software_provenance(root):
    root = Path(root)
    files = sorted([*root.glob("*.py"), *root.glob("src/*.py"), root / "requirements.txt"])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return dict(python=platform.python_version(), packages={p: version(p) for p in
        ("numpy", "scipy", "pandas", "astropy", "sgp4", "itur", "astropy-iers-data", "miepython")}, source_sha256=hashes,
        propagation_revisions={"rain": "ITU-R P.838-3", "gas": "ITU-R P.676-12",
                               "cloud": "ITU-R P.840-7", "sky_noise": "ITU-R P.618-14 Eq.69",
                               "optional_complex_rain": "spherical Mie / Marshall-Palmer / P.840-7 water permittivity"},
        revision_note="Gas and cloud implementations are pinned older editions supported by itur 0.4.0, not the latest P.676-13 / P.840-9.",
        scope="Synthetic public-structure single-link experiment; coherent per-ray rain phase and finite moving-rain-scatterer S/X/R model, with explicit angular antenna gains, 1--4 synchronized receive channels, spherical per-channel geometry and fixed receive-chain PO. No measured-weather validation, operational beam scheduling or coherent inter-frame IQ clock model.")
