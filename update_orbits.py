"""Download and archive the latest public Starlink GP data from CelesTrak."""
import argparse
import json
from pathlib import Path

import yaml

from src.orbit import load_orbit_batch
from src.orbit_data import configured_orbit_path, epoch_summary, read_provenance, update_starlink

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    parser.add_argument("--status", action="store_true", help="Inspect saved data without any network request")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    path = configured_orbit_path(ROOT, config["geometry"])
    if args.status:
        info = dict(epoch_summary(load_orbit_batch(path)), provenance=read_provenance(path))
    else:
        if path.suffix.lower() != ".json":
            parser.error("Set geometry.orbit_path to a .json file for current Starlink downloads")
        info = update_starlink(path)
    print(json.dumps(dict(path=str(path), **info), indent=2))


if __name__ == "__main__":
    main()
