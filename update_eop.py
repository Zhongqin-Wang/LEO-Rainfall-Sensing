"""Pin the installed astropy-iers-data IERS-A table and archive previous versions."""
import json
from src.earth_orientation import install_bundled

if __name__ == "__main__":
    print(json.dumps(install_bundled(), indent=2))
