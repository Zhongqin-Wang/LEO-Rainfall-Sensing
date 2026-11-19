"""Read one BOM rain-rate sample without generating IQ or using orbit truth."""
import argparse
import json
from pathlib import Path
from src.bom_rain import BOMRain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent/'data/bom')
    parser.add_argument('--time', default='2026-10-12T13:20:01Z')
    parser.add_argument('--lat', type=float, default=-34.58374)
    parser.add_argument('--lon', type=float, default=150.15013)
    args = parser.parse_args()
    result = BOMRain(args.directory).query_rain(dict(lat_deg=args.lat, lon_deg=args.lon), args.time)
    if not result['valid_mask']:
        result['rain_rate_mm_h'] = None
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
