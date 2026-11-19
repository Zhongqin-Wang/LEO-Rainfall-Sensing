"""Run a short synthetic satellite pass from geometry to rainfall retrieval."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'example')
    output = parser.parse_args().output.resolve()
    config = ROOT / 'configs' / 'example.yaml'
    steps = (
        ('run_geometry.py', '--config', config, '--output', output),
        ('run_link.py', '--config', config, '--geometry-dir', output, '--output', output, '--norad-id', '99999'),
        ('run_pass.py', '--root', output, '--satellite', 'satellite_0000'),
    )
    for step in steps:
        subprocess.run([sys.executable, *(str(arg) for arg in step)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
