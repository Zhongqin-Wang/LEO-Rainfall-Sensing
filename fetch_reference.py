"""Install the published Starlink waveform reference files with checksum verification."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / 'data' / 'starlink_public_reference'


def main():
    manifest_path = ROOT / 'reference_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        target = DESTINATION / item['file']
        expected_hash = item['sha256']
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash:
            print(f'Verified {target.name}')
            continue
        with urlopen(item['source_url'], timeout=30) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f'The published reference checksum changed: {target.name}')
        with NamedTemporaryFile(dir=DESTINATION, delete=False) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        temporary.replace(target)
        print(f'Installed {target.name}')
    (DESTINATION / 'provenance.json').write_bytes(manifest_path.read_bytes())


if __name__ == '__main__':
    main()
