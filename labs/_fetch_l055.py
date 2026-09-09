"""Download three public TabReD release files, validating official SHA256 hashes."""
import hashlib
import json
import tarfile
import urllib.request
from urllib.parse import quote
from pathlib import Path

REVISION = 'b5ef15b3749f30da7a1eb8fba21a5b54d706bf32'
HASHES = {
    'ecom-offers': 'de7f96400b9006eab4bfa4318993506aa667c0508da32a5e00b912b2bc702bc9',
    'homesite-insurance': '9c94a9ba8a2dc68221c97d45273076714ad308b5e556ac59fd52117e1ea17d75',
    'sberbank-housing': 'a6a11bc09204a5d6d0015dfae504cd630186c813a87e384eaf8caf5b960eb334',
}
ROOT = Path(__file__).resolve().parent


def fetch(root=None):
    root = Path(root) if root else ROOT / 'data/cache/l055'
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        archive = root / (name + '.tabred')
        url = 'https://www.kaggle.com/api/v1/datasets/download/irubachev/tabred/' + quote('preprocessed/' + archive.name, safe='')
        if not archive.exists():
            partial = archive.with_suffix('.partial')
            urllib.request.urlretrieve(url, partial)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'{name}: unexpected archive; inspect release before proceeding')
            partial.replace(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'{name}: archive checksum mismatch')
        dest = root / name
        if not (dest / name / 'info.json').exists():
            with tarfile.open(archive) as stream:
                stream.extractall(dest, filter='data')
    return root


if __name__ == '__main__':
    print(fetch())
