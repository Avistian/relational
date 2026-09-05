"""Check Colab image packaging, not live Colab browser rendering.

Run from the repo root: .venv/bin/python labs/_check_notebook_images_l047.py
Colab's markdown image representation is an inline data URI (colabtools #3836).
Verify the downloadable notebook carries each complete PNG in that representation
and that the prepared HTML renders identical image bytes without attachments.
"""
import base64
import hashlib
import io
from pathlib import Path
import re

import nbformat
from bs4 import BeautifulSoup
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGE = re.compile(r'!\[[^\]]*\]\((data:image/png;base64,[A-Za-z0-9+/=]+)\)')


def image_hash(uri):
    data = base64.b64decode(uri.split(',', 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == 'PNG'
        image.verify()
    return hashlib.sha256(data).hexdigest()


def main():
    expected = sorted(hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (HERE / 'figures/l047').glob('*.png'))
    assert len(expected) == 9
    paths = [HERE / '0047-saint.ipynb']
    solution = HERE / 'solutions/0047-saint.ipynb'
    if solution.exists():
        paths.append(solution)
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        sources = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'markdown')
        assert '](attachment:' not in sources, f'{path}: Colab cannot resolve Jupyter attachment links'
        assert not any(c.get('attachments') for c in notebook.cells), 'Do not depend on attachment metadata'
        actual = sorted(image_hash(uri) for uri in IMAGE.findall(sources))
        assert actual == expected, f'{path}: missing, corrupted, or changed inline PNGs'
        print(f'PASS: {path.name}: all nine Colab-format inline PNGs match source figures')
    html = BeautifulSoup((HERE / 'html/0047-saint.html').read_text(), 'html.parser')
    actual = sorted(image_hash(img['src']) for img in html.select('img[src^="data:image/png;base64,"]'))
    assert actual == expected, 'Prepared HTML must carry the same nine PNGs'
    print('PASS: prepared HTML has the same nine images; live Colab UI not checked')


if __name__ == '__main__':
    main()
