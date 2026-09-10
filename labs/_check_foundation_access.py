"""The lesson-to-lab route must be present in static HTML, before the reading."""
from pathlib import Path
from urllib.parse import unquote,urlsplit
from bs4 import BeautifulSoup
from _foundation_config import SLUGS

ROOT=Path(__file__).resolve().parents[1]


def check(root=ROOT):
    directory=root/'labs/html/foundation-sequence.html'
    assert directory.exists(), 'Missing a single directory of the 58–70 lesson/lab packages'
    directory_soup=BeautifulSoup(directory.read_text(),'html.parser')
    gallery=BeautifulSoup((root/'notebooks.html').read_text(),'html.parser')
    for n in range(58,71):
        slug=f'{n:04}-{SLUGS[n]}'
        lesson=BeautifulSoup((root/'lessons'/f'{slug}.html').read_text(),'html.parser')
        launch=lesson.select_one('[data-lab-launch]')
        assert launch is not None, (n,'Missing prominent lab launcher')
        assert launch in list(lesson.find('h2').previous_elements), (n,'Lab access is buried below reading')
        for suffix in [f'/html/{slug}.html',f'/{slug}.ipynb']:
            assert any(a['href'].endswith(suffix) for a in launch.select('a[href]')),(n,suffix)
        assert launch.select_one('a[href*="colab.research.google.com"]'),n
        card=directory_soup.select_one(f'#lesson-{n}')
        assert card is not None and len(card.select('a[href]'))>=6,(n,'Incomplete package directory entry')
        assert gallery.select_one(f'a[href="labs/html/{slug}.html"]'),(n,'Gallery depends on JavaScript to expose this lab')
        prepared=BeautifulSoup((root/'labs/html'/f'{slug}.html').read_text(),'html.parser')
        toolbar=prepared.select_one('[data-lab-launch]')
        assert toolbar and toolbar.select_one('a[href*="colab.research.google.com"]'),(n,'Read-only lab has no run route')
        assert prepared.select_one('#lab-exercises'),(n,'No direct jump to exercises')
        reference=BeautifulSoup((root/'reference'/f'{slug}.html').read_text(),'html.parser')
        assert reference.select_one(f'a[href="../labs/html/{slug}.html"]'),(n,'Reference does not lead to lab')
    for a in directory_soup.select('a[href]'):
        value=urlsplit(a['href'])
        if value.scheme or value.netloc or not value.path:continue
        assert (directory.parent/unquote(value.path)).resolve().exists(),a['href']
    print('PASS: all 13 packages have static, early lesson/lab/run/reference routes')


if __name__=='__main__':check()
