"""Check repaired lesson packages without requiring historical code to stay frozen.

This read-only content check records current notebook fingerprints. It establishes
delivery consistency, not correctness of claims, source parity, or reproduction.
Use the numbered human review and lesson-specific behavioral checks for those.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat
from bs4 import BeautifulSoup
from nbconvert.filters.markdown import markdown2html_mistune

from _architecture_revision import PANELS
from _lesson_depth import authored

ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / 'labs'
REVIEWS = ROOT / 'reviews/lesson-quality-audit-047-070'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plain(node):
    return re.sub(r'\s+', ' ', node.get_text(' ', strip=True)).strip()


def check_lesson(n, require_review=True):
    lesson = next((ROOT / 'lessons').glob(f'{n:04}-*.html'))
    notebook = LABS / f'{lesson.stem}.ipynb'
    soup = BeautifulSoup(lesson.read_text(), 'html.parser')
    blocks, _ = authored(n)
    expected = {b['key']: b['text'] for b in blocks}
    assert set(expected) == {x['data-lesson-depth'] for x in soup.select('[data-lesson-depth]')}, (n, 'lesson depth keys')
    for key, source in expected.items():
        wanted = plain(BeautifulSoup(markdown2html_mistune(source), 'html.parser'))
        assert plain(soup.select_one(f'[data-lesson-depth="{key}"]')) == wanted, (n, key, 'lesson prose drift')
    assert len(soup.select('.arch-atlas')) == len(PANELS.get(n, [])), (n, 'architecture count')
    records = {}
    prose_by_kind = {}
    for kind, path in [('student', notebook), ('solution', LABS / 'solutions' / notebook.name)]:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        prose_by_kind[kind] = [c.source for c in nb.cells if c.cell_type == 'markdown']
        assert {c.metadata['lesson_depth']: c.source for c in nb.cells if c.metadata.get('lesson_depth')} == expected, (n, kind, 'notebook prose drift')
        diagrams = {c.metadata['architecture_revision']: c for c in nb.cells if c.metadata.get('architecture_revision')}
        assert set(diagrams) == {p['key'] for p in PANELS.get(n, [])}, (n, kind, 'architecture keys')
        for key, cell in diagrams.items():
            encoded = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', cell.source)
            assert encoded, (n, key, 'missing portable image')
            assert base64.b64decode(encoded[1], validate=True) == (LABS / 'figures/architecture-revision' / f'{n:04}-{key}.png').read_bytes(), (n, key, 'stale portable image')
        images = 0
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                assert 'attachment:' not in cell.source, (n, 'nonportable attachment')
                for encoded in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', cell.source):
                    assert base64.b64decode(encoded, validate=True).startswith(b'\x89PNG\r\n\x1a\n')
                    images += 1
        code = [c for c in nb.cells if c.cell_type == 'code']
        if kind == 'student':
            assert all(c.execution_count is None and not c.outputs for c in code), (n, 'student contains execution')
            assert any('____' in c.source or 'raise NotImplementedError(' in c.source for c in code), (n, 'missing student blanks')
        else:
            assert all(c.execution_count is not None for c in code), (n, 'solution not fully executed')
            assert not any(o.output_type == 'error' for c in code for o in c.outputs), (n, 'solution contains errors')
        records[kind] = dict(code_cells=len(code), images=images,
                             code_sha256=digest([c.source for c in code]),
                             execution_sha256=digest([dict(source=c.source, count=c.execution_count, outputs=c.outputs) for c in code]))
    assert prose_by_kind['student'] == prose_by_kind['solution'], (n, 'student/solution explanatory text differs')
    prepared = BeautifulSoup((LABS / 'html' / f'{lesson.stem}.html').read_text(), 'html.parser')
    prepared_text = plain(prepared)
    for i, source in enumerate(prose_by_kind['student']):
        expected_text = plain(BeautifulSoup(markdown2html_mistune(source), 'html.parser'))
        assert not expected_text or expected_text in prepared_text, (n, i, 'prepared notebook prose stale')
    references = [(lesson.parent / a['href'].split('#')[0]).resolve() for a in soup.select('a[href]')
                  if a['href'].startswith('../reference/') and 'glossary' not in a['href']]
    assert references, (n, 'missing reference')
    links = 0
    for page in {lesson, references[0], LABS / 'html' / f'{lesson.stem}.html'}:
        document = BeautifulSoup(page.read_text(), 'html.parser')
        ids = [x['id'] for x in document.select('[id]')]
        assert len(ids) == len(set(ids)), (n, str(page), 'duplicate IDs')
        for node in document.select('[href],[src]'):
            url = urlsplit(node.get('href', node.get('src')))
            if url.scheme or url.netloc or not url.path:
                continue
            target = (page.parent / unquote(url.path)).resolve()
            assert target.exists(), (n, str(page), url.path, 'broken local link')
            links += 1
    review = REVIEWS / f'{n:03}.md'
    if require_review:
        assert review.exists(), (n, 'individual review missing')
    return dict(lesson=n, status='PASS', local_links=links, notebooks=records,
                review=str(review.relative_to(ROOT)) if review.exists() else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lessons', type=int, nargs='*')
    parser.add_argument('--baseline', action='store_true', help='Allow absent reviews while establishing packaging baseline')
    args = parser.parse_args()
    results = [check_lesson(n, require_review=not args.baseline) for n in args.lessons or range(47, 71)]
    report = dict(status='PASS', lessons=results,
                  scope='Current canonical prose, portable diagrams, notebook execution records and local link consistency; no claim or reproduction certification',
                  live_colab='NOT_CHECKED', deployment='NOT_CHECKED')
    target = REVIEWS / ('baseline-delivery.json' if args.baseline else 'delivery.json')
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', lessons=[r['lesson'] for r in results], report=str(target.relative_to(ROOT)))))
