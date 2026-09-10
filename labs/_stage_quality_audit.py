"""Validate one repaired lesson against a fresh copy of the real Pages build."""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def check(n):
    stage = Path(tempfile.mkdtemp(prefix=f'quality-audit-{n:03}-pages-'))
    workflow = (ROOT / '.github/workflows/pages.yml').read_text()
    block = workflow.split('        run: |\n', 1)[1].split('\n      - name: Setup Pages', 1)[0]
    lines = [line[10:] for line in block.splitlines() if line.strip()]
    # The manifest patch affects only root landing pages and would mutate the
    # working checkout. The lesson assets use the actual copy commands unchanged.
    lines = [line for line in lines if not line.startswith(('VER=', 'sed -i'))]
    script = re.sub(r'\bpublic\b', str(stage), '\n'.join(lines))
    subprocess.run(['bash', '-eu', '-c', script], cwd=ROOT, check=True)
    lesson = next((stage / 'lessons').glob(f'{n:04}-*.html'))
    parsed = BeautifulSoup(lesson.read_text(), 'html.parser')
    reference = next(a['href'] for a in parsed.select('a[href]')
                     if a['href'].startswith('../reference/') and 'glossary' not in a['href'])
    pages = [lesson, stage / 'labs/html' / lesson.name,
             (lesson.parent / reference.split('#')[0]).resolve()]
    links = 0
    broken = []
    documents = {}
    for page in pages:
        document = BeautifulSoup(page.read_text(), 'html.parser')
        for node in document.select('[href],[src]'):
            url = urlsplit(node.get('href', node.get('src')))
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            if not target.exists():
                broken.append([str(page.relative_to(stage)), url.path, 'missing file'])
                continue
            links += 1
            if url.fragment and target.suffix == '.html':
                if target not in documents:
                    documents[target] = BeautifulSoup(target.read_text(), 'html.parser')
                # Browsers first match the literal fragment, then its decoded
                # form. nbconvert emits percent signs in some heading IDs.
                fragments = {url.fragment, unquote(url.fragment)}
                if not any(documents[target].find(id=f) or documents[target].find('a', attrs={'name': f}) for f in fragments):
                    broken.append([str(page.relative_to(stage)), url.path, unquote(url.fragment), 'missing anchor'])
    report = dict(status='FAIL' if broken else 'PASS', lesson=n, stage=str(stage),
                  local_links=links, broken=broken,
                  scope='Real Pages copy commands in fresh tree, no symlinks; lesson/lab/reference files and HTML fragment targets')
    (ROOT / f'reviews/lesson-quality-audit-047-070/{n:03}-pages.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert not broken, f'{n}: copied Pages links or anchors failed'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lesson', type=int)
    check(parser.parse_args().lesson)
