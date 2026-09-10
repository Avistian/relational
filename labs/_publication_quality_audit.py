"""Verify live lesson files against the published commit, not an evolving checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import urlsplit, unquote
from urllib.request import urlopen
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://avistian.github.io/relational/'


def verify(n, commit):
    commit = subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT, text=True).strip()
    runs = json.loads(subprocess.check_output(['gh', 'run', 'list', '--workflow', 'pages.yml', '--limit', '30',
                                              '--json', 'databaseId,headSha,status,conclusion,url'], cwd=ROOT, text=True))
    run = next(r for r in runs if r['headSha'] == commit)
    assert run['conclusion'] == 'success', run
    def blob(path):
        return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'lessons'], cwd=ROOT, text=True).splitlines()
    lesson = next(p for p in names if p.startswith(f'lessons/{n:04}-') and p.endswith('.html'))
    document = BeautifulSoup(blob(lesson), 'html.parser')
    ref = next(a['href'] for a in document.select('a[href]') if a['href'].startswith('../reference/') and 'glossary' not in a['href'])
    import posixpath
    reference = posixpath.normpath(posixpath.join('lessons', ref.split('#')[0]))
    paths = {lesson, reference, 'labs/html/' + PurePosixPath(lesson).name}
    for page in list(paths):
        for node in BeautifulSoup(blob(page), 'html.parser').select('[href],[src]'):
            url = urlsplit(node.get('href', node.get('src')))
            if url.scheme or url.netloc or not url.path:
                continue
            target = posixpath.normpath(posixpath.join(str(PurePosixPath(page).parent), unquote(url.path)))
            if target.endswith(('.json', '.css', '.js', '.png', '.svg', '.ipynb', '.md', '.py')):
                paths.add(target)
    def check(path):
        expected = blob(path)
        with urlopen(BASE + path + '?audit=' + commit, timeout=45) as response:
            actual = response.read()
            status = response.status
        assert actual == expected, (path, 'live bytes differ from committed artifact')
        return dict(path=path, http_status=status, sha256=hashlib.sha256(actual).hexdigest())
    records = list(ThreadPoolExecutor(max_workers=4).map(check, sorted(paths)))
    report = dict(status='PASS', lesson=n, commit=commit, pages_run=run, records=records,
                  scope='Successful Pages deployment and exact live HTTP bytes for lesson, prepared lab, reference and their directly linked assets/evidence/downloads; no live Colab check')
    output = Path(f'/tmp/quality-audit-{n:03}-publication.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', lesson=n, commit=commit, files=len(records), report=str(output))))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lesson', type=int)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    verify(args.lesson, args.commit)
