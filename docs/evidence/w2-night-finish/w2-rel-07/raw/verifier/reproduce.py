"""Read-only REL-07 verifier; all generated sites/exports live outside source."""
import collections
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path('/tmp/lead-wt/night-w2-vrel07')
OUT = Path(__file__).resolve().parent
PY = str(ROOT / '.venv/bin/python')
RED = 'de7887c54a76dccba2e69c905036e2104fee3f5e'
GREEN = '8b961503896e9fb9ed4cc78235c60d2c11eeee85'
HEAD = '758820f0fc221065a5c5be0ff4337b609d4c0d4c'

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])

def run(name, argv, cwd=ROOT):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=60,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    (OUT / (name + '.stdout')).write_bytes(result.stdout)
    (OUT / (name + '.stderr')).write_bytes(result.stderr)
    save(name + '.command.json', {'argv': argv, 'cwd': str(cwd), 'exit_code': result.returncode})
    return result

def snapshot():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in git('ls-files', '-z').decode().split('\0') if p and (ROOT / p).is_file()}

# Independent parser/resolver: never imports check_links or its Document/audit.
class Scan(HTMLParser):
    def __init__(self, file):
        super().__init__()
        self.ids, self.urls = set(), []
        self.feed(file.read_text(encoding='utf-8'))
        self.close()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value is not None and (key == 'id' or (tag == 'a' and key == 'name')):
                self.ids.add(value)
        values = dict(attrs)
        self.urls.extend((self.getpos()[0], key, values[key])
                         for key in ('href', 'src', 'poster') if values.get(key) is not None)

    handle_startendtag = handle_starttag

def independent():
    deny = {'vendor', 'vendors', 'vendored', 'archive', 'archives',
            'third_party', 'third-party', 'node_modules'}
    banned = lambda p: bool({x.lower() for x in p.parts} & deny)
    pages = []
    for parent, dirs, files in os.walk(ROOT):
        rel = Path(parent).relative_to(ROOT)
        for name in files:
            p = rel / name
            scope = len(p.parts) == 1 or p.parts[0] in ('lessons', 'reference') or p.parts[:2] == ('labs', 'html')
            if scope and p.suffix == '.html' and not banned(p) and (ROOT / p).resolve().is_relative_to(ROOT):
                pages.append(p.as_posix())
        # Do not crawl unrelated trees (including ignored .venv).
        dirs[:] = [d for d in dirs if not (ROOT / rel / d).is_symlink() and not banned(rel / d)
                   and (len(rel.parts) == 0 and d in ('lessons', 'reference', 'labs')
                        or rel.parts and (rel.parts[0] in ('lessons', 'reference')
                                         or rel.parts[0] == 'labs' and (len(rel.parts) > 1 and rel.parts[1] == 'html' or len(rel.parts) == 1 and d == 'html')))]
    report = {'pages': sorted(pages), 'checked': [], 'broken': [], 'excluded': []}
    cache = {}
    def parse(p):
        if p not in cache:
            cache[p] = Scan(p)
        return cache[p]
    for page in sorted(pages):
        for line, attr, url in parse(ROOT / page).urls:
            row = dict(source=page, line=line, attribute=attr, url=url)
            try:
                u = urlsplit(url.strip())
            except ValueError:
                row['reason'] = 'invalid-url'
                report['checked'].append(row)
                report['broken'].append(row)
                continue
            if u.scheme or u.netloc or banned(Path(unquote(u.path))):
                row['reason'] = 'external-url' if u.scheme or u.netloc else 'excluded-tree'
                report['excluded'].append(row)
                continue
            decoded = unquote(u.path)
            base = ROOT if decoded.startswith('/') else (ROOT / page).parent
            dest = (base / decoded.lstrip('/')).resolve() if decoded else (ROOT / page).resolve()
            if dest.is_relative_to(ROOT) and dest.is_dir():
                dest = (dest / 'index.html').resolve()
            row['target'] = dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else None
            reason = None
            if row['target'] is None:
                reason = 'outside-root'
            elif not dest.is_file():
                reason = 'missing-file'
            elif dest.suffix.lower() in ('.html', '.htm'):
                frag = u.fragment.partition(':~:')[0]
                if frag and unquote(frag).lower() != 'top' and not ({frag, unquote(frag)} & parse(dest).ids):
                    reason = 'missing-fragment'
            if reason:
                row['reason'] = reason
                report['broken'].append(row)
            report['checked'].append(row)
    report['counts'] = {k: len(v) for k, v in report.items()}
    report['status'] = 'FAILED' if report['broken'] else 'OK'
    return report

assert git('rev-parse', 'HEAD').decode().strip() == HEAD
assert not git('status', '--porcelain')
before = snapshot()
save('tracked-before.json', before)
save('source-binding.json', {
    'head': HEAD, 'code': GREEN, 'red': RED, 'dirty_before': False,
    'delivery_delta': git('diff', '--name-only', GREEN, HEAD).decode().splitlines(),
    'actor': os.environ.get('LEAD_ACTOR'), 'role': os.environ.get('LEAD_ROLE'),
    'tdd_kind': 'Independent verification of frozen blobs; no authored tests or canonical replay'})

testpath = 'tests/links/test_links.py'
test_blobs = {sha: git('show', sha + ':' + testpath) for sha in (RED, GREEN, HEAD)}
assert len(set(test_blobs.values())) == 1
freeze = hashlib.sha256(test_blobs[RED]).hexdigest()
assert freeze == 'c36e71299d71cb0bf1129f18149239f6055ee695056ad73c3122e165b96f9386'
with tempfile.TemporaryDirectory(prefix='rel07-frozen-') as temporary:
    for name, sha in (('red', RED), ('green', GREEN)):
        export = Path(temporary) / name
        (export / 'tests/links').mkdir(parents=True)
        (export / testpath).write_bytes(test_blobs[sha])
        if name == 'green':
            (export / 'scripts').mkdir()
            (export / 'scripts/check_links.py').write_bytes(git('show', sha + ':scripts/check_links.py'))
        else:
            assert subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', sha + ':scripts/check_links.py'], capture_output=True).returncode != 0
        result = run('frozen-' + name, [PY, '-B', '-m', 'unittest', 'discover', '-s', 'tests/links', '-p', 'test_*.py', '-v'], export)
        assert result.returncode == (1 if name == 'red' else 0)
        if name == 'red':
            assert b'FAILED (failures=1)' in result.stderr and b'Published HTML checker CLI must exist' in result.stderr
        else:
            assert b'Ran 1 test' in result.stderr and b'\nOK\n' in result.stderr
save('frozen-binding.json', {'test_sha256': freeze, 'unchanged_red_green_delivery': True,
                           'red_exit': 1, 'red_failure': 'checker CLI absent assertion', 'green_exit': 0})

focused = run('focused', [PY, '-B', '-m', 'unittest', 'discover', '-s', 'tests/links', '-p', 'test_*.py', '-v'])
assert focused.returncode == 0
live = run('inventory', [PY, '-B', 'scripts/check_links.py'])
assert live.returncode == 1
actual = json.loads(live.stdout)
again = run('inventory-repeat', [PY, '-B', 'scripts/check_links.py'])
assert (live.returncode, live.stdout, live.stderr) == (again.returncode, again.stdout, again.stderr)
other = independent()
save('independent-inventory.json', other)
assert other == actual, 'Independent census differs'
save('census-comparison.json', {'equal_all_records': True, 'counts': actual['counts'],
                              'reasons': dict(collections.Counter(x['reason'] for x in actual['broken']))})

# New external disposable cases against real CLI, with explicit expected outcomes.
with tempfile.TemporaryDirectory(prefix='rel07-probes-') as temporary:
    site = Path(temporary) / 'site'
    site.mkdir()
    def write(name, text):
        p = site / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
    links = [
        ('/lessons/a%20b.HTML#caf%C3%A9', None),
        ('lessons/a%20b.HTML#literal%20id', None),
        ('lessons/a%20b.HTML#literal%2520id', None),
        ('lessons/a%20b.HTML#old', None),
        ('lessons/a%20b.HTML#CAF%C3%89', 'missing-fragment'),
        ('lessons/a%20b.HTML#caf%C3%A9:~:text=test', None),
        ('#self', None), ('#ToP', None), ('reference/?q=x#r', None),
        ('assets/data%20file.pdf#missing', None),
        ('../escape.html', 'outside-root'), ('outside.html', 'outside-root'),
        ('dir/', 'outside-root'), ('missing/', 'missing-file'),
        ('https://example.invalid/', 'excluded'), ('//example.invalid/', 'excluded'),
        ('data:text/html,hello', 'excluded'), ('javascript:void(0)', 'excluded'),
        ('%76endor/missing', 'excluded'), ('ARCHIVE/missing', 'excluded'),
        ('node_modules/missing', 'excluded'), ('third-party/missing', 'excluded'),
        ('http://[invalid', 'invalid-url')]
    write('index.html', '<h1 id="self"></h1>\n' + '\n'.join('<a href="' + u + '">x</a>' for u, _ in links))
    write('lessons/a b.HTML', '<h1 id="café"></h1><i id="literal%20id"></i><a name="old"></a>')
    write('reference/index.html', '<p id="r"></p><img src="../assets/data%20file.pdf"><video poster="/assets/data%20file.pdf"></video>')
    write('assets/data file.pdf', 'fixture')
    write('lessons/.hidden/p.html', '')
    for excluded in ('lessons/vendor/a.html', 'reference/archive/a.html', 'labs/sources/a.html', 'docs/a.html'):
        write(excluded, '<a href="absent">ignored</a>')
    outside = Path(temporary) / 'outside.html'
    outside.write_text('outside sentinel')
    (site / 'outside.html').symlink_to(outside)
    (site / 'dir').mkdir()
    (site / 'dir/index.html').symlink_to(outside)
    def site_snapshot():
        return {str(p.relative_to(site)): hashlib.sha256(p.read_bytes()).hexdigest() for p in site.rglob('*') if p.is_file()}
    snap = site_snapshot()
    result = run('external-probes', [PY, '-B', str(ROOT / 'scripts/check_links.py'), '--root', str(site)])
    assert result.returncode == 1
    report = json.loads(result.stdout)
    outcomes = {r['url']: r.get('reason') for r in report['checked']}
    outcomes.update({r['url']: 'excluded' for r in report['excluded']})
    for url, expected in links:
        assert outcomes[url] == expected, (url, outcomes[url], expected)
    assert site_snapshot() == snap and outside.read_text() == 'outside sentinel'
    assert report['pages'] == ['index.html', 'lessons/.hidden/p.html', 'reference/index.html']
    save('external-probes-summary.json', {'passed': len(links), 'expected': links, 'read_only': True,
                                         'scope_pages': report['pages'], 'src_poster_checked': True})

# Bind retained NEW builder logs, without reading deleted old worktree evidence.
parent = OUT.parent
ready = json.loads((parent / 'READY.json').read_text())
checks = {name: hashlib.sha256((parent / name).read_bytes()).hexdigest() == digest
          for name, digest in ready['rawproof']['sha256'].items()}
save('retained-raw-hash-check.json', checks)
assert all(checks.values())
after = snapshot()
save('tracked-after.json', after)
assert after == before
assert not git('status', '--porcelain')
save('result.json', {'verdict': 'PASS', 'focused': 'PASS', 'independent_census': 'PASS',
                    'tracked_files': len(before), 'tracked_bytes_unchanged': True,
                    'frozen_red_green': 'PASS', 'canonical_execution': 'NOT_RUN',
                    'configured_verify_project': 'NOT_RUN: configured suite calls model.fit',
                    'source_head': HEAD, 'source_dirty': False})
print(json.dumps({'verdict': 'PASS', 'counts': actual['counts'], 'tracked_files': len(before)}))
