#!/usr/bin/env python3
"""Read-only local HTML link inventory (standard library, no network).

Run ``python3 scripts/check_links.py [--root SITE_ROOT]``. JSON goes to stdout;
LEAD_RESULT goes to stderr; exit 1 means broken links, 2 means audit error.
Sources are root *.html plus lessons/, labs/html/, reference/ recursively.
Targets may be any file inside SITE_ROOT, with directory URLs using index.html.
This checks the chosen filesystem tree, not a deployed Pages artifact. HTML href,
src and poster attributes are checked; CSS/JS-generated URLs are not evaluated.
Fragments are checked on HTML targets only (id or legacy a[name]).
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


EXCLUDED = frozenset({'vendor', 'vendors', 'vendored', 'archive', 'archives',
                      'third_party', 'third-party', 'node_modules'})


def excluded(path):
    return any(part.lower() in EXCLUDED for part in path.parts)


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.anchors = set()
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id') is not None:
            self.anchors.add(attrs['id'])
        if tag == 'a' and attrs.get('name') is not None:
            self.anchors.add(attrs['name'])
        for attribute in ('href', 'src', 'poster'):
            if attrs.get(attribute) is not None:
                self.links.append((self.getpos()[0], attribute, attrs[attribute]))

    handle_startendtag = handle_starttag


def audit(root):
    """Return deterministic checked/broken/excluded inventories without writes."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Site root must be an existing directory')
    pages = set(root.glob('*.html'))
    for directory in ('lessons', 'labs/html', 'reference'):
        pages.update((root / directory).rglob('*.html'))
    pages = sorted(p for p in pages if not excluded(p.relative_to(root)))
    documents = {}

    def document(path):
        if path not in documents:
            documents[path] = Document(path.read_text(encoding='utf-8'))
        return documents[path]

    report = {'pages': [], 'checked': [], 'broken': [], 'excluded': []}
    for source in pages:
        if not source.resolve().is_relative_to(root):
            continue  # Never read symlinks outside the selected site.
        relative = source.relative_to(root).as_posix()
        report['pages'].append(relative)
        for line, attribute, url in document(source).links:
            record = {'source': relative, 'line': line, 'attribute': attribute, 'url': url}
            try:
                parsed = urlsplit(url.strip())
            except ValueError:
                record['reason'] = 'invalid-url'
                report['checked'].append(record)
                report['broken'].append(record)
                continue
            if parsed.scheme or parsed.netloc:
                report['excluded'].append(dict(record, reason='external-url'))
                continue
            path = unquote(parsed.path)
            if excluded(Path(path)):
                report['excluded'].append(dict(record, reason='excluded-tree'))
                continue
            target = ((root / path.lstrip('/')) if path.startswith('/') else
                      (source.parent / path) if path else source).resolve()
            reason = None
            if not target.is_relative_to(root):
                reason = 'outside-root'
            else:
                if target.is_dir():
                    target = (target / 'index.html').resolve()
                if not target.is_relative_to(root):
                    reason = 'outside-root'
                elif not target.is_file():
                    reason = 'missing-file'
                else:
                    raw_fragment = parsed.fragment.split(':~:', 1)[0]
                    fragment = unquote(raw_fragment)
                    if (fragment and target.suffix.lower() in ('.html', '.htm')
                            and raw_fragment not in document(target).anchors
                            and fragment not in document(target).anchors
                            and fragment.lower() != 'top'):
                        reason = 'missing-fragment'
            record['target'] = target.relative_to(root).as_posix() if target.is_relative_to(root) else None
            if reason:
                record['reason'] = reason
                report['broken'].append(record)
            report['checked'].append(record)
    report['status'] = 'FAILED' if report['broken'] else 'OK'
    report['counts'] = {key: len(report[key]) for key in ('pages', 'checked', 'broken', 'excluded')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        report = audit(args.root)
    except (OSError, UnicodeError, ValueError) as error:
        print(json.dumps({'status': 'FAILED', 'error': str(error)}, indent=2))
        print('LEAD_RESULT: FAILED audit-error', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('LEAD_RESULT: {} pages={} checked={} broken={} excluded={}'.format(
        report['status'], *(report['counts'][key] for key in ('pages', 'checked', 'broken', 'excluded'))), file=sys.stderr)
    return int(bool(report['broken']))


if __name__ == '__main__':
    sys.exit(main())
