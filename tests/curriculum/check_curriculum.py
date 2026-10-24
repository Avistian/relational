"""Read-only manifest-page coverage audit under Piotr's grouped-row decision.

Only explicit identifiers in a table's first cell establish intent: exact IDs,
closed numeric ranges, or comma/slash lists. A parent ID never implies suffixes.
Planned rows may precede publication, but cannot label a published page planned.
This audits row coverage, not lesson existence, paper fidelity or learner mastery.
"""
import argparse
import json
from pathlib import Path
import re

ID = r'(B?)([0-9]+)([a-z]?)'


def canonical(value):
    match = re.fullmatch(ID, str(value).strip(), re.I)
    if not match:
        raise ValueError('invalid lesson id: ' + str(value))
    prefix, number, suffix = match.groups()
    return prefix.upper() + (str(int(number)).zfill(2) if prefix else str(int(number))) + suffix.lower()


def row_ids(cell):
    cell = re.sub(r'[★◆*`]', '', cell).strip()
    identities = set()
    for part in re.split(r'[,/]', cell):
        part = part.strip()
        if re.fullmatch(ID, part, re.I):
            identities.add(canonical(part))
            continue
        match = re.fullmatch(r'(B?)([0-9]+)\s*[-–—]\s*(B?)([0-9]+)', part, re.I)
        if not match:
            return set()
        prefix, start, end_prefix, end = match.groups()
        if end_prefix and prefix.upper() != end_prefix.upper():
            return set()
        start, end = int(start), int(end)
        if not 0 < start <= end <= 10000:
            return set()
        identities.update(canonical(prefix + str(i)) for i in range(start, end + 1))
    return identities


def audit(root):
    report = dict(status='BLOCKED', manifest_pages=0, covered_pages=0,
                  pages=[], rows=[], issues=[])
    try:
        entries = json.loads((root / 'lessons/manifest.json').read_text())['lessons']
        if not isinstance(entries, list) or not entries:
            raise ValueError('manifest lessons must be a nonempty list')
        seen = set()
        for entry in entries:
            identity = canonical(entry['id'])
            if identity in seen or not isinstance(entry['slug'], str) or not entry['slug']:
                raise ValueError('duplicate id or invalid slug: ' + identity)
            if not isinstance(entry.get('published'), bool):
                raise ValueError('published must be boolean: ' + identity)
            seen.add(identity)
            report['pages'].append(dict(id=identity, slug=entry['slug'],
                                         published=entry['published'], rows=[]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['issues'].append(dict(kind='invalid_manifest', detail=str(error)))
        return report
    report['manifest_pages'] = len(entries)
    try:
        curriculum = (root / 'CURRICULUM.md').read_text()
    except OSError as error:
        report['issues'].append(dict(kind='invalid_curriculum', detail=str(error)))
        return report
    for line_number, line in enumerate(curriculum.splitlines(), 1):
        if not line.lstrip().startswith('|'):
            continue
        cells = line.strip().strip('|').split('|')
        if len(cells) < 2:
            continue
        identities = row_ids(cells[0])
        if not identities:
            continue
        planned = bool(re.search(r'\bplanned\b|\bnot yet available\b', cells[1], re.I))
        row = dict(line=line_number, label=cells[0].strip(), ids=sorted(identities), planned=planned)
        report['rows'].append(row)
        for page in report['pages']:
            if page['id'] in identities:
                page['rows'].append(line_number)
                if planned and page['published']:
                    report['issues'].append(dict(kind='planned_row_disagreement',
                        id=page['id'], slug=page['slug'], line=line_number))
    for page in report['pages']:
        if not page['rows']:
            report['issues'].append(dict(kind='zero_row_coverage', id=page['id'], slug=page['slug']))
    report['covered_pages'] = sum(bool(page['rows']) for page in report['pages'])
    report['status'] = 'BLOCKED' if report['issues'] else 'PASS'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    report = audit(parser.parse_args().root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
