"""Visible L188 collection/replay contracts. No third-party runtime dependencies."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit

ATOM = '{http://www.w3.org/2005/Atom}'
OPEN = '{http://a9.com/-/spec/opensearch/1.1/}'
START = '2026-07-01T00:00:00Z'
END = '2026-10-01T00:00:00Z'


def canonical_id(value):
    """Validate arXiv identity, then remove a version suffix (not other digits)."""
    value = value.strip()
    if '://' in value:
        u = urlsplit(value)
        if u.scheme not in ('http', 'https') or u.netloc not in ('arxiv.org', 'export.arxiv.org') or u.query or u.fragment:
            raise ValueError('Expected an arXiv identifier or official URL')
        if not u.path.startswith(('/abs/', '/pdf/')):
            raise ValueError('Expected abs/pdf path')
        value = u.path[5:]
    if value.endswith('.pdf'):
        value = value[:-4]
    pattern = r'(?P<base>(?:\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}|[a-z][a-z.-]+/\d{2}(?:0[1-9]|1[0-2])\d{3}))(?:v[1-9]\d*)?'
    m = re.fullmatch(pattern, value)
    if not m:
        raise ValueError('Malformed arXiv identity: ' + value)
    return m['base']


def coverage(pages):
    """A complete traversal has one stable total, contiguous offsets and unique IDs."""
    if not pages or any(p.get('error') for p in pages):
        return 'INCOMPLETE'
    try:
        total = pages[0]['total']
        if type(total) is not int or total < 0:
            return 'INCOMPLETE'
        expected, seen = 0, set()
        for p in sorted(pages, key=lambda p: p['start']):
            if p['total'] != total or p['start'] != expected or (not p['ids'] and total != 0):
                return 'INCOMPLETE'
            ids = [canonical_id(i) for i in p['ids']]
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                return 'INCOMPLETE'
            seen.update(ids)
            expected += len(ids)
        if total == 0 and len(pages) != 1:
            return 'INCOMPLETE'
        return 'COMPLETE' if expected == total else 'INCOMPLETE'
    except (KeyError, TypeError, ValueError):
        return 'INCOMPLETE'


def triage(record):
    """Admission to a reading log is not verification of a scientific result."""
    if not record.get('verified') or not record.get('reviewed') or not record.get('reason', '').strip():
        return 'DEFER'
    if not record.get('in_window'):
        return 'EXCLUDE'
    if record.get('evidence') not in ('reported', 'replayed', 'reproduced'):
        return 'DEFER'
    return 'INCLUDE' if record.get('relevance') in ('sota', 'failure_mode', 'baseline') else 'EXCLUDE'


def parse_atom(raw):
    root = ET.fromstring(raw)
    if root.tag != ATOM + 'feed':
        raise ValueError('Not an Atom feed')
    total, start = int(root.findtext(OPEN + 'totalResults')), int(root.findtext(OPEN + 'startIndex'))
    entries = []
    for e in root.findall(ATOM + 'entry'):
        uri = e.findtext(ATOM + 'id')
        identity = canonical_id(uri)
        published, updated = e.findtext(ATOM + 'published'), e.findtext(ATOM + 'updated')
        for date in (published, updated):
            if not date or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', date):
                raise ValueError('Unrecognized UTC timestamp')
        entries.append(dict(id=identity, version_uri=uri, title=' '.join(e.findtext(ATOM+'title', '').split()),
                            published=published, updated=updated, summary=' '.join(e.findtext(ATOM+'summary', '').split())))
    return dict(start=start, total=total, ids=[e['id'] for e in entries], entries=entries)


def replay(packet, identity_fn=canonical_id, coverage_fn=coverage, triage_fn=triage):
    """Authenticate frozen bytes and regenerate counts using all three learner functions."""
    packet = Path(packet)
    manifest = json.loads((packet/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        path = (packet/name).resolve()
        if not path.is_relative_to(packet.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Packet hash/path mismatch: ' + name)
    config = json.loads((packet/'config.json').read_text())
    receipt = json.loads((packet/'collection.json').read_text())
    by_id, query_results = {}, {}
    for query in config['queries']:
        pages = []
        for attempt in receipt['attempts']:
            if attempt['kind'] != query['name'] or not attempt.get('accepted'):
                continue
            page = parse_atom((packet/attempt['file']).read_bytes())
            if page['start'] != attempt['start']:
                raise ValueError('Response/request offset mismatch')
            pages.append(page)
            for e in page['entries']:
                k = identity_fn(e['version_uri'])
                row = by_id.setdefault(k, dict(id=k, title=e['title'], published=e['published'], versions=[], queries=[], in_window=START <= e['published'] < END))
                if e['version_uri'] not in row['versions']:row['versions'].append(e['version_uri'])
                if query['name'] not in row['queries']:row['queries'].append(query['name'])
        status = coverage_fn(pages)
        if any(not (START <= e['published'] < END) for p in pages for e in p['entries']):status = 'INCOMPLETE'
        query_results[query['name']] = dict(status=status, pages=len(pages), received=sum(len(p['ids']) for p in pages), total=pages[0]['total'] if pages else None)
    screening = json.loads((packet/'screening.json').read_text())
    records = []
    for k,row in sorted(by_id.items()):
        review = screening.get(k, dict(verified=True, reviewed=False, reason='Awaiting primary-source screening', relevance='unknown', evidence='reported'))
        record = dict(row, **review)
        record['decision'] = triage_fn(record)
        records.append(record)
    return dict(experiment=config['experiment'], window=[START, END],
                collection_status='COMPLETE' if all(q['status']=='COMPLETE' for q in query_results.values()) else 'INCOMPLETE',
                replay_status='COMPLETE', queries=query_results, unique_papers=len(records), papers=records,
                http_attempts=len(receipt['attempts']), failed_attempts=sum(not a.get('ok',False) for a in receipt['attempts']),
                paper_result_reproduction='NOT_RUN', historical_leaderboard='NOT_ESTABLISHED', learner='PENDING_WRITTEN_DEFENSE')


def review_log(report, reviews, triage_fn=triage):
    """Apply separately versioned author screening without altering discovery bytes."""
    known = {r['id'] for r in report['papers']}
    if set(reviews) - known:
        raise ValueError('Review refers to a paper outside this query packet')
    rows = []
    for row in report['papers']:
        annotation = reviews.get(row['id'], dict(reviewed=False, reason='Deferred: full abstract/method screening remains pending.', relevance='unknown', evidence='reported'))
        item = dict(row, **annotation)
        item['decision'] = triage_fn(item)
        rows.append(item)
    return rows
