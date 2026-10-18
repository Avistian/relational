#!/usr/bin/env python3
"""Regenerate assets/search-index.json: one entry per lessons/*.html.

Entry: {url, title, headings (h2/h3 text), glossary (terms linked/defined for
that lesson in reference/glossary.html, where determinable)}.
Deterministic: sorted by url, fixed JSON formatting, no timestamps.
`--check` exits 1 if the committed JSON (or the copy embedded in index.html,
which lets the search box work over file://) differs from a fresh build.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "search-index.json"
DEFAULT_HTML = ROOT / "index.html"
EMBED_RE = re.compile(r'(<script type="application/json" id="lesson-search-data">)(.*?)(</script>)', re.S)


def embed(page_text, index_json):
    """Return page_text with the compact index embedded (works over file://)."""
    compact = json.dumps(json.loads(index_json), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    if not EMBED_RE.search(page_text):
        raise SystemExit("index.html has no lesson-search-data script block")
    return EMBED_RE.sub(lambda m: m.group(1) + compact + m.group(3), page_text, count=1)


def clean(fragment):
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment))
    text = re.sub(r"\s+", " ", text).replace("¶", "")
    return text.strip()


def lesson_entry(path):
    text = path.read_text(encoding="utf-8")
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
    title = clean(m.group(1)) if m else ""
    headings = [h for h in (clean(x) for x in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", text, re.S | re.I)) if h]
    return {"url": "lessons/" + path.name, "title": title, "headings": headings, "glossary": []}


def glossary_terms(root, names):
    """Map lesson filename -> ordered unique glossary terms."""
    src = (root / "reference" / "glossary.html").read_text(encoding="utf-8")
    by_number = {}
    for n in names:
        mm = re.match(r"(\d{4})-", n)
        if mm:
            by_number.setdefault(mm.group(1), n)
    out = {}
    for row in re.findall(r"<tr>(.*?)</tr>", src, re.S):
        t = re.search(r'<td class="term">(.*?)</td>', row, re.S)
        c = re.search(r'<td class="lsn">(.*?)</td>', row, re.S)
        if not t or not c:
            continue
        term, cell = clean(t.group(1)), c.group(1)
        targets = []
        for href in re.findall(r'href="([^"#]*)', cell):
            name = href.rsplit("/", 1)[-1]
            if name in names:
                targets.append(name)
        if not targets:
            for num in re.findall(r"(?<!\d)(\d{3})(?!\d)", clean(cell)):
                n = by_number.get("0" + num)
                if n:
                    targets.append(n)
        for n in targets:
            terms = out.setdefault(n, [])
            if term not in terms:
                terms.append(term)
    return out


def build(root=ROOT):
    files = sorted((root / "lessons").glob("*.html"), key=lambda p: p.name)
    names = {p.name for p in files}
    gloss = glossary_terms(root, names)
    entries = []
    for p in files:
        e = lesson_entry(p)
        e["glossary"] = gloss.get(p.name, [])
        entries.append(e)
    return json.dumps(entries, ensure_ascii=False, indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if the JSON file is stale")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--html", type=Path, default=None,
                    help="page holding the embedded copy (default index.html when --out is the default)")
    args = ap.parse_args(argv)
    if args.html is None and args.out == DEFAULT_OUT:
        args.html = DEFAULT_HTML
    fresh = build()
    page = None
    if args.html is not None:
        page = embed(args.html.read_text(encoding="utf-8"), fresh)
    if args.check:
        have = args.out.read_bytes() if args.out.exists() else None
        if have != fresh.encode("utf-8"):
            print("STALE: %s differs from a fresh build; run scripts/search_index.py" % args.out, file=sys.stderr)
            return 1
        if page is not None and args.html.read_text(encoding="utf-8") != page:
            print("STALE: embedded index in %s; run scripts/search_index.py" % args.html, file=sys.stderr)
            return 1
        print("search index up to date")
        return 0
    args.out.write_bytes(fresh.encode("utf-8"))
    if page is not None:
        args.html.write_text(page, encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
