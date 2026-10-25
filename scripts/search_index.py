#!/usr/bin/env python3
"""Regenerate the classic-script assets/search-index.js for file:// and HTTP.

Entries contain url, title, headings and glossary terms. Output is deterministic.
`--check` exits 1 for a missing/stale asset or invalid home-page wiring.
Explicit --out *.json retains the legacy JSON export interface.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "search-index.js"
DEFAULT_HTML = ROOT / "index.html"


def script_asset(index_json):
    """A classic script needs neither fetch nor a module server for file://."""
    compact = json.dumps(json.loads(index_json), ensure_ascii=False, separators=(",", ":"))
    return "var LessonSearchIndex = " + compact.replace("</", "<\\/") + ";\n"


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
    ap.add_argument("--check", action="store_true", help="exit 1 if the generated asset is missing or stale")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--html", type=Path, default=None,
                    help="page loading the asset (default index.html when --out is the default)")
    args = ap.parse_args(argv)
    if args.html is None and args.out == DEFAULT_OUT:
        args.html = DEFAULT_HTML
    fresh = build()
    if args.out.suffix == ".js":
        fresh = script_asset(fresh)
    if args.html is not None:
        page = args.html.read_text(encoding="utf-8")
        data_tag = '<script src="assets/search-index.js"></script>'
        search_tag = '<script src="assets/search.js"></script>'
        if (data_tag not in page or search_tag not in page
                or page.index(data_tag) > page.index(search_tag)
                or 'id="lesson-search-data"' in page):
            print("STALE: search script wiring in %s" % args.html, file=sys.stderr)
            return 1
    if args.check:
        have = args.out.read_bytes() if args.out.exists() else None
        if have != fresh.encode("utf-8"):
            print("STALE: %s differs from a fresh build; run scripts/search_index.py" % args.out, file=sys.stderr)
            return 1
        print("search index up to date")
        return 0
    args.out.write_bytes(fresh.encode("utf-8"))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
