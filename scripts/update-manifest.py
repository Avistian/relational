#!/usr/bin/env python3
"""Regenerate lessons/manifest.json entries from lesson HTML files (published only)."""
from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"
MANIFEST = LESSONS / "manifest.json"

TITLE_RE = re.compile(r"<title>Lesson (?:\d+|B\d+[a-z]?) — (.+?)</title>")
H1_RE = re.compile(r"<h1>(?:Q\d Checkpoint: )?(.+?)</h1>")


def slug_meta(slug: str) -> dict:
    if re.match(r"^b\d{2}(?:[a-z])?-", slug):
        identifier = "B" + slug.split("-", 1)[0][1:]
        previous = next((e for e in json.loads(MANIFEST.read_text())["lessons"] if e["id"] == identifier), {})
        base_order = 200 + int(identifier[1:3]) / 100
        suffix_order = (ord(identifier[3]) - 96) / 10000 if len(identifier) > 3 else 0
        return {"id": identifier, "slug": slug, "year": 5, "quarter": 5,
                "sortOrder": previous.get("sortOrder", base_order + suffix_order),
                "checkpoint": False, "labPath": "labs/" + slug + ".ipynb" if (ROOT / "labs" / (slug + ".ipynb")).exists() else None}
    n = int(slug[:4])
    year = (n - 1) // 40 + 1
    pos_in_year = (n - 1) % 40
    quarter = pos_in_year // 10 + 1
    checkpoint = n % 10 == 0 or "checkpoint" in slug
    lab = ROOT / "labs" / f"{slug}.ipynb"
    if not lab.exists():
        # e.g. 0006-missingness-taxonomy → 0006-missingness.ipynb
        alt = ROOT / "labs" / f"{slug[:4]}-*.ipynb"
        matches = sorted((ROOT / "labs").glob(f"{slug[:4]}-*.ipynb"))
        lab = matches[0] if matches else lab
    lab_path = f"labs/{lab.name}" if lab.exists() else None
    return {
        "id": n,
        "slug": slug,
        "year": year,
        "quarter": quarter,
        "checkpoint": checkpoint,
        "labPath": lab_path,
    }


def title_from_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = TITLE_RE.search(text) or H1_RE.search(text)
    if not m:
        return path.stem
    return m.group(1).replace("&amp;", "&").replace("&#39;", "'")


def build_manifest() -> dict:
    existing: dict[int, dict] = {}
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for entry in data.get("lessons", []):
            existing[entry["id"]] = entry

    lessons = []
    for html in sorted(LESSONS.glob("*.html")):
        meta = slug_meta(html.stem)
        meta["title"] = title_from_html(html)
        prev = existing.get(meta["id"], {})
        meta["published"] = prev.get("published", True)
        # Some checkpoints are not named "checkpoint" (e.g. the Year 1 exit exam), so the flag can only
        # be set by hand — never let a regeneration silently clear it.
        meta["checkpoint"] = meta["checkpoint"] or bool(prev.get("checkpoint", False))
        lessons.append(meta)

    return {"version": 1, "lessons": lessons}


def parsed_content_equal(current: object, expected: object) -> bool:
    """Compare JSON recursively, keeping booleans distinct from numbers."""
    if isinstance(current, bool) or isinstance(expected, bool):
        return type(current) is type(expected) and current == expected
    if isinstance(current, dict) and isinstance(expected, dict):
        return current.keys() == expected.keys() and all(
            parsed_content_equal(current[key], expected[key]) for key in current
        )
    if isinstance(current, list) and isinstance(expected, list):
        return len(current) == len(expected) and all(
            parsed_content_equal(left, right) for left, right in zip(current, expected)
        )
    return current == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="check parsed manifest content without writing files")
    args = parser.parse_args()
    if args.check:
        try:
            current = json.loads(MANIFEST.read_text(encoding="utf-8"))
            expected = build_manifest()
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            print(f"Manifest check failed: {MANIFEST}: {exc}", file=sys.stderr)
            return 1
        if not parsed_content_equal(current, expected):
            print(f"Manifest is stale: {MANIFEST}; run without --check to regenerate.",
                  file=sys.stderr)
            return 1
        print(f"Manifest is current: {MANIFEST}")
        return 0

    manifest = build_manifest()
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest['lessons'])} entries to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
