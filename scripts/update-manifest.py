#!/usr/bin/env python3
"""Regenerate lessons/manifest.json entries from lesson HTML files (published only)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"
MANIFEST = LESSONS / "manifest.json"

TITLE_RE = re.compile(r"<title>Lesson \d+ — (.+?)</title>")
H1_RE = re.compile(r"<h1>(?:Q\d Checkpoint: )?(.+?)</h1>")


def slug_meta(slug: str) -> dict:
    n = int(slug[:4])
    year = (n - 1) // 40 + 1
    pos_in_year = (n - 1) % 40
    quarter = pos_in_year // 10 + 1
    checkpoint = "checkpoint" in slug
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


def main() -> None:
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
        lessons.append(meta)

    MANIFEST.write_text(
        json.dumps({"version": 1, "lessons": lessons}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(lessons)} entries to {MANIFEST}")


if __name__ == "__main__":
    main()
