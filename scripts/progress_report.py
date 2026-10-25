#!/usr/bin/env python3
"""Build assets/progress-data.json and progress.html from learning-records/ (F2).

Definitions (deterministic; no stored review history exists):
- lesson: every top-level lessons/*.html file; id = filename prefix before the first "-".
- done: the lesson has a `lesson-NNN-complete` learning record (duplicates count once).
- completion-reported: a `lesson-NNN-completion-reported` record and no complete record.
  Counted separately, labelled, and kept in "pending" (no mastery claim).
- pending: every other lesson (published/prepared/start records, or no record at all).
- flashcards due: a card in assets/paper-deck.js is eligible once its integer lesson has a
  complete record; with no review history every eligible card is Leitner box 1 (interval 0
  days), hence due on the reference date. Cards of other lessons are "not yet eligible".
"""
import argparse
import datetime
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_RE = re.compile(r"^\d+-lesson-(\d+)-(complete|completion-reported|published|prepared|start)\.md$")
KEY_RE = {k: re.compile(r"""(?:^|[{,\s])["']?%s["']?\s*:\s*(?:["']([^"']*)["']|([0-9.]+))""" % k)
          for k in ("id", "lesson")}


def top_level_objects(text):
    """Yield the text of each {...} directly inside the PAPER_DECK array (string-aware)."""
    start = list(re.finditer(r"PAPER_DECK\s*=\s*\[", text))[-1].end() - 1  # last: header comment repeats it
    depth, quote, begin = 0, None, None
    i = start + 1
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield text[begin:i + 1]
        elif ch == "]" and depth == 0:
            return
        i += 1


def parse_deck(text):
    """Return [(id, lesson float)] for cards written in any of the deck's object styles."""
    cards = []
    for obj in top_level_objects(text):
        cid, les = KEY_RE["id"].search(obj), KEY_RE["lesson"].search(obj)
        if cid and les:
            cards.append((cid.group(1), float(les.group(2) or les.group(1))))
    return cards


def dumps(obj):
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def scan_records(root):
    """Return {int lesson number: set(kinds)} from learning-records/ filenames."""
    kinds = {}
    for p in sorted((Path(root) / "learning-records").glob("*.md")):
        m = RECORD_RE.match(p.name)
        if m:
            kinds.setdefault(int(m.group(1)), set()).add(m.group(2))
    return kinds


def build_report(root, today):
    root = Path(root)
    datetime.date.fromisoformat(today)
    kinds = scan_records(root)
    lesson_files = sorted(p.name for p in (root / "lessons").glob("*.html"))
    rows = []
    for name in lesson_files:
        lid = name.split("-", 1)[0]
        ks = kinds.get(int(lid), set()) if lid.isdigit() else set()
        done = "complete" in ks
        rows.append({
            "id": lid,
            "file": name,
            "state": "done" if done else "pending",
            "completion_reported": ("completion-reported" in ks) and not done,
            "records": sorted(ks),
        })
    ids = {r["id"] for r in rows}
    done_nums = {n for n, ks in kinds.items() if "complete" in ks and f"{n:04d}" in ids}
    orphans = sorted(n for n, ks in kinds.items() if "complete" in ks and f"{n:04d}" not in ids)

    deck = (root / "assets" / "paper-deck.js").read_text(encoding="utf-8")
    cards = parse_deck(deck)
    due = sorted(cid for cid, les in cards if les == int(les) and int(les) in done_nums)
    n_done = sum(r["state"] == "done" for r in rows)
    return {
        "today": today,
        "counts": {
            "lessons_total": len(rows),
            "done": n_done,
            "pending": len(rows) - n_done,
            "completion_reported": sum(r["completion_reported"] for r in rows),
            "orphan_complete_records": len(orphans),
        },
        "flashcards": {
            "total": len(cards),
            "due": len(due),
            "due_ids": due,
            "not_yet_eligible": len(cards) - len(due),
            "assumption": ("No review history is stored (state lives in browser localStorage). "
                           "A card is eligible once its lesson has a complete record; every "
                           "eligible card is treated as Leitner box 1 (interval 0 days), i.e. due."),
        },
        "lessons": rows,
    }


def render_html(report):
    c, f = report["counts"], report["flashcards"]
    e = html.escape
    trs = []
    for r in report["lessons"]:
        label = r["state"] + (" (completion reported, unverified)" if r["completion_reported"] else "")
        trs.append(f'<tr data-state="{r["state"]}"><td>{e(r["id"])}</td><td>{e(label)}</td>'
                   f'<td>{e(", ".join(r["records"]) or "none")}</td></tr>')
    rows = "\n".join(trs)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Progress</title>
  <link rel="stylesheet" href="assets/lesson.css">
</head>
<body>
<article>
  <p class="mission-tag">Progress · generated from learning-records/</p>
  <h1>Progress</h1>
  <p class="subtitle">Reference date {e(report["today"])} · generated by scripts/progress_report.py from assets/progress-data.json</p>
  <ul>
    <li>Lessons done: <strong data-stat="done">{c["done"]}</strong> of {c["lessons_total"]}</li>
    <li>Lessons pending: <strong data-stat="pending">{c["pending"]}</strong>
      (including <span data-stat="completion-reported">{c["completion_reported"]}</span> completion-reported, not counted as done)</li>
    <li>Flashcards due: <strong data-stat="flashcards-due">{f["due"]}</strong> of {f["total"]}
      ({f["not_yet_eligible"]} not yet eligible)</li>
  </ul>
  <p>{e(f["assumption"])}</p>
  <table class="compact-trace">
    <thead><tr><th>Lesson</th><th>State</th><th>Records</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <nav class="nav-links"><ul><li><a href="index.html">Home</a></li></ul></nav>
</article>
</body>
</html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--today", default=datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json-out")
    ap.add_argument("--html-out")
    ap.add_argument("--check", action="store_true",
                    help="check both generated outputs without writing (nonzero if missing or stale)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    rep = build_report(root, a.today)
    outputs = [(Path(a.json_out or root / "assets" / "progress-data.json"), dumps(rep)),
               (Path(a.html_out or root / "progress.html"), render_html(rep))]
    if a.check:
        stale = False
        for path, expected in outputs:
            try:
                current = path.read_bytes()
            except OSError as exc:
                print(f"Missing or unreadable progress output: {path}: {exc}", file=sys.stderr)
                stale = True
                continue
            if current != expected.encode("utf-8"):
                print(f"Stale progress output for {a.today}: {path}", file=sys.stderr)
                stale = True
        return 1 if stale else 0
    for path, content in outputs:
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
