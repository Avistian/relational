"""Tests for scripts/progress_report.py (F2). Stdlib unittest only."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import progress_report as pr  # noqa: E402

TODAY = "2026-10-18"


def recount_records(root):
    """Independent recount: plain os.listdir + substring checks, no regex from the module."""
    counts = {"complete": set(), "completion-reported": set()}
    for name in os.listdir(os.path.join(root, "learning-records")):
        if not name.endswith(".md") or "lesson-" not in name:
            continue
        parts = name[:-3].split("-")
        # NNNN-lesson-LLL-<kind...>
        if len(parts) < 4 or parts[1] != "lesson" or not parts[2].isdigit():
            continue
        kind = "-".join(parts[3:])
        if kind in counts:
            counts[kind].add(int(parts[2]))
    return counts


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class SyntheticFixture(unittest.TestCase):
    """SYNTHETIC temp-dir repo covering odd/duplicate record names."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for lid in ("0001", "0002", "0003", "0004", "0005", "b01"):
            write(self.root / "lessons" / f"{lid}-x.html", "<html></html>")
        write(self.root / "lessons" / "manifest.json", "{}")
        rec = self.root / "learning-records"
        write(rec / "0001-lesson-001-complete.md")
        write(rec / "0002-lesson-001-complete.md")  # duplicate complete
        write(rec / "0003-lesson-002-start.md")
        write(rec / "0004-lesson-002-complete.md")
        write(rec / "0005-lesson-003-completion-reported.md")
        write(rec / "0006-lesson-004-published.md")
        write(rec / "0007-lesson-004-prepared.md")
        write(rec / "0008-mission-established.md")  # not a lesson record
        write(rec / "0009-lesson-abc-complete.md")  # odd name, ignored
        write(rec / "0010-lesson-099-complete.md")  # record without lesson file
        write(rec / "notes.txt")  # non-md ignored
        write(self.root / "assets" / "paper-deck.js",
              'global.PAPER_DECK = [\n'
              '{"id": "c1", "lesson": 1, "paper": "P"},\n'
              '{"id": "c2", "lesson": 2, "paper": "P"},\n'
              '{"id": "c3", "lesson": 3, "paper": "P"},\n'
              '{"id": "c4", "lesson": 200.06, "paper": "P"},\n'
              '{"id": "c5", "lesson": 1, "paper": "P"}\n'
              ',{id:\'c6\',lesson:2,paper:\'Q {brace} id: "zz"\'}\n'
              ',{\n  "id": "c7",\n  "paper": "P",\n  "lesson": 3\n}\n'
              ',{\n  id: "c8", paper: "P", year: 2024, lesson: 1,\n  back: "x"\n}\n];\n')
        self.report = pr.build_report(self.root, TODAY)

    def tearDown(self):
        self.tmp.cleanup()

    def status(self, lid):
        return next(r for r in self.report["lessons"] if r["id"] == lid)

    def test_done_counts_dedupe_duplicates(self):
        self.assertEqual(self.report["counts"]["done"], 2)  # 001, 002
        self.assertEqual(self.status("0001")["state"], "done")

    def test_completion_reported_counted_separately_and_pending(self):
        self.assertEqual(self.report["counts"]["completion_reported"], 1)
        row = self.status("0003")
        self.assertEqual(row["state"], "pending")
        self.assertTrue(row["completion_reported"])

    def test_every_lesson_exactly_one_state(self):
        ids = [r["id"] for r in self.report["lessons"]]
        self.assertEqual(sorted(ids), ["0001", "0002", "0003", "0004", "0005", "b01"])
        self.assertTrue(all(r["state"] in ("done", "pending") for r in self.report["lessons"]))
        c = self.report["counts"]
        self.assertEqual(c["done"] + c["pending"], c["lessons_total"])

    def test_orphan_record_not_counted(self):
        self.assertEqual(self.report["counts"]["orphan_complete_records"], 1)
        self.assertEqual(self.report["counts"]["done"], 2)

    def test_flashcards_due_only_for_complete_lessons(self):
        self.assertEqual(self.report["flashcards"]["due_ids"], ["c1", "c2", "c5", "c6", "c8"])
        self.assertEqual(self.report["flashcards"]["due"], 5)
        self.assertEqual(self.report["flashcards"]["not_yet_eligible"], 3)  # c3, c4, c7
        self.assertEqual(self.report["flashcards"]["total"], 8)

    def test_today_recorded_and_deterministic(self):
        self.assertEqual(self.report["today"], TODAY)
        again = pr.build_report(self.root, TODAY)
        self.assertEqual(pr.dumps(again), pr.dumps(self.report))

    def test_html_contains_numbers(self):
        html = pr.render_html(self.report)
        c = self.report["counts"]
        for key in ("done", "pending"):
            self.assertRegex(html, rf'data-stat="{key}">{c[key]}<')
        self.assertRegex(html, rf'data-stat="flashcards-due">{self.report["flashcards"]["due"]}<')
        self.assertIn(TODAY, html)
        self.assertIn("assets/lesson.css", html)


class RealRepo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = pr.build_report(ROOT, TODAY)

    def test_counts_match_independent_recount(self):
        rc = recount_records(ROOT)
        lesson_ids = {int(n.split("-")[0]) for n in os.listdir(ROOT / "lessons")
                      if n.endswith(".html") and n.split("-")[0].isdigit()}
        done = rc["complete"] & lesson_ids
        self.assertEqual(self.report["counts"]["done"] - sum(
            1 for r in self.report["lessons"] if r["state"] == "done" and not r["id"].isdigit()), len(done))
        self.assertEqual(self.report["counts"]["completion_reported"], len(rc["completion-reported"] - rc["complete"]))

    def test_lessons_total_equals_filesystem(self):
        n = len([f for f in os.listdir(ROOT / "lessons") if f.endswith(".html")])
        self.assertEqual(self.report["counts"]["lessons_total"], n)

    def test_each_lesson_exactly_once(self):
        ids = [r["id"] for r in self.report["lessons"]]
        self.assertEqual(len(ids), len(set(ids)))
        c = self.report["counts"]
        self.assertEqual(c["done"] + c["pending"], len(ids))

    def test_due_equals_recomputed_eligible(self):
        if shutil.which("node") is None:
            self.skipTest("node not on PATH; independent deck evaluation unavailable")
        js = ("global.window = {}; (function(){ const fs=require('fs'); "
              "new Function('window', fs.readFileSync(process.argv[1],'utf8'))(global.window); })();"
              "console.log(JSON.stringify(window.PAPER_DECK.map(c=>[c.id,c.lesson])));")
        out = subprocess.run(["node", "-e", js, str(ROOT / "assets" / "paper-deck.js")],
                             check=True, capture_output=True, text=True).stdout
        cards = [(i, float(l)) for i, l in json.loads(out)]
        rc = recount_records(ROOT)
        done = {n for n in rc["complete"]}
        due = sorted(cid for cid, les in cards if les == int(les) and int(les) in done)
        self.assertEqual(self.report["flashcards"]["due_ids"], due)
        self.assertEqual(self.report["flashcards"]["due"], len(due))
        self.assertEqual(self.report["flashcards"]["total"], len(cards))

    def test_committed_json_and_html_match_fixed_today(self):
        data = json.loads((ROOT / "assets" / "progress-data.json").read_text(encoding="utf-8"))
        fresh = pr.build_report(ROOT, data["today"])
        self.assertEqual(data, fresh, "assets/progress-data.json is stale; rerun scripts/progress_report.py")
        html = (ROOT / "progress.html").read_text(encoding="utf-8")
        self.assertEqual(html, pr.render_html(data))
        self.assertRegex(html, rf'data-stat="done">{data["counts"]["done"]}<')

    def test_cli_deterministic_with_today(self):
        with tempfile.TemporaryDirectory() as d:
            outs = []
            for i in range(2):
                j, h = Path(d) / f"{i}.json", Path(d) / f"{i}.html"
                subprocess.run([sys.executable, str(ROOT / "scripts" / "progress_report.py"),
                                "--today", TODAY, "--json-out", str(j), "--html-out", str(h)],
                               check=True, capture_output=True)
                outs.append((j.read_bytes(), h.read_bytes()))
            self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
