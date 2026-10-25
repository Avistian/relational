"""F1 lesson-search tests. Expected values are derived independently from the
lesson/glossary HTML (own regexes), not from scripts/search_index.py."""
import html
import json
import re
import shutil
import subprocess
import tempfile
import sys
import unicodedata
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "search_index.py"
INDEX = ROOT / "assets" / "search-index.js"
SEARCH_JS = ROOT / "assets" / "search.js"
PY = sys.executable


def load_index():
    # Load the public classic-script export independently of the generator.
    js = "const vm=require('vm'),fs=require('fs'),c={};vm.createContext(c);" + \
         "vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),c);" + \
         "process.stdout.write(JSON.stringify(c.LessonSearchIndex));"
    result = subprocess.run(["node", "-e", js, str(INDEX)], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def fold(s):
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def lesson_files():
    return sorted(p.name for p in (ROOT / "lessons").glob("*.html"))


def raw_title(name):
    text = (ROOT / "lessons" / name).read_text(encoding="utf-8")
    m = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
    return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip())


class IndexCoverage(unittest.TestCase):
    def test_every_lesson_indexed_exactly_once(self):
        idx = load_index()
        urls = [e["url"] for e in idx]
        self.assertEqual(sorted(urls), sorted("lessons/" + n for n in lesson_files()))
        self.assertEqual(len(urls), len(set(urls)))
        self.assertGreater(len(urls), 100)

    def test_entries_have_shape_and_nonempty_title(self):
        for e in load_index():
            self.assertTrue(e["title"].strip(), e["url"])
            self.assertIsInstance(e["headings"], list)
            self.assertIsInstance(e["glossary"], list)

    def test_titles_match_html_title(self):
        by_url = {e["url"]: e for e in load_index()}
        for n in lesson_files():
            self.assertEqual(by_url["lessons/" + n]["title"], raw_title(n))

    def test_headings_match_independent_extraction(self):
        by_url = {e["url"]: e for e in load_index()}
        n = "0001-single-table-assumption.html"
        text = (ROOT / "lessons" / n).read_text(encoding="utf-8")
        expect = [html.unescape(re.sub(r"<[^>]+>", "", h)).strip()
                  for h in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", text, re.S)]
        expect = [re.sub(r"\s+", " ", h).rstrip("¶").strip() for h in expect]
        self.assertEqual(by_url["lessons/" + n]["headings"], [h for h in expect if h])
        self.assertIn("Worked example: e-commerce churn", expect)

    def test_glossary_terms_attached_to_linked_lesson(self):
        # glossary row: "Design matrix (X)" -> lesson cell "002"
        gl = (ROOT / "reference" / "glossary.html").read_text(encoding="utf-8")
        self.assertIn('<td class="term">Design matrix (X)</td>', gl)
        e = next(x for x in load_index() if x["url"].startswith("lessons/0002-"))
        self.assertIn("Design matrix (X)", e["glossary"])
        e48 = next(x for x in load_index() if x["url"].startswith("lessons/0048-"))
        self.assertIn("Explicit feature cross", e48["glossary"])


class Determinism(unittest.TestCase):
    def test_check_passes_on_committed_index(self):
        r = subprocess.run([PY, str(SCRIPT), "--check"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_regeneration_byte_identical(self):
        before = INDEX.read_bytes()
        out = Path(tempfile.mkdtemp()) / "regen.js"
        for _ in range(2):
            r = subprocess.run([PY, str(SCRIPT), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(out.read_bytes(), before)

    def test_check_fails_when_stale(self):
        stale = Path(tempfile.mkdtemp()) / "stale.js"
        stale.write_text("[]\n", encoding="utf-8")
        r = subprocess.run([PY, str(SCRIPT), "--check", "--out", str(stale)], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


@unittest.skipUnless(shutil.which("node"), "node not on PATH")
class SearchJs(unittest.TestCase):
    def run_search(self, query, index=None):
        js = ("const {search}=require(%s);const idx=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "process.stdout.write(JSON.stringify(search(idx,%s)));"
              % (json.dumps(str(SEARCH_JS)), json.dumps(query)))
        payload = json.dumps(index if index is not None else load_index())
        r = subprocess.run(["node", "-e", js], input=payload, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_known_title_word_top_result(self):
        # title of lesson 0085 (read from the file) contains "Over-smoothing"-style word
        title = raw_title("0085-over-smoothing.html")
        word = next(w for w in re.findall(r"[A-Za-z]{6,}", title) if w.lower() not in ("lesson",))
        res = self.run_search(word)
        self.assertIn("lessons/0085-over-smoothing.html", [r["url"] for r in res])

    def test_exact_title_query_ranks_that_lesson_first(self):
        title = raw_title("0001-single-table-assumption.html")
        res = self.run_search(title)
        self.assertEqual(res[0]["url"], "lessons/0001-single-table-assumption.html")

    def test_h2_query_found(self):
        res = self.run_search("Worked example: e-commerce churn")
        self.assertIn("lessons/0001-single-table-assumption.html", [r["url"] for r in res])

    def test_glossary_term_found(self):
        res = self.run_search("explicit feature cross")
        self.assertIn("lessons/0048-dcnv2.html", [r["url"] for r in res])

    def test_case_and_diacritic_insensitive(self):
        idx = [{"url": "a.html", "title": "Café Résumé", "headings": [], "glossary": []}]
        for q in ("cafe resume", "CAFÉ", "résumé"):
            self.assertEqual([r["url"] for r in self.run_search(q, idx)], ["a.html"])

    def test_multiword_is_and(self):
        idx = [{"url": "a", "title": "alpha beta", "headings": [], "glossary": []},
               {"url": "b", "title": "alpha", "headings": [], "glossary": []}]
        self.assertEqual([r["url"] for r in self.run_search("alpha beta", idx)], ["a"])

    def test_ranking_title_over_heading_over_glossary(self):
        idx = [{"url": "g", "title": "zzz", "headings": [], "glossary": ["quux"]},
               {"url": "h", "title": "zzz", "headings": ["quux"], "glossary": []},
               {"url": "t", "title": "quux", "headings": [], "glossary": []}]
        self.assertEqual([r["url"] for r in self.run_search("quux", idx)], ["t", "h", "g"])

    def test_empty_and_nomatch_return_empty(self):
        self.assertEqual(self.run_search(""), [])
        self.assertEqual(self.run_search("   "), [])
        self.assertEqual(self.run_search("qqzzxxnomatchtoken"), [])


class IndexHtmlWiring(unittest.TestCase):
    def test_search_box_and_script_present_and_content_kept(self):
        text = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="lesson-search"', text)
        self.assertIn('assets/search.js', text)
        for keep in ('id="lesson-nav"', 'id="continue-nav"', "assets/home.js", "<h2>Reference</h2>"):
            self.assertIn(keep, text)
        self.assertEqual(text.count('id="lesson-search"'), 1)

    def test_external_classic_script_loads_before_search(self):
        text = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertLess(text.index('src="assets/search-index.js"'), text.index('src="assets/search.js"'))
        self.assertNotIn('id="lesson-search-data"', text)
        self.assertNotIn("fetch(", text)

    def test_check_fails_when_script_wiring_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "index.html"
            page.write_text((ROOT / "index.html").read_text(encoding="utf-8").replace(
                'src="assets/search-index.js"', 'src="assets/missing.js"'), encoding="utf-8")
            r = subprocess.run([PY, str(SCRIPT), "--check", "--html", str(page)], cwd=ROOT,
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("STALE: search script wiring", r.stderr)

    def test_check_fails_when_generated_asset_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.js"
            r = subprocess.run([PY, str(SCRIPT), "--check", "--out", str(missing)],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("STALE:", r.stderr)
            self.assertFalse(missing.exists())

    def test_legacy_json_export_matches_script_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "index.json"
            r = subprocess.run([PY, str(SCRIPT), "--out", str(out)],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(out.read_text()), load_index())


if __name__ == "__main__":
    unittest.main()
