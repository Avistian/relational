"""Standalone public page contract; Node exercises scripts, not browser proof."""
import json
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.current = [dict(attrs), ""]

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if tag == "script" and self.current is not None:
            self.scripts.append(self.current)
            self.current = None


class ExternalAssetContract(unittest.TestCase):
    def test_small_home_loads_external_index_and_renders_known_lesson(self):
        page = (ROOT / "index.html").read_bytes()
        self.assertLess(len(page), 20000, "home page must stay below 20000 bytes")
        text = page.decode("utf-8")
        self.assertIn('<a href="progress.html">Course progress</a>', text)
        parser = Scripts()
        parser.feed(text)
        sources = [attrs.get("src") for attrs, _ in parser.scripts]
        self.assertIn("assets/search-index.js", sources)
        self.assertLess(sources.index("assets/search-index.js"), sources.index("assets/search.js"))
        self.assertNotIn('id="lesson-search-data"', text)
        self.assertNotIn('"headings":', text, "no full inline index fallback")
        # Execute the page's actual search scripts and input handler with a tiny
        # DOM adapter. This is deliberately separate from real browser coverage.
        code = r'''
const vm = require('vm');
function element() { return {children: [], value: '', textContent: '',
  appendChild(child) { this.children.push(child); },
  set innerHTML(value) { this.children = []; },
  addEventListener(name, fn) { this[name] = fn; }}; }
const box = element(), out = element();
const context = {document: {
  getElementById(id) { return id === 'lesson-search' ? box : out; },
  createElement: element
}, HomeNav: {mount() {}}};
vm.createContext(context);
for (const script of SCRIPTS) vm.runInContext(script, context);
box.value = 'single-table assumption';
box.input();
if (!out.children.some(li => li.children.some(a =>
    a.href === 'lessons/0001-single-table-assumption.html' && a.textContent.includes('Single-Table'))))
  throw Error('known lesson must render as a result link');
'''
        scripts = []
        for attrs, body in parser.scripts:
            src = attrs.get("src")
            if src == "assets/home.js":
                continue  # unrelated navigation has its own tests
            if attrs.get("type") == "application/json":
                continue
            scripts.append((ROOT / src).read_text() if src else body)
        result = subprocess.run(["node", "-e", "const SCRIPTS = " + json.dumps(scripts) + ";\n" + code],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
