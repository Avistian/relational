"""Static navigation contract; rendered visibility is verified in the browser."""

from html.parser import HTMLParser
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class NavigationParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        style = "".join(attrs.get("style", "").lower().split())
        hidden = ("hidden" in attrs or attrs.get("aria-hidden", "").lower() == "true"
                  or "display:none" in style or "visibility:hidden" in style
                  or tag in {"template", "script", "style"})
        node = {"tag": tag, "attrs": attrs, "hidden": hidden, "text": []}
        if tag == "a":
            node["in_navigation"] = any(n["tag"] == "nav" for n in self.stack)
            node["hidden"] |= any(n["hidden"] for n in self.stack)
            self.links.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        for node in self.stack:
            if node["tag"] == "a":
                node["text"].append(data)


class CourseProgressNavigation(unittest.TestCase):
    def test_index_navigation_links_to_existing_course_progress_page(self):
        page = NavigationParser()
        page.feed((ROOT / "index.html").read_text(encoding="utf-8"))
        links = [link for link in page.links
                 if link["attrs"].get("href") == "progress.html"
                 and " ".join("".join(link["text"]).split()) == "Course progress"
                 and link["in_navigation"] and not link["hidden"]]
        self.assertTrue(links, "Index navigation must expose a visible Course progress link to progress.html")
        self.assertTrue((ROOT / links[0]["attrs"]["href"]).is_file(),
                        "Course progress link must resolve to an existing local page")
