"""Exercise a repaired foundation lesson's static learner route in copied Pages."""
import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def check(n):
    report_dir = ROOT / 'reviews/lesson-quality-audit-047-070'
    stage = Path(json.loads((report_dir / f'{n:03}-pages.json').read_text())['stage'])
    lesson = next((stage / 'lessons').glob(f'{n:04}-*.html'))
    slug = lesson.stem

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(stage)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    shots = Path(f'/tmp/quality-l{n:03}-access')
    shots.mkdir(exist_ok=True)
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for width in (900, 375):
            context = browser.new_context(java_script_enabled=False,
                                          viewport={'width': width, 'height': 1000})
            page = context.new_page()
            page.goto(f'{base}/lessons/{slug}.html')
            launcher = page.locator('[data-lab-launch]').first
            preview = launcher.locator(f'a[href$="/html/{slug}.html"]')
            assert preview.bounding_box()['y'] < 750, 'Lab entry buried below reading'
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            preview.click()
            assert page.url == f'{base}/labs/html/{slug}.html'
            toolbar = page.locator('[data-lab-launch]').first
            assert 'read-only' in toolbar.inner_text().lower()
            colab = toolbar.locator('a[href*="colab.research.google.com"]')
            assert colab.is_visible() and colab.get_attribute('href').endswith(f'/labs/{slug}.ipynb')
            download = toolbar.locator(f'a[href$="/{slug}.ipynb"]:not([href*="colab"])')
            response = context.request.get(base + '/labs/' + slug + '.ipynb')
            assert response.ok and response.body() == (stage / 'labs' / f'{slug}.ipynb').read_bytes()
            assert download.is_visible()
            toolbar.locator('a[href="#lab-exercises"]').click()
            assert page.url.endswith('#lab-exercises')
            assert page.locator('#lab-exercises').bounding_box()['y'] < 220
            page.screenshot(path=str(shots / f'{width}-exercises.png'))
            page.goto(f'{base}/notebooks.html')
            page.locator('#foundation-materials a').click()
            card = page.locator(f'#lesson-{n}')
            assert card.locator('a[href*="reproduction.md"]').count() == 1
            assert card.locator('a[href*="results.json"]').count() >= 1
            card.locator(f'a[href="{slug}.html"]').click()
            assert page.url == f'{base}/labs/html/{slug}.html'
            records.append(dict(width=width, javascript=False, lesson_preview_exercises='PASS',
                                gallery_directory_preview='PASS', notebook_bytes='MATCH'))
            context.close()
        page = browser.new_page()
        page.route('**/lessons/manifest.json*', lambda route: route.abort())
        page.goto(f'{base}/notebooks.html', wait_until='networkidle')
        page.locator(f'#lab-{n} a[href="labs/html/{slug}.html"]').click()
        assert page.url == f'{base}/labs/html/{slug}.html'
        browser.close()
    server.shutdown()
    result = dict(status='PASS', lesson=n, records=records, failed_manifest_gallery='PASS',
                  screenshots=str(shots), scope='Actual no-JavaScript and failed-fetch routes in copied Pages; Colab link inspected, live Colab not run')
    (report_dir / f'{n:03}-access.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lesson', type=int)
    check(parser.parse_args().lesson)
