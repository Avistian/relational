"""Check every native aggregation-slider value against independent rank counts."""
import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright
from _independent_l060 import ranks

ROOT = Path(__file__).resolve().parents[1]
records = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    for width in (900, 375):
        page = browser.new_page(viewport={'width': width, 'height': 1100})
        page.goto((ROOT / 'lessons/0060-broad-model-comparison.html').as_uri())
        widget = page.locator('#aggregation-viz')
        slider = widget.locator('input[type=range]')
        for tick in range(21):
            value = tick / 20
            slider.evaluate('(e,v)=>{e.value=String(v);e.dispatchEvent(new Event("input",{bubbles:true}));}', value)
            rows = [widget.locator('tbody tr').nth(i).locator('td').all_text_contents() for i in range(2)]
            means = [value / 2, .4]
            mean_rank = ranks(means)
            rank0, rank1 = ranks([0, .4]), ranks([value, .4])
            for i, row in enumerate(rows):
                assert row[0] == ['A', 'B'][i]
                assert math.isclose(float(row[3]), means[i], abs_tol=.005000001)
                assert float(row[4]) == mean_rank[i]
                assert float(row[5]) == (rank0[i] + rank1[i]) / 2
            records.append({'width': width, 'seed1_loss': value,
                            'rank_of_mean': mean_rank,
                            'mean_seed_rank': [(rank0[i] + rank1[i]) / 2 for i in range(2)]})
        slider.evaluate('(e)=>{e.value="0";e.dispatchEvent(new Event("input",{bubbles:true}));}')
        slider.focus()
        page.keyboard.press('ArrowRight')
        assert slider.input_value() == '0.05'
        assert '0.05' in widget.locator('output').inner_text()
        widget.locator('button').click()
        assert slider.input_value() == '1'
        region = widget.get_by_role('region', name='Scrollable aggregation table')
        assert region.get_attribute('tabindex') == '0'
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        if width == 375:
            region.focus()
            page.keyboard.press('End')
            region.evaluate('(e)=>e.scrollLeft=e.scrollWidth')
            assert region.evaluate('(e)=>e.scrollLeft') > 0
        page.close()
    browser.close()
report = {'status': 'PASS', 'states': records,
          'scope': 'All 21 slider values at 900/375 pixels, both tie boundaries, mean/rank arithmetic, keyboard increment and reset, accessible horizontal region; synthetic teaching fixture'}
(ROOT / 'reviews/lesson-quality-audit-047-070/060-widget.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'states': len(records)}))
