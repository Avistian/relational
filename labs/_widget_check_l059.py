"""Actual DOM arithmetic and held-label intervention checks, L059 only."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':375,'height':1200})
    page.goto((ROOT/'lessons/0059-validation-set-overfitting.html').as_uri())
    root=page.locator('#nested-boundary')
    expected={(1,False):('A','B'),(1,True):('A','A'),(0,False):('B','B'),(0,True):('B','B')}
    for (fold,flip),(internal,external) in expected.items():
        root.locator('select').select_option(str(fold));root.locator('input').set_checked(flip)
        assert root.get_attribute('data-internal')==internal
        assert root.get_attribute('data-external')==external
        assert root.locator('tbody tr').count()==8
        assert root.locator('td').all_text_contents().count('No · held')==4
        records.append({'held_fold':fold,'flip':flip,'internal':internal,'external':external,'readout':root.locator('output').inner_text()})
    root.locator('button').click()
    assert root.locator('output').inner_text()=='Internal: A from [0.100, 0.725]. External: B from [0.413, 0.375].'
    assert root.locator('tbody tr').nth(0).locator('td').nth(2).inner_text()=='0.04'
    # Keyboard-only held-label intervention must preserve internal predictions.
    before=root.locator('.detail').inner_text().split('].')[0]
    root.locator('input').focus();page.keyboard.press('Space')
    after=root.locator('.detail').inner_text().split('].')[0]
    assert before==after and root.get_attribute('data-external')=='A'
    browser.close()
report={'status':'PASS','states':records,'scope':'Four actual-browser selection states, row eligibility, exact default means, reset and keyboard held-label intervention'}
(ROOT/'reviews/lesson-quality-audit-047-070/059-widget.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
