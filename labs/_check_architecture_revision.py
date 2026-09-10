"""Export and inspect every architecture panel in actual Chromium rendering.

Use --export before notebook regeneration, then run the complete delivery/browser
checks on the integrated pages. Screenshots remain in /tmp; portable PNGs ship.
"""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
from _architecture_revision import ROOT,PANELS,preview


def check(export=False):
    preview();records=[];shots=Path('/tmp/relational-architecture-review');shots.mkdir(exist_ok=True)
    out=ROOT/'labs/figures/architecture-revision';out.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':800,'height':1000},device_scale_factor=2)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        for n,panels in PANELS.items():
            for panel in panels:
                name=f'{n:04}-{panel["key"]}'
                for width in [800,375]:
                    page.set_viewport_size({'width':width,'height':1000})
                    page.goto((ROOT/'labs/html/architecture-review'/f'{name}.html').as_uri())
                    problems=page.locator('.arch-atlas').evaluate('''root=>{
                      let out=[],r=root.getBoundingClientRect();
                      for(let el of root.querySelectorAll('h3,h4,p,li,td,th,code,.aa-op,.aa-equation')){
                        let b=el.getBoundingClientRect();if(b.left<r.left-1||b.right>r.right+1)out.push('outside '+el.tagName+' '+el.textContent.slice(0,50));
                        if(el.scrollWidth>el.clientWidth+2)out.push('overflow '+el.tagName+' '+el.textContent.slice(0,50));
                      }
                      for(let svg of root.querySelectorAll('svg')){
                        let v=svg.viewBox.baseVal;
                        for(let el of svg.querySelectorAll('text,rect')){let b=el.getBBox();if(b.x<0||b.y<0||b.x+b.width>v.width+.1||b.y+b.height>v.height+.1)out.push('SVG bounds '+el.textContent)}
                        let boxes=[...svg.querySelectorAll('[data-architecture-node] rect')].map(x=>x.getBBox());
                        for(let node of svg.querySelectorAll('[data-architecture-node]')){let a=node.querySelector('rect').getBBox(),b=node.querySelector('text').getBBox();if(b.x<a.x+1||b.x+b.width>a.x+a.width-1||b.y<a.y||b.y+b.height>a.y+a.height)out.push('node label outside box '+node.textContent)}
                        boxes.forEach((a,i)=>boxes.slice(i+1).forEach(b=>{if(a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y)out.push('overlapping nodes')}));
                      }
                      if(document.documentElement.scrollWidth>innerWidth+1)out.push('page overflow');return out;
                    }''')
                    assert not problems,(name,width,problems)
                    page.locator('.arch-atlas').screenshot(path=str(shots/f'{name}-{width}.png'))
                    if width==800 and export:page.locator('.arch-atlas').screenshot(path=str(out/f'{name}.png'))
                    records.append(dict(lesson=n,panel=panel['key'],width=width,status='PASS'))
                page.set_viewport_size({'width':800,'height':1100});page.emulate_media(media='print')
                page.locator('.arch-atlas').screenshot(path=str(shots/f'{name}-print.png'))
                assert page.locator('.aa-focus').is_visible()
                page.emulate_media(media='screen')
        assert not errors,errors
        browser.close()
    report=dict(status='PASS',panels=sum(map(len,PANELS.values())),records=records,print=f'{sum(map(len,PANELS.values()))} panels rendered',
                checks=['HTML reflow','text containment','table and equation overflow','SVG text/box bounds','nonoverlapping graph nodes'],
                screenshots=str(shots),portable_exports=export,live_colab='NOT_RUN')
    (ROOT/'labs/_architecture_revision_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');check(p.parse_args().export)
