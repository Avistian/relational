"""Delivery checks for the editorial revision; never runs or upgrades model evidence."""
from pathlib import Path
from urllib.parse import urlsplit,unquote
import argparse,base64,hashlib,json,re,subprocess
import nbformat
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]

def check(root, baseline=None):
    reports=[];failures=[]
    for n in range(71,91):
        page=next((root/'lessons').glob(f'{n:04}-*.html'));soup=BeautifulSoup(page.read_text(),'html.parser')
        maps=soup.select('.model-map img');route=soup.select_one('.learning-route')
        assert route and maps,(n,'missing overview')
        assert len(route.select('a'))>=1 and '[Lesson' not in route.get_text(),(n,'unrendered route')
        assert soup.select_one('link[href="../assets/learning-walkthrough.css"]'),(n,'missing stylesheet')
        links=0
        for el in soup.select('[href],img[src],script[src]'):
            u=urlsplit(el.get('href',el.get('src','')))
            if u.scheme or u.netloc:continue
            target=(page.parent/unquote(u.path)).resolve() if u.path else page
            if not target.exists():failures.append((n,'missing',str(target.relative_to(root))))
            elif u.fragment and target.suffix=='.html':
                doc=soup if target==page else BeautifulSoup(target.read_text(),'html.parser')
                if not doc.find(id=u.fragment) and not doc.find(id=unquote(u.fragment)) and not doc.find('a',attrs={'name':unquote(u.fragment)}):failures.append((n,'anchor',el.get('href')))
            links+=1
        notebooks=[]
        for folder in ['labs','labs/solutions']:
            p=root/folder/(page.stem+'.ipynb')
            if not p.exists():continue
            nb=nbformat.read(p,4);nbformat.validate(nb);md='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
            assert 'THE BIG PICTURE' in md,(n,folder,'missing walkthrough')
            for img in maps:
                data=base64.b64encode((root/'assets/architectures'/Path(img['src']).with_suffix('.png').name).read_bytes()).decode()
                assert 'data:image/png;base64,'+data in md,(n,folder,'stale/missing portable diagram')
            assert not re.search(r'src="[^"\n]*assets/architectures/[^"\n]*\.svg"',md),(n,'external notebook image')
            code=[c for c in nb.cells if c.cell_type=='code'];status='not versioned at baseline'
            if baseline:
                old=subprocess.run(['git','show',f'{baseline}:{folder}/{p.name}'],cwd=ROOT,capture_output=True,text=True)
                if old.returncode==0:
                    original=nbformat.reads(old.stdout,4);previous=[c for c in original.cells if c.cell_type=='code']
                    assert [c.source for c in code]==[c.source for c in previous],(n,folder,'code changed')
                    assert [(c.execution_count,c.outputs) for c in code]==[(c.execution_count,c.outputs) for c in previous],(n,folder,'saved outputs changed')
                    status='code and outputs unchanged'
            notebooks.append({'path':str(p.relative_to(root)),'code_cells':len(code),'baseline':status})
        reports.append({'lesson':n,'diagrams':len(maps),'local_links':links,'notebooks':notebooks})
    return {'scope':'delivery and editorial consistency; not a fresh training or paper reproduction audit','lessons':reports,'link_failures':failures,'baseline':baseline}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--baseline');ap.add_argument('--output',type=Path);args=ap.parse_args()
    r=check(args.root.resolve(),args.baseline)
    if args.output:args.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'lessons':len(r['lessons']),'diagrams':sum(x['diagrams'] for x in r['lessons']),'links':sum(x['local_links'] for x in r['lessons']),'failures':r['link_failures']},indent=2))
    raise SystemExit(bool(r['link_failures']))
