"""Validate all L047–070 explanations, portable diagrams and unchanged lab code."""
import base64,hashlib,json,math,re,subprocess
from pathlib import Path
from urllib.parse import unquote,urlsplit
import nbformat
import numpy as np
from bs4 import BeautifulSoup
from nbconvert.filters.markdown import markdown2html_mistune
from _lesson_depth import authored,enrich_html
from _architecture_revision import PANELS
from _revise_lesson_depth import code_hash,output_hash

ROOT=Path(__file__).resolve().parents[1];LABS=ROOT/'labs'
BASELINE='938d76f94c6aa57e7c48dce053a31a9b8f89cf22'


def arithmetic():
    import torch
    from relkit.foundation_core import attention,posterior_predictive,sample_scm
    from relkit.tabm import batchensemble_linear
    torch.set_num_threads(1)
    checks=[]
    def same(name,actual,expected):np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7);checks.append(name)
    for changed,want in [(6.,5.),(10.,8.)]:
        same(f'attention value {changed}',attention(torch.tensor([[1.]]),torch.tensor([[0.],[math.log(3)]]),torch.tensor([[2.],[changed]])).item(),want)
    x=np.array([2.,3.]);u=np.array([1.,3.]);v=np.array([1.,2.])
    same('DCNv2 rank-one cross',x+x*(u*(v@x)),[18,75])
    same('DCNv2 two-layer scalar polynomial',2+.75*2**2+.125*2**3,6)
    same('Trompt column reduction',np.array([.2,.3,.5])@np.array([[1,0],[0,2],[4,1]]),[2.2,1.1])
    same('ExcelFormer allowed averages',[2,(2+4)/2,(2+4+9)/3],[2,3,5])
    same('TabR selected context',np.array([1,math.exp(-1)])@np.array([3,4])/(1+math.exp(-1)),3.26894142137)
    same('smooth clipping',[z/math.sqrt(1+(z/3)**2) for z in [0,3,6]],[0,2.12132034356,2.683281573])
    same('coslog4 troughs',[.5*(1-math.cos(2*math.pi*math.log2(1+15*t))) for t in [0,1/15,3/15,7/15,1]],np.zeros(5))
    tx=torch.tensor([[[2.,3.],[2.,3.]]]);w=torch.tensor([[1.,2.],[3.,4.]])
    same('TabM shared matrix adapters',batchensemble_linear(tx,w,torch.tensor([[1.,1.],[1.,-1.]]),torch.ones(2,2),torch.zeros(2,2)).numpy(),[[[11,16],[-7,-8]]])
    same('probabilities versus logits',1/(1+math.exp(-.5*(math.log(9)+math.log(1.5)))),.78606123087)
    same('posterior predictive',posterior_predictive([1,1,1,0]),2/3)
    same('posterior total variance',4/21+2/63,2/9)
    sw=np.zeros((3,3));sw[0,1]=2;sw[1,2]=-1
    same('SCM computed parents',sample_scm([[1,2,3]],sw,activation=lambda z:z),[[1,4,-1]])
    sw[1,2]=0;same('SCM paired edge deletion',sample_scm([[1,2,3]],sw,activation=lambda z:z),[[1,4,3]])
    same('axial attention counts',[100*11**2,11*100*80],[12100,88000])
    same('inducing attention counts',16*500+500*16,16000)
    same('conditioned affine embedding',2*np.array([.5,-1])+np.array([.1,.3]),[1.1,-1.7])
    same('unsupported-class all-row loss',(-4*math.log(.8)-math.log(1e-12))/5,5.70471906464)
    same('cost-sensitive threshold',1/(1+9),.1)
    same('lifecycle cost break-even',[(100+30),(10+4*30)],[130,130])
    return checks


def check():
    records=[];links=0;broken=[]
    ledger={r['lesson']:r for r in json.loads((LABS/'_depth_revision_results.json').read_text())}
    for n in range(47,71):
        blocks,reference=authored(n);assert len(blocks)>=4
        path=next((ROOT/'lessons').glob(f'{n:04}-*.html'))
        before=path.read_bytes();enrich_html(n);assert path.read_bytes()==before,(n,'HTML generation is not idempotent')
        soup=BeautifulSoup(before,'html.parser')
        assert {x['data-lesson-depth'] for x in soup.select('[data-lesson-depth]')}=={b['key'] for b in blocks}
        def normalized(node):return re.sub(r'\s+',' ',node.get_text(' ',strip=True)).strip()
        for block in blocks:
            expected=BeautifulSoup(markdown2html_mistune(block['text']),'html.parser')
            assert normalized(soup.select_one(f'[data-lesson-depth="{block["key"]}"]'))==normalized(expected),(n,'HTML prose differs from source')
        assert len(soup.select('.arch-atlas'))==len(PANELS.get(n,[])),n
        notebook=next(LABS.glob(f'{n:04}-*.ipynb'));student=nbformat.read(notebook,4)
        baseline=nbformat.reads(subprocess.check_output(['git','show',f'{BASELINE}:{notebook.relative_to(ROOT)}'],cwd=ROOT).decode(),4)
        assert code_hash(student)==code_hash(baseline),(n,'student code changed')
        solution=nbformat.read(LABS/'solutions'/notebook.name,4)
        assert code_hash(solution)==ledger[n]['teacher_code_sha256'],(n,'teacher code changed')
        assert output_hash(solution)==ledger[n]['teacher_outputs_sha256'],(n,'teacher code/output pairing changed')
        for nb in [student,solution]:
            nbformat.validate(nb)
            actual={c.metadata['lesson_depth']:c.source for c in nb.cells if c.metadata.get('lesson_depth')}
            assert actual=={b['key']:b['text'] for b in blocks},(n,'notebook explanations differ from source')
            diagrams={c.metadata['architecture_revision']:c for c in nb.cells if c.metadata.get('architecture_revision')}
            assert set(diagrams)=={p['key'] for p in PANELS.get(n,[])}
            for key,cell in diagrams.items():
                payload=re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)',cell.source).group(1)
                assert base64.b64decode(payload)==(LABS/'figures/architecture-revision'/f'{n:04}-{key}.png').read_bytes(),(n,'stale diagram')
            assert all('attachment:' not in c.source for c in nb.cells if c.cell_type=='markdown')
        for c in student.cells:
            if c.cell_type=='code':assert c.execution_count is None and not c.outputs
        a=[c for c in solution.cells if c.cell_type=='code']
        assert all(x.execution_count is not None and not any(o.output_type=='error' for o in x.outputs) for x in a)
        reference_href=next(a['href'] for a in soup.select('a[href]') if a['href'].startswith('../reference/') and 'glossary' not in a['href'])
        ref=(path.parent/reference_href.split('#')[0]).resolve();rs=BeautifulSoup(ref.read_text(),'html.parser')
        assert len(rs.select('[data-depth-reference]'))==1
        assert normalized(rs.select_one('[data-depth-reference]'))==normalized(BeautifulSoup(markdown2html_mistune(reference),'html.parser')),(n,'reference differs from source')
        for page in [path,ref,LABS/'html'/notebook.with_suffix('.html').name]:
            ps=BeautifulSoup(page.read_text(),'html.parser')
            ids=[el['id'] for el in ps.select('[id]')]
            assert len(ids)==len(set(ids)),(str(page),'duplicate element IDs')
            for el in ps.select('[href],[src]'):
                value=el.get('href',el.get('src'));u=urlsplit(value)
                if u.scheme or u.netloc or not u.path:continue
                links+=1
                if not (page.parent/unquote(u.path)).resolve().exists():broken.append((str(page.relative_to(ROOT)),value))
        original_html=subprocess.check_output(['git','show',f'{BASELINE}:{path.relative_to(ROOT)}'],cwd=ROOT).decode()
        def words(text):
            s=BeautifulSoup(text,'html.parser')
            for x in s(['script','style','nav']):x.decompose()
            return len(re.findall(r'\b\w+\b',s.get_text(' ')))
        records.append(dict(lesson=n,explanations=len(blocks),before_words=words(original_html),after_words=words(before),
                            architecture_panels=len(PANELS.get(n,[])),preserved_executed_cells=len(a),status='PASS'))
    assert not broken,broken
    report=dict(status='PASS',baseline_commit=BASELINE,lessons=records,local_links=links,arithmetic_checks=arithmetic(),
                executable_code='unchanged against pre-revision student notebooks and saved teacher code ledger',
                author_outputs='exact code/output pairing checked against portable hashes captured from pre-revision backups',
                content='canonical blocks agree across 24 lessons, student/teacher notebooks and reference extensions',
                architecture='16 canonical panels; exact portable PNG bytes checked in both notebook variants',
                live_colab='NOT_RUN',new_training='NOT_RUN; explanatory/visual revision')
    (LABS/'_depth_delivery_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='lessons'},indent=2))


if __name__=='__main__':check()
