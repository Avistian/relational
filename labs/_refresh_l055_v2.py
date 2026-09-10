"""Refresh current v2 evidence and portable notebook HTML; preserve only identical executed code."""
import json,os
from urllib.parse import urlsplit,unquote
import numpy as np
import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter
from _build_l055 import ROOT,SLUG,build


def update_evidence():
    result=json.loads((ROOT/'_verify_l055_v2_results.json').read_text())
    paper=json.loads((ROOT/'_paper_l055_results.json').read_text())
    path=ROOT.parent/'lessons'/(SLUG+'.html');s=BeautifulSoup(path.read_text(),'html.parser')
    rows=[]
    for name,task in result['tasks'].items():
        for strategy,records in task.items():
            cells=[name,strategy]
            for arm in ('MLP','TabM-mini','XGBoost'):
                v=np.array([r['error'] for r in records[0]['arms'][arm]])
                cells.append(f'{v.mean():.5f} ± {v.std(ddof=1):.5f}')
            rows.append(cells)
    def table(headers,rows):
        return '<p>On narrow screens, scroll horizontally to inspect the complete table.</p><div class="l055-detail" role="region" tabindex="0" aria-label="Scrollable numeric comparison table"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    rank=result['summary']
    text='<p><strong>Our corrected local measurements:</strong> mean ± sample SD over three seeds on one fixed capped split pair. Ecom changes from TabM-mini to XGBoost as the point-estimate winner; Homesite retains XGBoost; Sberbank changes from XGBoost to MLP. These small selected experiments do not establish stable population rankings.</p>'
    text+='<p>All three Sberbank mean errors are lower temporally; all three Ecom mean errors are higher. This illustrates why time need not always increase error. Both model fitting and test populations change. These are current v2 results; historical scores retain their original operator and are not relabeled as corrected runs.</p>'
    text+='<p>Mean ranks in MLP / TabM-mini / XGBoost order: random '+ ' / '.join(f'{v:.3f}' for v in rank['random']['mean_ranks'])+'; temporal '+' / '.join(f'{v:.3f}' for v in rank['temporal']['mean_ranks'])+'. Friedman p values '+f"{rank['random']['friedman_p']:.4f} / {rank['temporal']['friedman_p']:.4f}"+'; CD = 1.9136. Only three dataset units: exploratory, low-power comparisons. <strong>INCOMPARABLE to fresh paper training.</strong></p>'
    target=s.find(id='measured-results');target.clear();target.append(BeautifulSoup(table(['Task','Split','MLP','TabM-mini','XGBoost'],rows)+text,'html.parser'))
    target.find_parent('details')['id']='local-reveal'
    rows=[]
    for name in dict.fromkeys(r['task'] for r in paper['summary']):
        records={(r['model'],r['protocol']):r for r in paper['summary'] if r['task']==name}
        metric=records['XGBoost','random']['metric'];sign=1 if metric=='roc-auc' else -1
        margins=[sign*(records['XGBoost',p]['mean']-records['MLP-PLR',p]['mean']) for p in ('random','temporal')]
        rows.append([name,'AUROC' if metric=='roc-auc' else 'RMSE',*[f'{v:+.6f}' for v in margins],f'{margins[1]-margins[0]:+.6f}'])
    target=s.find(id='paper-report-table');target.clear();target.append(BeautifulSoup(table(['Task','Units','Random advantage','Temporal advantage','Change'],rows),'html.parser'));target.find_parent('details')['id']='paper-reveal'
    for node in s.find_all(string=True):
        if 'Author-reference measurements' in str(node):node.replace_with(str(node).replace('Author-reference measurements','Our corrected local measurements'))
    path.write_text(str(s))


def prepared(student):
    body,_=HTMLExporter(template_name='lab').from_notebook_node(student)
    soup=BeautifulSoup(body,'html.parser')
    for el in soup.find_all(id=True):el['id']=unquote(el['id'])
    for a in soup.find_all('a',href=True):
        p=urlsplit(a['href'])
        if not p.scheme and not p.netloc and p.path:
            a['href']=os.path.relpath((ROOT/p.path).resolve(),ROOT/'html')+('#'+p.fragment if p.fragment else '')
    (ROOT/'html'/(SLUG+'.html')).write_text(str(soup))


def refresh():
    update_evidence()
    from _lesson_depth import enrich_html
    enrich_html(55)
    student=build(False);nbformat.write(student,ROOT/(SLUG+'.ipynb'))
    path=ROOT/'solutions'/(SLUG+'.ipynb');prior=nbformat.read(path,as_version=4);fresh=build(True)
    old=[c for c in prior.cells if c.cell_type=='code'];new=[c for c in fresh.cells if c.cell_type=='code']
    assert len(old)==len(new),'Rebuild and execute the new solution first'
    for a,b in zip(old,new):
        assert a.source==b.source,'Changed code requires execution'
        assert a.execution_count is not None and not any(o.output_type=='error' for o in a.outputs)
        b.execution_count=a.execution_count;b.outputs=a.outputs
    nbformat.write(fresh,path);prepared(student)
    print('Updated v2 evidence, canonical depth, student/solution prose and prepared HTML')

if __name__=='__main__':refresh()
