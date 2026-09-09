"""Update measured lesson table; render notebook prose without changing executed code."""
import json,os
from urllib.parse import urlsplit,unquote
import numpy as np
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _build_l055 import ROOT,SLUG,build

def update_evidence():
    r=json.loads((ROOT/'_verify_l055_results.json').read_text())
    rows=[]
    for name,task in r['tasks'].items():
        for strategy,records in task.items():
            values=[]
            for arm in ('MLP','TabM-mini','XGBoost'):
                es=np.array([v['error'] for v in records[0]['arms'][arm]])
                values.append(f'{es.mean():.5f} ± {es.std(ddof=1):.5f}')
            rows.append('<tr><td>'+name+'</td><td>'+strategy+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    table='<div class="l055-detail"><table><thead><tr><th>Task</th><th>Split</th><th>MLP</th><th>TabM-mini</th><th>XGBoost</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    text='''<p>Mean ± sample SD across three training seeds. On Ecom and Homesite, all three errors rise on temporal test rows, while XGBoost stays first. On Sberbank, all three mean errors fall and the point-estimate winner changes from TabM-mini to MLP. The small neural gaps and broad temporal intervals do not establish a reliable winner.</p><p>The Ecom TabM-minus-XGBoost error margin <em>widens</em> from .01267 to .03922. This local result does not mirror the paper's headline shrinking XGBoost margin. Our different arms, omitted categories, row caps, preprocessing, data version and search budget prevent a direct comparison. Sberbank's lower temporal error also shows why “time always hurts” is not a valid rule.</p><p>Mean ranks (MLP / TabM-mini / XGBoost): random 2.667 / 1.667 / 1.667; temporal 2.333 / 2.333 / 1.333. Both Friedman p-values are .3679; Nemenyi CD=1.9136. Across three tasks, this is exploratory evidence. <strong>Local protocol verified; paper Figure 2 INCOMPARABLE; larger three-window run NOT_RUN.</strong></p>'''
    p=ROOT.parent/'lessons'/f'{SLUG}.html';soup=BeautifulSoup(p.read_text(),'html.parser')
    target=soup.find(id='measured-results');target.clear();target.append(BeautifulSoup(table+text,'html.parser'))
    for figure in soup.find_all('figure'):figure['class']=['l055-detail'];figure['tabindex']='0';figure['aria-label']='Scrollable figure'
    p.write_text(str(soup))

def refresh():
    update_evidence()
    student=build(False);nbformat.write(student,ROOT/(SLUG+'.ipynb'))
    path=ROOT/'solutions'/(SLUG+'.ipynb');prior=nbformat.read(path,as_version=4);fresh=build(True)
    old=[c for c in prior.cells if c.cell_type=='code'];new=[c for c in fresh.cells if c.cell_type=='code']
    assert len(old)==len(new)
    for a,b in zip(old,new):
        assert a.source==b.source,'Code changed: re-execute before preserving outputs'
        assert a.execution_count is not None,'Solution not yet executed'
        assert not any(o.output_type=='error' for o in a.outputs)
        b.execution_count=a.execution_count;b.outputs=a.outputs
    nbformat.write(fresh,path)
    body,_=HTMLExporter(template_name='lab').from_notebook_node(student)
    soup=BeautifulSoup(body,'html.parser')
    for el in soup.find_all(id=True):el['id']=unquote(el['id'])
    for a in soup.find_all('a',href=True):
        p=urlsplit(a['href'])
        if not p.scheme and not p.netloc and p.path:
            a['href']=os.path.relpath((ROOT/p.path).resolve(),ROOT/'html')+('#'+p.fragment if p.fragment else '')
    (ROOT/'html'/(SLUG+'.html')).write_text(str(soup))
    print('Refreshed measured table, student/solution prose, portable images and prepared HTML')

if __name__=='__main__':refresh()
