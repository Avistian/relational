"""Authenticate source, reconstruct the selected printed target and audit saved tutorial output.
--fresh refuses historical execution until unresolved identities and equivalence pass.
"""
import hashlib,json,re,sys
from pathlib import Path
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent


def replay(source_dir):
    s=Path(source_dir);manifest=json.loads((s/'manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((s/row['file']).read_bytes()).hexdigest()==row['sha256'],'SOURCE_HASH_MISMATCH '+row['file']
    html=BeautifulSoup((s/'paper.html').read_text(),'html.parser')
    for math in html.find_all('math'):math.replace_with(math.get('alttext',''))
    selected=[];task=None
    for row in html.find(id='S6.T4').select('tr'):
        cells=[c.get_text(' ',strip=True) for c in row.find_all(['td','th'])]
        if not cells:continue
        if cells[0] in ['driver-dnf','driver-top3','driver-position','qualifying-position']:task=cells.pop(0)
        if task!='qualifying-position':continue
        method=cells[0]
        for budget,value in zip([1,25,50,75,100],cells[1:]):
            numbers=re.findall(r'\d+\.\d+',value)
            assert len(numbers)==2
            selected.append(dict(method=method,budget=budget,mean=float(numbers[0]),std=float(numbers[1])))
    assert len(selected)==35
    notebook=json.loads((s/'tuto.ipynb').read_text());table=BeautifulSoup(''.join(notebook['cells'][11]['outputs'][0]['data']['text/html']),'html.parser')
    header=[c.get_text(' ',strip=True) for c in table.select('thead tr')[-1].select('th')][1:]
    saved=[]
    for tr in table.select('tbody tr'):
        values=[c.get_text(' ',strip=True) for c in tr.select('td')];assert len(values)==len(header)
        saved.append(dict(zip(header,values)))
    assert len(saved)==35 and {int(x['budget']) for x in saved}=={1,125,250,375,500}
    assert all(x['n_seeds']=='1' and x['n_batches']=='1' and x['n_runs']=='1' for x in saved)
    methods={'Random':'random','Random + exact rerank':'random_exact','Gradient raw':'gradient_raw_only','Gradient z-score':'gradient_global_z_only','Gradient robust z-score':'gradient_robust_z_only','Gradient min-max relation':'gradient_minmax_relation','Gradient + exact rerank':'gradient_exact'}
    comparison=[]
    for row in selected:
        if row['budget']!=1:continue
        record=next(x for x in saved if x['method']==methods[row['method']] and x['budget']=='1')
        comparison.append(dict(method=row['method'],budget=1,paper=row['mean'],tutorial=float(record['attacked_metric_mean']),same_protocol=False))
    assert len(comparison)==7
    return dict(status='COMPLETE_SOURCE_AND_SAVED_OUTPUT_AUDIT',paper_target='Table4 qualifying-position, 175 seed/method/budget conditions',
       selected_printed_cells=selected,saved_tutorial_rows=saved,common_budget_comparison=comparison,
       historical_status='INCOMPLETE_SOURCE_PROTOCOL_GATE',historical_execution='NOT_RUN',
       blockers=['Tutorial has one epoch/seed and different budgets; historical checkpoints/batches/environment absent from pinned tree.',
                 'Literal evaluation seed loop uses a list as a dict key; saved outputs do not come from a clean execution of current cells.',
                 'Clean model conversion fails in current-runtime probes; strict=False leaves new root biases initialized.',
                 'Candidate generation alone does not enforce temporal availability or coupled functional dependencies.',
                 'Original fitted predictions and per-seed attack artifacts are not supplied by this pinned repository.'],
       boundaries='Transcribed paper summaries and stale one-run tutorial output are not independently reproduced paper results. Full training and whole paper NOT_RUN.')


if __name__=='__main__':
    report=replay(P/'sources/b21')
    if '--fresh' in sys.argv:raise SystemExit('NOT_RUN: INCOMPLETE_SOURCE_PROTOCOL_GATE. Resolve b21-reproduction.md; no model training dispatched.')
    (P/'evidence/b21/reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(report['selected_printed_cells']),'paper cells;',len(report['saved_tutorial_rows']),'saved tutorial rows;',report['historical_execution'])
