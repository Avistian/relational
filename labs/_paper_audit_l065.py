"""Reanalyse the published rounded embedding table, separately from model fits."""
import hashlib,json,urllib.request
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent
URL='https://arxiv.org/html/2502.17361v1'

def audit(raw=None):
    raw=urllib.request.urlopen(URL,timeout=45).read() if raw is None else raw
    soup=BeautifulSoup(raw,'html.parser');table=soup.find(id='A3.T10');assert table is not None
    names=['TabPFN v2','Vanilla','Layer 6','Layer 9','Layer 12','Combined'];rows=[]
    for tr in table.select('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'],recursive=False)]
        if len(cells)!=8:continue
        try:values=[float(v) for v in cells[1:7]]
        except ValueError:continue
        layers=[int(v.strip()) for v in cells[7].strip('()').split(',')]
        assert 1<=len(layers)<=3 and all(1<=v<=12 for v in layers)
        rows.append(dict(dataset=cells[0],accuracy_percent=dict(zip(names,values)),selected_layers=layers))
    assert len(rows)==29 and len({r['dataset'] for r in rows})==29
    values=np.array([[r['accuracy_percent'][n] for n in names] for r in rows]);ranks=np.array([rankdata(-v,method='average') for v in values]);assert np.allclose(ranks.sum(1),21)
    published=[]
    for tr in soup.find(id='S6.T2').select('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'],recursive=False)]
        if cells and cells[0]=='Rank':published=[float(v) for v in cells[1:]]
    assert len(published)==6
    # Unknown ordering within printed ties can move a method between these bounds.
    # This is an identifiability bound, not a claim to recover the original tie rule.
    low=((values[:,:,None]<values[:,None,:]).sum(2)+1).mean(0)
    high=(values[:,:,None]<=values[:,None,:]).sum(2).mean(0)
    counts=dict(combined_better=int(sum(values[:,-1]>values[:,0])),printed_tie=int(sum(values[:,-1]==values[:,0])),combined_worse=int(sum(values[:,-1]<values[:,0])))
    result=dict(status='PASS',source_url=URL,source_sha256=hashlib.sha256(raw).hexdigest(),table='Appendix Table 10; comparison with main Table 2',rows=rows,
        method_order=names,published_table2_ranks=published,rounded_table10_average_tie_ranks=ranks.mean(0).tolist(),
        unknown_within_printed_tie_rank_bounds={n:[float(low[i]),float(high[i])] for i,n in enumerate(names)},
        published_ranks_within_those_bounds=bool(np.all(np.array(published)>=low-.005)&np.all(np.array(published)<=high+.005)),
        rounded_accuracy_mean_percent=dict(zip(names,values.mean(0).tolist())),combined_vs_native=counts,
        interpretation='Published Table 2 ranks do not exactly reconstruct from the rounded Table 10 means with average tie ranks. All published values lie inside bounds for unknown ordering within printed ties; this does not establish the actual unpublished precision or ranking procedure. Combined wins 16, ties 4 and loses 9 on the printed accuracies; it is not uniformly best.',
        scope='Frozen-paper rounded table reanalysis; no new embeddings, head fits, original split recovery, significance test or benchmark reproduction')
    return result
if __name__=='__main__':
    result=audit();(ROOT/'_paper_l065_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
