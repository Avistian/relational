"""Reconcile lesson prose/table with the separately measured corrected v2 artifact."""
import json
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
def refresh():
    result=json.loads((ROOT/'labs/_verify_l054_v2_results.json').read_text())
    path=ROOT/'lessons/0054-tabm-parameter-efficient-ensembling.html';s=BeautifulSoup(path.read_text(),'html.parser')
    arms=result['ranks']['arms'];rows=[]
    for name,d in result['results'].items():
        values=[]
        for arm in arms:
            stat=d['summary'][arm];fmt='.0f' if name=='house' else '.3f'
            values.append(f'<td>{stat["mean"]:{fmt}} ± {stat["sd"]:{fmt}}</td>')
        fmt='.0f' if name=='house' else '.3f'
        rows.append('<tr><th>'+name+' / '+d['metric']+'</th>'+''.join(values)+f'<td>{d["diversity"]["mean"]:{fmt}}</td></tr>')
    table='<div class="tabm-table" id="v2-results-table"><p>Scroll horizontally to compare all models.</p><div role="region" aria-label="Corrected v2 results, scroll horizontally" tabindex="0" class="lesson-table-scroll"><table class="tabm-results"><thead><tr><th>Task / error ↓</th>'+''.join('<th>'+('MLP×32' if a=='MLP-xk' else a)+'</th>' for a in arms)+'<th>Mean submodel</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></div>'
    s.find(id='v2-results-table').replace_with(BeautifulSoup(table,'html.parser'))
    winners={name:min(d['summary'],key=lambda a:d['summary'][a]['mean']) for name,d in result['results'].items()}
    summary='Corrected v2, k=32/64 epochs, three seeds: '+ '; '.join(f'{n} has lowest mean error for {a}' for n,a in winners.items())+'. '
    summary+='Mean ranks: '+', '.join(f'{a} {v:.3f}' for a,v in result['ranks']['means'].items())+f'. Exploratory Friedman p={result["ranks"]["friedman_p"]:.4f}. '
    summary+='The table shows mean ± sample SD; the last column averages errors over individual members and seeds. This is a local protocol result, not the paper benchmark.'
    s.find(id='v2-evidence-summary').string=summary
    path.write_text(str(s));print(summary)
if __name__=='__main__':refresh()
