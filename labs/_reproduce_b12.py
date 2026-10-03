"""Read-only full-target preflight. Never dispatches cloud/API work in B12."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def audit():
    s=P/'sources/b12';ledger=json.loads((s/'source-ledger.json').read_text())
    for name,digest in ledger['files'].items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
    from bs4 import BeautifulSoup
    soup=BeautifulSoup((s/'openrfm-v1.html').read_bytes(),'html.parser')
    table=next(t for t in soup.select('table') if 'Bin-AUC' in t.get_text() and 'RT-Plurel' in t.get_text())
    # Separate extraction through column selectors verifies the serialized table.
    data=[]
    for tr in table.select('tr'):
        cells=[]
        for cell in tr.select('td,th'):
            for a in cell.select('annotation'):a.decompose()
            cells.append(cell.get_text(' ',strip=True))
        data.append(cells)
    assert data==json.loads((s/'openrfm-table5.json').read_text()) and len(data)==7
    numeric=sum(v!='N/A' for row in data[1:] for v in row[1:]);assert numeric==17
    cost=json.loads((s/'griffin/cost-decision.json').read_text());assert cost['decision']=='STOP' and cost['fresh_full_fits']==0
    return dict(status='PASS_SOURCE_AUDIT',source_files=len(ledger['files']),openrfm_table5_rows=6,openrfm_numeric_cells=17,
        targets={'Griffin Table12':'INCOMPLETE_BUDGET_GATE','OpenRFM Table5':'INCOMPLETE_SOURCE_PROTOCOL','KumoRFM-2 evaluation':'INCOMPLETE_MODEL_IDENTITY'},
        full_benchmark_execution='NOT_RUN',historical_identity='NOT_ESTABLISHED',paid_usd=0)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run-full',action='store_true');args=a.parse_args()
    if args.run_full:raise SystemExit('BLOCKED: USD0 scope; source/model/budget gates remain unresolved. No dispatch occurred.')
    r=audit();(P/'evidence/b12/source-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
