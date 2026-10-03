"""Freeze public primary sources and authenticate inherited Griffin model/trainer."""
import hashlib,json,shutil,time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent;R=P.parent;S=P/'sources/b12'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    if (S/'source-ledger.json').exists():
        ledger=json.loads((S/'source-ledger.json').read_text())
        for name,h in ledger['files'].items():assert digest(S/name)==h,name
        print('Existing sealed sources verified');return
    sources={'openrfm-v1.html':'https://arxiv.org/html/2606.04320v1','kumo-v2.html':'https://arxiv.org/html/2604.12596v1','griffin-pmlr.html':'https://proceedings.mlr.press/v267/wang25da.html','same-name-readme.md':'https://raw.githubusercontent.com/T-Lab/OpenRFM/main/README.md','author-profile.md':'https://raw.githubusercontent.com/CurryTang/CurryTang/main/README.md'}
    attempts=[]
    for name,url in sources.items():
        response=requests.get(url,timeout=45);attempts.append(dict(url=url,status=response.status_code,retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        if response.ok:(S/name).write_bytes(response.content)
        elif name.endswith('.html'):response.raise_for_status()
    for url in ['https://api.github.com/repos/kumo-ai/kumo-rfm','https://api.github.com/repos/CurryTang/OpenRFM']:
        response=requests.get(url,timeout=30);attempts.append(dict(url=url,status=response.status_code,retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    old=P/'sources/l164';ledger=json.loads((old/'source-ledger.json').read_text());authenticated=[]
    for entry in ledger['files']:
        p=old/entry['file']
        if p.is_file():assert digest(p)==entry['sha256'];authenticated.append(entry['file'])
    dest=S/'griffin/upstream';dest.mkdir(parents=True,exist_ok=True)
    for p in (old/'upstream').iterdir():
        if p.is_file():shutil.copyfile(p,dest/p.name)
    for p in [P/'l164-reproduction.md',P/'_prepare_l164.py',P/'_run_l164.py',P/'l164-requirements.txt',P/'relkit/griffin_l164.py',R/'modal/l164_repro.py',P/'evidence/l164/cost-decision.json',old/'source-ledger.json']:
        if p.exists():shutil.copyfile(p,S/'griffin'/p.name)
    # Table 5 is recorded as source data, not newly trained measurements.
    soup=BeautifulSoup((S/'openrfm-v1.html').read_bytes(),'html.parser')
    table=next(x for x in soup.find_all('table') if 'RT-Plurel' in x.get_text() and 'Bin-AUC' in x.get_text())
    for annotation in table.find_all('annotation'):annotation.decompose()
    rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])] for tr in table.find_all('tr')]
    assert len(rows)==7 and all(len(r)==4 for r in rows)
    (S/'openrfm-table5.json').write_text(json.dumps(rows,indent=2)+'\n')
    files={str(p.relative_to(S)):digest(p) for p in S.rglob('*') if p.is_file()}
    ledger=dict(files=files,urls=sources,attempts=attempts,griffin_commit=ledger['code_commit'],inherited_griffin_authenticated=authenticated,
      boundaries={'OpenRFM':'Author checkpoint/source identity unresolved; T-Lab same-name repository is an independent Kumo reproduction, not this paper release. Failed guessed endpoint is not proof of nonexistence.','KumoRFM-2':'Historical model identity, execution access and complete protocol unresolved. HTTP responses do not prove absence.','Griffin':'Complete inherited release model/trainer and20fit contract; no new training. Cost decision is historical, not new pricing.'})
    (S/'source-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n');print('Sealed',len(files),'source files; table rows:',rows)
if __name__=='__main__':prepare()
