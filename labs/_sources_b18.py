"""Freeze dated primary-source bytes and limited public release-search receipts."""
import hashlib,json,datetime
from pathlib import Path
import requests
P=Path(__file__).resolve().parent/'sources/b18';P.mkdir(exist_ok=True)
urls={'paper.html':'https://arxiv.org/html/2609.00460v1','github-search.json':'https://api.github.com/search/repositories?q=%22Animus%22+%22relational%22','openreview.json':'https://api2.openreview.net/notes?id=lkuOIfXLwJ'}
receipt=[]
for name,url in urls.items():
    r=requests.get(url,timeout=45);(P/name).write_bytes(r.content)
    receipt.append(dict(file=name,url=url,http_status=r.status_code,sha256=hashlib.sha256(r.content).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
(P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
