"""Pin primary source/code identities actually read for L070; no model refit."""
import hashlib,json,urllib.request
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).parent
if __name__=='__main__':
 cache=ROOT/'data/cache/l070-source';cache.mkdir(parents=True,exist_ok=True)
 papers={
 'tabpfn3':('https://arxiv.org/html/2605.13986v1',['2.1-2.5','3.1','3.4','Appendix C','Appendix E.2-E.3','Appendix F.1','Appendix G']),
 'tabiclv2':('https://arxiv.org/html/2602.11139v1',['3','4','6','7','Appendix A','Appendix H','Appendices J-K']),
 'tabpfn25':('https://arxiv.org/html/2511.08667v1',['2','3','Appendix C']),
 'tabm':('https://arxiv.org/html/2410.24210v3',['3.2-3.3','5.1','Appendices B-C']),
 }
 report=dict(papers={},code={},checkpoints=json.loads((ROOT/'_sources_foundation.json').read_text())['current_checkpoints'])
 for name,(url,sections) in papers.items():
  path=cache/(name+'.html')
  if not path.exists():path.write_bytes(urllib.request.urlopen(url).read())
  (cache/(name+'.txt')).write_text(BeautifulSoup(path.read_text(),'html.parser').get_text(' ',strip=True))
  report['papers'][name]=dict(url=url,read_sections=sections,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 for relative in ['tabpfn/architectures/tabpfn_v3.py','tabpfn/architectures/base/transformer.py','tabicl/_model/tabicl.py','tabicl/_model/embedding.py','tabicl/_model/learning.py','tabicl/_model/layers.py','tabicl/_model/interaction.py']:
  path=Path('/tmp/l070-current')/relative
  if path.exists():report['code'][relative]=dict(package='tabpfn 8.5.0' if relative.startswith('tabpfn/') else 'tabicl 2.2.0',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 import torch
 model=ROOT/'data/cache/foundation/tabpfn-v3-classifier-v3_default.ckpt'
 assert hashlib.sha256(model.read_bytes()).hexdigest()==report['checkpoints']['v3']['sha256']
 report['v3_loaded_checkpoint_config']=torch.load(model,map_location='cpu',weights_only=False)['config']
 report['source_wheels']=json.loads((ROOT/'_sources_foundation.json').read_text())['packages']
 (ROOT/'_sources_l070_v2.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Pinned primary sources, released source files and checkpoint identities')
