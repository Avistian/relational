"""Fetch unmodified public reference files for local parity checks; no license is supplied upstream."""
import hashlib,json,urllib.request
from pathlib import Path
COMMIT='9293d10d1943c4bd4a186337cf38ba98e4c8bb99'
P=Path(__file__).resolve().parent
FILES=['module.py','graph.py','utils.py','learn_edge.py','learn_node.py','process.py','README.md','requirements.txt']
def fetch():
 root=P/'sources/l103/original';root.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((P/'_sources_l103.json').read_text()) if (P/'_sources_l103.json').exists() else None
 hashes={}
 for f in FILES:
  path=root/f
  if not path.exists():urllib.request.urlretrieve(f'https://raw.githubusercontent.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/{COMMIT}/{f}',path)
  hashes[f]=hashlib.sha256(path.read_bytes()).hexdigest()
  if manifest:assert hashes[f]==manifest['files'][f],f
 return root,hashes
if __name__=='__main__':
 root,hashes=fetch()
 (P/'_sources_l103.json').write_text(json.dumps({'commit':COMMIT,'repository':'https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs','paper':'https://arxiv.org/html/2002.07962v1','files':hashes,'license':'No license file supplied; originals fetched locally, not redistributed','targets_ap_percent':{'Table 1 Wikipedia':95.34,'Table 2 Wikipedia':93.99},'runs':10,'numerical_tolerance_pp':.5,'historical_protocol':'INCOMPARABLE','full_paper':'NOT_ESTABLISHED'},indent=2)+'\n')
 print(root)
