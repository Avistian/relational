"""Execute complete solution, including all 100 Cora initializations."""
from pathlib import Path
import hashlib,json
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent;p=LAB/'solutions/0082-gcn.ipynb'
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=1800,resources={'metadata':{'path':str(LAB)}},kernel_name='python3').execute()
nbformat.write(nb,p)
r=json.loads((LAB/'l082-student-results.json').read_text());author=json.loads((LAB/'_paper_l082_results.json').read_text())
assert r['runs']==author['runs'], 'Score or stopping trace differs from canonical run'
result={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'seeds_executed':len(r['runs']),'all_seed_scores_match_author':True,'mean':r['mean'],'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED'}
(LAB/'_execution_l082_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
