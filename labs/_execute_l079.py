"""Execute the full source-visible solution and compare computed summaries."""
import json,hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient
LAB=Path(__file__).resolve().parent
p=LAB/'solutions/0079-neural-tabular-decision-guide.ipynb'
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=180,resources={'metadata':{'path':str(LAB)}},kernel_name='python3').execute()
nbformat.write(nb,p)
actual=json.loads((LAB/'l079-student-results.json').read_text());expected=json.loads((LAB/'_verify_l079_results.json').read_text())
for key in ['records','evidence_sha256','summary','matched']:assert actual[key]==expected[key],key
assert actual['choices']==expected['synthetic_choices']
r={'status':'PASS','code_cells':sum(c.cell_type=='code' for c in nb.cells),'records_audited':actual['records'],'all_summaries_match':True,'solution_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_colab':'NOT_CHECKED','fresh_training':'NOT_RUN'}
(LAB/'_execution_l079_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
