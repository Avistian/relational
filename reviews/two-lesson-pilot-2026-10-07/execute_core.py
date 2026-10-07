"""Execute portable solution exercises in a fresh kernel; leave training unrun."""
from pathlib import Path
import hashlib,json,sys,tempfile
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[2];number=int(sys.argv[1])
p=next((ROOT/'labs/solutions').glob(f'{number:04}-*.ipynb'))
nb=nbformat.read(p,as_version=4)
stop='## NEXT STEP' if number==81 else '## CHECK · full graph'
cut=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith(stop))
partial=nbformat.v4.new_notebook(cells=nb.cells[:cut],metadata=nb.metadata)
with tempfile.TemporaryDirectory(prefix=f'l{number}-core-') as cwd:
    NotebookClient(partial,timeout=120,kernel_name='python3',resources={'metadata':{'path':cwd}}).execute()
nb.cells[:cut]=partial.cells
nb.metadata['review_execution']={'scope':'core exercises only; optional data/training cells NOT_RUN','lesson':number}
nbformat.write(nb,p)
record=ROOT/'labs'/f'_execution_l{number:03}_results.json'
archive=Path(__file__).with_name(f'l{number}-previous-execution.json')
if not archive.exists():archive.write_bytes(record.read_bytes())
result={'status':'CORE_PASS_TRAINING_NOT_RUN','executed_code_cells':sum(c.cell_type=='code' for c in partial.cells),'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'fresh portable core exercises; previous benchmark evidence retained separately','live_colab':'NOT_CHECKED','full_training':'NOT_RUN','previous_execution_record':str(archive.relative_to(ROOT))}
record.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
