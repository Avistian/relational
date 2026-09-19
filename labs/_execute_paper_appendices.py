"""Execute newly added reproduction source cells; retain verified unchanged outputs.

This is incremental notebook validation, NOT a new training run of the teaching lane.
Each retained cell must byte-match an executed prior solution cell. Reproduction
training has separate smoke/full evidence files; its costly run gates remain off here.
"""
import ast
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
BACKUP=Path('/tmp/lesson-7174-solutions-before')
def executable_ast(text):
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and ast.get_docstring(node,clean=False) is not None:
            node.body=node.body[1:]  # IPython normalizes whitespace-only lines inside docstrings too.
    return ast.dump(tree)

reports=[]
for lesson in range(71,75):
    path=next((ROOT/'solutions').glob(f'00{lesson}-*.ipynb'))
    nb=nbformat.read(path,as_version=4);old=nbformat.read(BACKUP/path.name,as_version=4)
    previous={c.source:c for c in old.cells if c.cell_type=='code' and c.execution_count is not None}
    pending=[];retained=0
    for i,c in enumerate(nb.cells):
        if c.cell_type!='code':continue
        if c.source in previous:
            prior=previous[c.source];c.outputs=copy.deepcopy(prior.outputs);c.execution_count=prior.execution_count;retained+=1
        elif 'paper-reproduction' in c.metadata.get('tags',[]) or 'RUN_TEACHING_EXTENSION' in c.source or 'RUN_CONVERGENCE' in c.source:
            pending.append(i)
        else:
            raise AssertionError(f'Changed teaching code needs real re-execution: L{lesson} cell {i}: {c.source[:100]}')
    selected=nbformat.v4.new_notebook(cells=[copy.deepcopy(nb.cells[i]) for i in pending])
    with tempfile.TemporaryDirectory(prefix=f'l{lesson}-paper-inline-') as tmp:
        NotebookClient(selected,timeout=300,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
        for i,c in zip(pending,selected.cells):
            nb.cells[i].outputs=c.outputs;nb.cells[i].execution_count=c.execution_count
        # Every visible %%writefile must recreate exactly the source it displays.
        assembled={}
        for c in selected.cells:
            if c.source.startswith('%%writefile '):
                first,body=c.source.split('\n',1);file=first.split()[-1]
                assembled[file]=assembled.get(file,'')+body.rstrip('\n')+'\n'
        checked=0
        for file,body in assembled.items():
            target=Path(tmp)/file
            assert executable_ast(target.read_text())==executable_ast(body),str(target)
            checked+=1
    nbformat.write(nb,path)
    reports.append(dict(lesson=lesson,retained_previously_executed_cells=retained,newly_executed_cells=len(pending),
                        source_files_recreated=checked,paper_training='SEPARATE_EVIDENCE',live_colab='NOT_CHECKED'))
    print(reports[-1],flush=True)
(ROOT/'_paper_appendix_execution_results.json').write_text(json.dumps(reports,indent=2))
