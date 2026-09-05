"""Execute the gated notebook smoke path in a temporary copy, then test resume.

Requires local Jupyter kernel sockets. Keeps the delivered student/teacher copies
unchanged; the temporary copy and smoke artifacts go to gitignored data/cache/.
"""
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

HERE=Path(__file__).resolve().parent


def main():
    notebook=nbformat.read(HERE/'solutions/0048-dcnv2.ipynb',as_version=4)
    for cell in notebook.cells:
        if cell.cell_type=='code' and cell.source.startswith('# PROVIDED — gated larger run'):
            cell.source=cell.source.replace('RUN_PAPER_REPRO=False','RUN_PAPER_REPRO=True').replace("PRESET='closer'","PRESET='smoke'")
            gate=cell.source
    notebook.cells.append(nbformat.v4.new_code_cell(gate))
    notebook.cells.append(nbformat.v4.new_code_cell('''# CHECK — changed implementation cannot reuse completed seeds
try:
    run_paper_results(preset=PRESET,output_dir=str(OUTPUT_DIR),archive_path=ARCHIVE_PATH,
                     device='cuda' if torch.cuda.is_available() else 'cpu',
                     model_cls=DCNv2,train_fn=train_dcn,predict_fn=predict_dcn,
                     implementation_id=implementation_id+'-changed')
except ValueError as exc:
    assert 'different code/data/config/device' in str(exc)
    print('PASS · changed implementation rejected')
else:
    raise AssertionError('Stale implementation reused completed seeds')'''))
    NotebookClient(notebook,timeout=600,kernel_name='python3').execute(cwd=str(HERE))
    nbformat.write(notebook,HERE/'data/cache/l048-gate-executed.ipynb')
    streams='\n'.join(o.get('text','') for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='stream')
    assert 'Reusing completed seed 0' in streams
    assert 'PASS · changed implementation rejected' in streams
    print('PASS: gated live notebook smoke, completed-seed resume, changed implementation rejection.')


if __name__=='__main__':main()
