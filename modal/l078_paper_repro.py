"""Optional CPU execution of the complete Cora port; remote execution NOT_RUN locally.
From repo root: modal run modal/l078_paper_repro.py
"""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l078-cora-gcn')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.8.0','numpy==2.2.6','scipy==1.15.3').add_local_dir(ROOT/'labs/relkit',remote_path='/work/relkit').add_local_file(ROOT/'labs/_sources_l078.json',remote_path='/work/_sources_l078.json'))
@app.function(image=image,cpu=2,timeout=3600)
def experiment():
    import sys
    sys.path.insert(0,'/work')
    from relkit.message_passing import run_cora
    return run_cora(100,'/work')
@app.local_entrypoint()
def main():
    r=experiment.remote()
    (ROOT/'labs/_modal_l078_results.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Mean:',r['mean'],'Different pinned runtime; inspect deviations before comparison.')
