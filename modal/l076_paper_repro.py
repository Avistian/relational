"""Historical RelBench release replay; operator provided, cloud NOT_RUN.
modal run --detach modal/l076_paper_repro.py
modal volume get relational-l076-replay / ./l076-replay
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('relational-l076-replay')
image=(modal.Image.debian_slim(python_version='3.10').apt_install('git')
 .run_commands('pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121',
               'pip install --no-index pyg-lib==0.4.0 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html')
 .pip_install_from_requirements(str(ROOT/'labs/requirements-l076-replay.txt'))
 .add_local_file(ROOT/'labs/_replay_l076.py','/root/labs/_replay_l076.py')
 .add_local_file(ROOT/'labs/_sources_l076.json','/root/labs/_sources_l076.json'))
volume=modal.Volume.from_name('relational-l076-replay',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=14400,volumes={'/evidence':volume})
def run():
 import subprocess
 try:subprocess.run(['python','/root/labs/_replay_l076.py','--run','--output','/evidence'],check=True)
 finally:volume.commit()
@app.local_entrypoint()
def main():run.remote()
