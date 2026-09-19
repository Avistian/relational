"""Original TensorFlow 1.15 VIME on native x86 CPU, isolated from Modal's Python."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('relational-vime-paper-cpu')
image=(modal.Image.debian_slim(python_version='3.11').apt_install('libgomp1')
       .add_local_file(ROOT/'labs/reproductions/colab_vime.py','/opt/vime/colab_vime.py',copy=True)
       .add_local_file(ROOT/'labs/reproductions/requirements-vime.txt','/opt/vime/requirements-vime.txt',copy=True)
       .run_commands("cd /opt/vime && python -c 'from colab_vime import historical_python; historical_python(\"/opt/vime\")'")
       .add_local_dir(ROOT/'labs/reproductions',remote_path='/root/reproductions',ignore=['**/*results.json','**/__pycache__/**']))
volume=modal.Volume.from_name('relational-paper-071-074',create_if_missing=True)
@app.function(image=image,cpu=2,memory=8192,timeout=14400,volumes={'/evidence':volume})
def run(one_trial):
    import os,subprocess
    command=['/opt/vime/vime-python37/bin/python','/root/reproductions/vime_release.py','--output','/evidence/vime-release-results.json']
    if one_trial:command.append('--one-trial')
    env=dict(os.environ,TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
    subprocess.run(command,cwd='/root/reproductions/sources/vime',env=env,check=True)
    volume.commit()
    return '/evidence/vime-release-results.json'
@app.local_entrypoint()
def main(one_trial:bool=False):print(run.remote(one_trial))
