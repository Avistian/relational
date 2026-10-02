"""L181 local admission gate. No remote function is registered or dispatched.

modal run modal/l181_paper_repro.py
Expected exit2 while the authenticated training-health gate fails. The pinned
complete GNN trainer is visible in labs/sources/l181/upstream/examples/.
"""
from pathlib import Path
import subprocess,sys
import modal
app=modal.App('l181-autocomplete-admission')
@app.local_entrypoint()
def main():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'labs/_run_l181.py')],cwd=root)
    raise SystemExit(result.returncode)
