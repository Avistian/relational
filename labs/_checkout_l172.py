"""Verify the real Pages workflow and every replay input from the Git index."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l172-clean-index-') as tmp:
    root=Path(tmp)
    subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
    manifest=json.loads((root/'labs/evidence/l172/input-manifest.json').read_text())
    for name,digest in manifest['files'].items():assert hashlib.sha256((root/'labs'/name).read_bytes()).hexdigest()==digest,name
    workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
    script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
    result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for name in ['lessons/0172-schema-tokenization.html','labs/0172-schema-tokenization.ipynb','labs/solutions/0172-schema-tokenization.ipynb','labs/html/0172-schema-tokenization.html','reference/schema-tokenization.html','labs/evidence/l172/report.json','labs/sources/l172/source-ledger.json']:
        assert (root/'public'/name).is_file(),name
    # Check that the publication tree itself preserves all embedded-replay source evidence.
    for name,digest in manifest['files'].items():assert hashlib.sha256((root/'public/labs'/name).read_bytes()).hexdigest()==digest,name
r=dict(status='PASS',check='Complete real Pages build from Git index',seconds=time.perf_counter()-start,lesson=172,unchanged_replay_inputs=len(manifest['files']),published_replay_inputs=len(manifest['files']),live_deployment='NOT_CHECKED')
(P/'_checkout_l172_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
