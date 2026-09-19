"""Refresh measured prose/figures while preserving independently executed notebook code."""
import hashlib,json,subprocess,sys
from pathlib import Path
import nbformat
LAB=Path(__file__).resolve().parent;ROOT=LAB.parent
r=json.loads((LAB/'_paper_l084_results.json').read_text());assert len(r['runs'])==100
p=LAB/'solutions/0084-gat.ipynb';before=nbformat.read(p,as_version=4)
old=[(c.source,c.outputs,c.execution_count) for c in before.cells if c.cell_type=='code']
assert all(v[2] is not None for v in old)
for name in ['_figures_l084.py','_build_l084.py']:
 subprocess.run([sys.executable,str(LAB/name)],cwd=ROOT,check=True)
after=nbformat.read(p,as_version=4);new=[(c.source,c.outputs,c.execution_count) for c in after.cells if c.cell_type=='code'];assert old==new
report=LAB/'_execution_l084_results.json';e=json.loads(report.read_text());e['notebook_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();e['prose_refreshed_after_cli_completion']=True;e['executed_code_and_outputs_preserved']=True;report.write_text(json.dumps(e,indent=2)+'\n')
contract=LAB/'l084-reproduction.md';s=contract.read_text();start='<!-- MEASURED-L084:begin -->';end='<!-- MEASURED-L084:end -->'
if start in s:s=s[:s.index(start)]+s[s.index(end)+len(end):]
s+='\n'+start+f"\n## Measured full experiment\n\nAll100 declared seeds completed the full stopping schedule. Mean **{100*r['mean']:.3f}%**, sample SD **{100*r['sample_sd']:.3f} percentage points**, versus paper83.0 ±0.7%. Epoch counts ranged from{min(v['epochs'] for v in r['runs'])} to{max(v['epochs'] for v in r['runs'])}. Full epoch traces and file identities: `_paper_l084_results.json`. This is a measured modern port, not original-framework parity.\n"+end+'\n';contract.write_text(s)
for name in ['_browser_l084.py','_delivery_l084.py']:
 subprocess.run([sys.executable,str(LAB/name)],cwd=ROOT,check=True)
p=ROOT/'plan/lesson-084-delivery.md';s=p.read_text();marker='\n## Executed evidence\n'
if marker in s:s=s.split(marker)[0]
s+=marker+f"\nFull100-run Cora port: mean{100*r['mean']:.3f}%, sample SD{100*r['sample_sd']:.3f}pp. Independent inline seed0 replay matches every epoch and final score. Fresh isolated CPU installation and full659-epoch seed0 replay also match exactly. Source/data checks, browser controls at1100/375px, four portable figures and copied Pages delivery pass. Full notebook100-run lane NOT_RUN; CLI100-run lane executed. Historical framework parity INCOMPARABLE; live Colab and deployment NOT_CHECKED. No learner completion inferred.\n";p.write_text(s)
print('L084 package finalized with measured results and all delivery checks')
