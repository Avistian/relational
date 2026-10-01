"""Ensure recovery changes only run scheduling, never inference operations."""
from pathlib import Path
P=Path(__file__).resolve().parent
source=(P/'_run_l169.py').read_text().split("if __name__=='__main__':",1)[0]
expected=source.replace('def run169(root,source,out,phase):','def run169_tail(root,source,out,phase,jobs):').replace("                    name=f'{db}-{arm}-{k}-{seed}'", "                    if (db,arm,k,seed) not in jobs:continue\n                    name=f'{db}-{arm}-{k}-{seed}'").replace("    expected=6 if phase=='pilot' else 234 if phase=='remaining' else 240","    expected=len(jobs)")
assert (P/'_run_l169_tail.py').exists(),'Bounded subset scheduler not implemented'
assert (P/'_run_l169_tail.py').read_text()==expected,'Inference code drifted beyond scheduler filter'
print('PASS: exact original code with only function signature, job filter and expected count changed')
