"""Run with a clean interpreter containing requirements-l091-runtime.txt."""
import importlib.util,json
from pathlib import Path
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('rgcn_l091',P/'relkit/rgcn_l091.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
manifest=json.loads((P/'_sources_l091.json').read_text())
paper=m.run_aifb(P/'data/l091',manifest)
basis=m.run_aifb(P/'data/l091',manifest,seeds=range(3),bases=4)
(P/'_portable_paper_l091_results.json').write_text(json.dumps(paper,indent=2)+'\n')
(P/'_portable_basis_l091_results.json').write_text(json.dumps(basis,indent=2)+'\n')
result={'status':'PASS','paper_runs':len(paper['runs']),'basis_runs':len(basis['runs']),'environment':paper['environment'],'paper_mean':paper['mean'],'basis_mean':basis['mean'],'historical_exact_parity':'INCOMPARABLE','live_colab':'NOT_CHECKED'}
(P/'_clean_environment_l091_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
