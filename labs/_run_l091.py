"""Run complete AIFB benchmark or separately labeled basis extension."""
import argparse,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rgcn_l091',P/'relkit/rgcn_l091.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--basis-extension',action='store_true');args=ap.parse_args()
 result=m.run_aifb(P/'data/l091',json.loads((P/'_sources_l091.json').read_text()),seeds=range(3) if args.basis_extension else range(10),bases=4 if args.basis_extension else 0)
 target='_teaching_l091_results.json' if args.basis_extension else '_paper_l091_results.json'
 (P/target).write_text(json.dumps(result,indent=2)+'\n');print(target,result['mean'],result['sample_sd'])
