"""Run reproducible L057 fits and freeze prediction evidence."""
import argparse,json,os,hashlib
from pathlib import Path
from relkit.cross_experiment import run_suite
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','lab','closer'],default='lab')
 p.add_argument('--include-tfm',action='store_true');p.add_argument('--checkpoint');p.add_argument('--device',default='cpu');p.add_argument('--output')
 args=p.parse_args();os.chdir(ROOT)
 r=run_suite(args.preset,include_tfm=args.include_tfm,device=args.device,checkpoint=args.checkpoint,output_dir=f'data/cache/l057-{args.preset}')
 r['source_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['relkit/cross_ensemble.py','relkit/cross_experiment.py','relkit/tabm.py']}
 out=ROOT/(args.output or (f'_verify_l057_two_family_results.json' if not args.include_tfm and args.preset=='lab' else f'_verify_l057{("_"+args.preset) if args.preset!="lab" else ""}_results.json'));out.write_text(json.dumps(r,indent=2)+'\n');print(out)
