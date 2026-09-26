"""Collect complete checkpoint evaluations from the recorded Modal volume."""
import argparse,hashlib,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-sha',default=hashlib.sha256((P/'relkit/leakage_l104.py').read_bytes()+(P/'_run_l104.py').read_bytes()).hexdigest());args=p.parse_args()
 assert len(args.run_sha)==64 and all(c in '0123456789abcdef' for c in args.run_sha)
 dest=P/'evidence/l104';dest.mkdir(parents=True,exist_ok=True)
 subprocess.run([str(R/'.venv/bin/modal'),'volume','get','l104-leakage-evidence',f'/{args.run_sha}/full',str(dest)+'/', '--force'],check=True)
 subprocess.run([sys.executable,str(P/'_analyze_l104.py'),str(dest/'full')],check=True)
