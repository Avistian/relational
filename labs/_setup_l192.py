"""Create a dependency-consistent, isolated CPU preflight environment (no models)."""
import argparse,json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l192'
parser=argparse.ArgumentParser();parser.add_argument('--env',default='/tmp/l192-repro-env');args=parser.parse_args()
start=time.monotonic();env=Path(args.env).resolve()
subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
subprocess.run([str(env/'bin/python'),'-m','pip','install','--disable-pip-version-check','-r',str(P/'l192-preflight-requirements.txt')],check=True)
site=next((env/'lib').glob('python*/site-packages'))
(site/'l192-source.pth').write_text(str(P/'sources/l192/rdblearn')+'\n')
subprocess.run([str(env/'bin/python'),'-m','pip','check'],check=True)
subprocess.run([str(env/'bin/python'),'-c','import fastdfs,rdblearn;print(fastdfs.__file__,rdblearn.__file__)'],check=True)
freeze=subprocess.check_output([str(env/'bin/python'),'-m','pip','freeze'],text=True)
(E/'preflight-environment.lock.txt').write_text(freeze)
(E/'environment-setup.json').write_text(json.dumps(dict(status='PASS',seconds=time.monotonic()-start,pip_check='PASS',scope='Isolated dependency-consistent CPU diagnostic; no historical GPU model environment'),indent=2)+'\n')
