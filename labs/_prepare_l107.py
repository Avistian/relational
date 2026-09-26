"""Download and authenticate the inputs needed before building the Modal image."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
from sbm_l107 import load_sbm
from tgn_l102 import load_wikipedia
from auth_l107 import authenticate_wiki_cache
if __name__=='__main__':
 manifest=json.loads((P/'_sources_l107.json').read_text())
 for name,h in manifest['files'].items():assert hashlib.sha256((P/'sources/l107/original'/name).read_bytes()).hexdigest()==h,name
 load_sbm(P/'data/l107');authenticate_wiki_cache(P/'data/l102');load_wikipedia(P/'data/l102');authenticate_wiki_cache(P/'data/l102');print('Source and both complete datasets authenticated')
