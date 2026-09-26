import hashlib,json,platform,time
from pathlib import Path
from relkit.tgat_l103 import load_wikipedia
from relkit.sampling_l108 import benchmark_sampler
P=Path(__file__).resolve().parent
n,e,d,a=load_wikipedia(P/'l103-cache')
r=benchmark_sampler(d['full'],len(n));r.update(raw_sha256=a['raw_sha256'] if 'raw_sha256' in a else None,hardware=platform.platform(),implementation_sha256=hashlib.sha256((P/'relkit/sampling_l108.py').read_bytes()).hexdigest())
(P/'_benchmark_l108_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
