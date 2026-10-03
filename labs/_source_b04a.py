"""Authenticate a bounded source search; absence here is not universal absence."""
import hashlib,json,tarfile,io,re
from pathlib import Path

def source_gate(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    for name,meta in manifest['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Source hash mismatch: '+name)
    with tarfile.open(folder/'code.tar.gz') as tar:
        names=[m.name for m in tar.getmembers() if m.isfile()]
        code={n:tar.extractfile(n).read().decode('utf8',errors='replace') for n in names}
    if 'class LinearAttention' not in code['ticl/models/linear_attention.py']:raise ValueError('Missing operator')
    hits={term:[n for n,t in code.items() if re.search(term,t,re.I)] for term in ['flash_attn','fla.ops','do_bench','Figure 9']}
    return {'experiment':'B04A-TABFLEX-FIG9-ATTENTION-SCALING','status':'INCOMPLETE_SOURCE_PROTOCOL','revision':manifest['revision'],'verified_source_files':len(names),'search_hits':hits,'published':{'operators':['FlashAttention-2','causal FlashLinearAttention','noncausal linear attention'],'head_dimensions':[32,64,128,256],'heads':[2,4,8,16],'lengths':[2**k for k in range(4,16)],'batch':10,'repetitions':5,'total_operator_repetitions':2880,'reported_failure_slots':120},'unresolved':['Exact Figure 9 hardware is not tied to a run artifact (training A100 is not sufficient evidence).','Original random-input distribution, seeds and Q/K/V sharing are not authenticated.','Dtype, kernel versions, warmups, synchronization, forward/backward scope and memory reset/aggregation are not authenticated.','Original measurement script and per-repetition numeric references were not recovered in the archived tree.'],'benchmark_runs':0,'full_pretraining':'NOT_RUN','whole_paper_benchmark':'NOT_RUN','historical_identity':'NOT_ESTABLISHED'}

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=source_gate(p/'sources/b04a')
    print(json.dumps(r,indent=2))
