"""Independent audit of saved recommendations, all official folds and failure boundaries."""
import copy,json,math
from pathlib import Path
import numpy as np
from relkit.bipartite_l095 import read_release,audit_release,sha256
P=Path(__file__).resolve().parent
arrays,hashes=read_release(P/'data/l095/ml-100k.zip');report=json.loads((P/'_experiment_l095_results.json').read_text())
assert report['file_sha256']==hashes and report['source_sha256']==sha256(P/'relkit/bipartite_l095.py')
checked=0
for fold in report['runs']:
    n=fold['fold'];base=arrays[f'u{n}.base'];test=arrays[f'u{n}.test']
    rows=json.loads((P/f'results/l095/fold-{n}.json').read_text())
    assert sha256(P/f'results/l095/fold-{n}.json')==fold['rankings_sha256']
    seen={};truth={}
    for u,i,r,t in base:seen.setdefault(int(u),set()).add(int(i))
    for u,i,r,t in test:
        if r>=4:truth.setdefault(int(u),set()).add(int(i))
    for name,rankings in rows.items():
        recalls=[];ndcgs=[];assert {r['user'] for r in rankings}==set(truth)
        for row in rankings:
            u=row['user'];rank=row['top_items'];relevant=truth[u]
            assert len(set(rank))==len(rank)==10 and not set(rank)&seen.get(u,set())
            assert row['candidates']==1682-len(seen.get(u,set())) and row['relevant']==len(relevant)
            recall=len(set(rank)&relevant)/len(relevant)
            dcg=sum(1/math.log2(j+2) for j,i in enumerate(rank) if i in relevant)
            ideal=sum(1/math.log2(j+2) for j in range(min(10,len(relevant))))
            assert abs(row['recall']-recall)<1e-14 and abs(row['ndcg']-dcg/ideal)<1e-14
            recalls.append(recall);ndcgs.append(dcg/ideal);checked+=1
        m=fold['test'][name];assert m['users']==len(rankings) and m['excluded_no_relevant']==943-len(rankings)
        assert abs(sum(recalls)/len(recalls)-m['recall'])<1e-12 and abs(sum(ndcgs)/len(ndcgs)-m['ndcg'])<1e-12
    # Match frozen selection to inner validation only; no model or data resampling.
    selected=max(fold['validation'],key=lambda r:r['ndcg'])['alpha'];assert selected==fold['alpha']
# Same counts are insufficient: corrupt a rating in a released test partition.
bad=dict(arrays);bad['u1.test']=arrays['u1.test'].copy();bad['u1.test'][0,2]=6
try:audit_release(bad)
except AssertionError:pass
else:raise AssertionError('Release audit accepted altered row content')
result={'status':'PASS','independently_reconstructed_user_rankings':checked,'folds':5,'raw_content_corruption_rejected':True,'source_and_data_hashes':'MATCH','selection_uses_recorded_validation':'PASS'}
(P/'_audit_l095_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
