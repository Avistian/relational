"""Freeze then execute the complete course matrix; no checkpoint/model training."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine
from relkit.retrieval_b05 import episode,neighbors
P=Path(__file__).resolve().parent;E=P/'evidence/b05'
def generate():
    E.mkdir(exist_ok=True)
    d=load_wine();inputs=dict(name='sklearn Wine features only; original class labels unused',X=d.data.tolist(),columns=d.feature_names)
    raw=(json.dumps(inputs,sort_keys=True)+'\n').encode()
    protocol=dict(name='B05-COURSE-REAL-COLUMN-EPISODES',input_sha256=hashlib.sha256(raw).hexdigest(),seeds=[0,1,2],targets=[0,6,12],episode_sizes=[32,96],queries_per_episode=8,predictor='mean target of four nearest support rows; no learned TabDPT weights',anchor='seed index 0/1/2',partition='NumPy default_rng(seed) permutation of selected row IDs',retrieval='support population standard deviations; zero scale replaced by1; squared L2; row ID tie-break',target_intervention='reverse entire target column; selection identities must be unchanged',selection='none',precision='float64',paper_relationship='Course mechanism only; not paper checkpoint or pretraining reproduction')
    for path,content in [(E/'inputs.json',raw),(E/'course-protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())]:
        if path.exists() and path.read_bytes()!=content:raise ValueError('Frozen input/protocol differs')
        path.write_bytes(content)
    records=[];start=time.monotonic()
    for target in protocol['targets']:
      for seed in protocol['seeds']:
       for size in protocol['episode_sizes']:
        ep=episode(d.data,target,seed,size,size-8,seed)
        changed=d.data.copy();changed[:,target]=changed[::-1,target]
        counter=episode(changed,target,seed,size,size-8,seed)
        assert ep['selected_ids']==counter['selected_ids']
        nn=neighbors(ep['support_x'],ep['query_x'],ep['support_ids'],4)
        ymap=dict(zip(ep['support_ids'],ep['support_y']))
        pred=[float(np.mean([ymap[int(i)] for i in ids])) for ids in nn]
        records.append(dict(target=target,seed=seed,size=size,**ep,neighbor_ids=nn.tolist(),prediction=pred,target_intervention_ids=counter['selected_ids']))
    (E/'episodes.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Complete course matrix',len(records),'episodes',sum(len(r['query_ids']) for r in records),'predictions',time.monotonic()-start,'s')
if __name__=='__main__':generate()
