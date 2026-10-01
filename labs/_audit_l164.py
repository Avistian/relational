"""Read-only audit of immutable release inputs, source contracts and temporal access."""
import hashlib,json,os,sys,copy
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;E=P/'evidence/l164';S=P/'sources/l164/upstream'
sys.path.insert(0,str(S));torch.set_num_threads(1)

def audit(root='/tmp/l164-release'):
    from hdataset import Graph,Task,TIMESTAMPADJNAME
    from hFloatEmb import SimpleRepeater
    from hmodel import GriffinMod
    from safetensors.torch import load_file
    root=Path(root);os.chdir(root);g=Graph(root/'data');t=Task(root/'data')
    inputs=json.loads((root/'input-manifest.json').read_text())
    for f in inputs['files']:assert hashlib.sha256((root/f['path']).read_bytes()).hexdigest()==f['sha256']
    table=t.tasks['rel-f1-driver-dnf'][:];keys=list(zip(table['nodeidx'].tolist(),table['timestamp'].tolist()));assert len(keys)==len(set(keys))
    sizes=[11411,566,702];splits=[];offset=0
    for name,size in zip(['train','valid','test'],sizes):
        y=table['label'][offset:offset+size];time=table['timestamp'][offset:offset+size]
        splits.append(dict(split=name,rows=size,positive_count=int(y.sum()),cutoff_min=int(time.min()),cutoff_max=int(time.max()),key_sha256=hashlib.sha256(json.dumps(keys[offset:offset+size]).encode()).hexdigest()));offset+=size
    assert splits[0]['cutoff_max']<splits[1]['cutoff_min'] and splits[1]['cutoff_max']<splits[2]['cutoff_min']
    assert sizes==t.metatask['rel-f1-driver-dnf']['split']
    # All adjacencies stay in the downloaded F1 component.
    for typ,node in g.nodes.items():
        for edge in node.meta['in']+node.meta['out']:
            assert 'rel-f1-' in str(edge)
    checked=0;calls=0
    for node in g.nodes.values():
        original=node.getedge
        def checked_getedge(idx,fanout,timestamp,original=original,node=node):
            nonlocal checked,calls
            found=original(idx,fanout,timestamp);calls+=1
            for edge,pairs in found.items():
                for src,dst in pairs.T.tolist():
                    row=node.adj[int(idx[src])];targets=row[edge];times=row[edge+TIMESTAMPADJNAME]
                    assert ((targets==dst)&(times<timestamp[src])).any(),(edge,src,dst)
                    checked+=1
            return found
        node.getedge=checked_getedge
    torch.manual_seed(164)
    # Different owner cutoffs and repeated entities exercise owner-specific trees.
    ix=torch.tensor([0,1,2,3,11411,11412,11413,11414])
    q=table['nodeidx'][ix];cutoff=table['timestamp'][ix]
    g.subgraph('rel-f1-drivers',q,2,SimpleRepeater(512),fanout=20,timestamp=cutoff)
    driver_times=g.nodes['rel-f1-drivers'].feat[:]['timestamp']
    assert (driver_times<min(table['timestamp'])).all()
    # The few-shot helper ignores its timestamp argument in general. Here its
    # root type is entirely timeless and has no target label feature.
    mask=torch.zeros((len(q),2),dtype=torch.bool)
    torch.manual_seed(164);few,owners=g.fewshot('rel-f1-drivers',q,mask,SimpleRepeater(512),3,cutoff,prefetch_factor=1)
    assert (driver_times[few]<cutoff[owners]).all()
    # Exercise the ignored-time weakness on a minimal counterexample.
    # With root index1, source necessarily picks index0 regardless of timestamp.
    f,_=g.fewshot('rel-f1-drivers',torch.tensor([1]),torch.zeros((1,2),dtype=torch.bool),None,3,torch.tensor([5]),prefetch_factor=1)
    synthetic_future=bool((torch.tensor([10,0])[f]>=5).all());assert synthetic_future
    model=GriffinMod(hiddim=512,num_mp=4,use_rev=True,use_gate=False)
    ck=load_file(str(root/'checkpoint/model.safetensors'));model.load_state_dict(ck,strict=True)
    counts={n:int(len(node)) for n,node in g.nodes.items()}
    r=dict(status='PASS_WITH_RECORDED_SOURCE_LIMITATIONS',inputs=inputs,split_audit=splits,unique_query_keys=len(keys),node_counts=counts,temporal_sample_edges_checked=checked,temporal_sample_calls=calls,fewshot_selected_rows=len(few),fewshot_f1_timeless_roots='PASS',fewshot_general_cutoff_counterexample='FAIL: helper ignores timestamp; index prefix is not a temporal proof',checkpoint_tensor_count=len(ck),checkpoint_gate_metadata='config says true; tensor keys strictly match use_gate=False required by transfer.sh',historical_identity='NOT_ESTABLISHED',normalizer_fit_scope='NOT_ESTABLISHED_FROM_PROCESSED_RELEASE',label_sql_identity='NOT_ESTABLISHED: processed labels replayed; no claim of current RelBench archive identity',model_seed=42,subset_seeds=list(range(42,47)))
    (E/'source-audit.json').write_text(json.dumps(r,indent=2)+'\n')
    print({k:v for k,v in r.items() if k not in ['inputs','node_counts']})
    return r

if __name__=='__main__':audit()
