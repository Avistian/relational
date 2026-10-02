"""Complete six-run inference operator. Refuses a failed context audit.

Requires the pinned rt-v1 source, its native sampler, original preprocessed F1
under ~/scratch/pre/rel-f1, and both checkpoint files. No training or selection.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np

def run(source,checkpoints,audit_dir,out):
    audit=json.loads((Path(audit_dir)/'context-audit.json').read_text())
    if audit['inference_gate']!='PASS' or any(audit['summary'][k]['cells'] for k in ['unmasked_query_targets','unavailable_labels','future_cells']):
        raise RuntimeError('BLOCKED_TEMPORAL_AUDIT: do not score an invalid clean evaluation')
    out=Path(out)
    if out.exists():raise FileExistsError('Immutable output: choose a new run directory')
    out.mkdir(parents=True);sys.path.insert(0,str(source))
    import torch
    from rt.data import RelationalDataset
    from rt.model import RelationalTransformer
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(1)
    files={'pretrain':'pretrain_rel-f1_driver-dnf.pt','finetuned':'finetune-from-contd-pretrain_rel-f1_driver-dnf.pt'}
    rows=[];meta=json.loads((Path.home()/'scratch/pre/rel-f1/table_info.json').read_text());driver_offset=meta['drivers:Db']['node_idx_offset']
    for arm,filename in files.items():
        ckpt=Path(checkpoints)/filename
        net=RelationalTransformer(12,256,384,8,1024)
        net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True),strict=True)
        net=net.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
        initial={k:hashlib.sha256(v.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in net.state_dict().items()}
        for seed in [0,1,2]:
            start=time.monotonic();ds=RelationalDataset([('rel-f1','driver-dnf','did_not_finish','test',[])],8,1024,0,1,256,'all-MiniLM-L12-v2',384,seed);ds.sampler.shuffle_py(0)
            keys=[];labels=[];logits=[];nodes=[]
            with torch.inference_mode():
                for i in range(len(ds)):
                    batch=ds[i];n=int(batch.pop('true_batch_size'))
                    batch['masks'][n:]=False;batch['is_targets'][n:]=False;batch['is_padding'][n:]=True
                    target=batch['is_targets'];node=batch['node_idxs'][target];cutoff=batch['timestamps'][target]
                    fk=batch['f2p_nbr_idxs'][target][:,0]-driver_offset
                    keys.extend(zip(fk.tolist(),cutoff.tolist()));nodes.extend(node.tolist())
                    labels.extend((batch['boolean_values'][target].flatten()>0).int().tolist())
                    batch={k:v.to('cuda') for k,v in batch.items()};_,pred=net(batch)
                    logits.extend(pred['boolean'][batch['is_targets']].float().flatten().cpu().tolist())
            assert len(set(keys))==len(keys)==702
            assert sorted(nodes)==list(range(97606,98308))
            final={k:hashlib.sha256(v.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in net.state_dict().items()};assert initial==final
            np.savez_compressed(out/f'{arm}-{seed}.npz',keys=np.array(keys),labels=labels,logits=logits,nodes=nodes)
            row=dict(arm=arm,seed=seed,n=702,auc=float(roc_auc_score(labels,logits)),seconds=time.monotonic()-start,checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest());rows.append(row)
            (out/'report.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(run(a.source,a.checkpoints,a.audit_dir,a.out))
