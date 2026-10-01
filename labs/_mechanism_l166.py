"""Small, explicit course mechanisms; pretrained outputs here are not benchmark scores."""
def mechanism166(prior_fn,dfs_fn,mask_fn,model):
    import numpy as np
    import torch
    packet=prior_fn(166,parents=6,children=24,strength=1.)
    traces=[]
    for strength in [0.,1.,2.]:
        draw=prior_fn(166,parents=6,children=24,strength=strength)
        x=dfs_fn(draw['parent_ids'],draw['child_parent'],draw['values'])
        for support in [2,4]:
            for flip in [0,1]:
                y=(draw['latent'][:support]>0).astype('float32')
                if flip:y=1-y
                mask=mask_fn(len(x),support)
                assert not mask[:,support:].any(),'Query keys are forbidden'
                with torch.no_grad():
                    logits=model((torch.tensor(x,dtype=torch.float32)[None],torch.tensor(y)[None]),support)
                    prob=logits.softmax(-1)[0,:,1].tolist()
                traces.append(dict(strength=strength,support=support,flip=flip,features=x.tolist(),labels=y.tolist(),
                                   mask=mask.astype(int).tolist(),query_probability=prob))
    return dict(scope='COURSE_TWO_TABLE_SCM_WITH_RELEASED_CHECKPOINT',seed=166,traces=traces,
                note='Toy feature/label choices; not benchmark accuracy or a copy of the released prior generator.')

if __name__=='__main__':
    import os
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import json,torch
    from pathlib import Path
    from relkit.rdbpfn_l166 import relational_prior,dfs_summary,context_mask,RDBPFN
    torch.set_num_threads(1);model=RDBPFN().eval()
    state=torch.load(Path(__file__).parent/'evidence/l166/checkpoints/RDBPFN.pt',map_location='cpu',weights_only=True);state=state.get('model_state_dict',state)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()})
    report=mechanism166(relational_prior,dfs_summary,context_mask,model)
    (Path(__file__).parent/'evidence/l166/mechanism.json').write_text(json.dumps(report,indent=2)+'\n');print('12 measured context/prior interventions')
