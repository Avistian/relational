"""Contrast the published schematic token formula with the historical encoder."""
import hashlib,json
from pathlib import Path
import torch,tabpfn
from tabpfn.model.loading import load_model
ROOT=Path(__file__).resolve().parent

def check():
    assert tabpfn.__version__=='2.0.9';torch.set_num_threads(1)
    path=ROOT/'data/cache/foundation/tabpfn-v2.ckpt';model,_,_=load_model(path=path,model_seed=0);model.double().eval()
    captured=[]
    hook=model.transformer_encoder.layers[0].register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
    x=torch.zeros(8,1,4,dtype=torch.float64);y=torch.tensor([0,1,0,1,0,1],dtype=torch.float64)[:,None]
    with torch.no_grad():
        model(x,y,single_eval_pos=6)
        random=torch.randn(2,48,generator=torch.Generator().manual_seed(0),dtype=torch.float64)
        position=model.feature_positional_embedding_embeddings(random)
    hook.remove();states=captured[0][0,:,:2]
    delta=float((states-position[None]).abs().max());assert delta<1e-12 and float(position.norm())>1
    assert all(torch.equal(states[0],r) for r in states)
    result=dict(status='PASS',package='tabpfn==2.0.9',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_shape=list(x.shape),context_rows=6,features_per_group=2,
        first_block_feature_states_shape=list(states.shape),position_norm=float(position.norm()),position_match_max_abs=delta,
        first_group_coordinates=position[0,:8].tolist(),
        source='model/transformer.py add_embeddings: subspace projected random vectors are added after feature encoding',
        paper='https://arxiv.org/html/2502.17361v1#S5.E2',
        interpretation='The paper schematic x_j*(u+r_j) becomes zero at x_j=0. In the historical released low-level encoder, zero normalized values with observed flags still produce the nonzero additive group-identity vector. The default also encodes feature pairs, rather than one scalar per position.',
        scope='Original low-level encoder input to block 1, with fixed zero values and observed flags. The complete raw-data wrapper would remove these constant columns; this is an encoder-computation fixture, not a claim about that wrapper accepting an all-constant table.')
    return result
if __name__=='__main__':
    result=check();(ROOT/'_token_identity_l065_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
