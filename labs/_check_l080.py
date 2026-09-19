"""Behavioral checks for the exit exam; deliberately reject invalid evidence."""
import copy
import numpy as np
from relkit.exit_l080 import choose_validation, binary_loss, audit_result

def check_primitives():
    assert choose_validation([.4,.2,.2]) == 1
    for errors in [[],[float('nan')],[float('inf')]]:
        try: choose_validation(errors)
        except ValueError: pass
        else: raise AssertionError('Invalid validation losses accepted')
    assert abs(binary_loss([0,1],[.2,.8])+np.log(.8)) < 1e-12
    for y,p in [([0,1],[.2]),([0,2],[.2,.3]),([0,1],[-.1,.9])]:
        try: binary_loss(y,p)
        except ValueError: pass
        else: raise AssertionError('Invalid predictions accepted')

if __name__=='__main__':
    check_primitives(); print('Primitive checks PASS')

# Loaded after primitive checks when run as a script.
def check_source():
    import types,sys,torch
    from pathlib import Path
    from relkit.ft_l080 import Transformer
    root=Path(__file__).resolve().parent
    torch.set_num_threads(1)
    source=(root/'sources/l080/ft_transformer.py.txt').read_text().split('# %%\nif __name__')[0]
    stub=types.ModuleType('lib');stub.get_activation_fn=lambda name:lambda x:x.chunk(2,-1)[0]*torch.relu(x.chunk(2,-1)[1]);stub.get_nonglu_activation_fn=lambda name:torch.relu
    saved={name:sys.modules.get(name) for name in ['lib','zero']}
    sys.modules['lib']=stub;sys.modules['zero']=types.ModuleType('zero');env={}
    try: exec(compile(source,'<pinned original>','exec'),env)
    finally:
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    cfg=dict(d_numerical=4,categories=None,token_bias=True,n_layers=3,d_token=32,n_heads=4,d_ffn_factor=4/3,attention_dropout=0.,ffn_dropout=0.,residual_dropout=0.,activation='reglu',prenormalization=True,initialization='kaiming',kv_compression=None,kv_compression_sharing=None,d_out=1)
    torch.manual_seed(80);a=Transformer(**cfg);b=env['Transformer'](**cfg);b.load_state_dict(a.state_dict());a.eval();b.eval()
    x=torch.randn(7,4,requires_grad=True);u=a(x,None);v=b(x,None)
    assert torch.allclose(u,v,atol=1e-7,rtol=0)
    ga=torch.autograd.grad(u.sum(),x,retain_graph=True)[0];gb=torch.autograd.grad(v.sum(),x)[0]
    assert torch.allclose(ga,gb,atol=1e-7,rtol=0)
    return {'output_max_error':float((u-v).abs().max().detach()),'gradient_max_error':float((ga-gb).abs().max())}

def check_evidence(path):
    import json
    from relkit.exit_l080 import load_panel
    r=json.loads(path.read_text());assert r['status']=='COMPLETE';summary=audit_result(r);assert summary==r['summary']
    arrays,meta=load_panel(path.parent,r['preset'])
    for key,panel in meta['panels'].items():
        for part in ['train','val','test']: assert arrays[key+'/'+part+'/y'].tolist()==panel['targets'][part]
    rejected=[]
    for mutation in ['missing','duplicate','selection','prediction','val_prediction','test_ids','target','overlap','time','epoch']:
        z=copy.deepcopy(r);first=z['records'][0]
        if mutation=='missing':z['records'].pop()
        elif mutation=='duplicate':z['records'].append(copy.deepcopy(first))
        elif mutation=='selection':first['selected']=1-first['selected']
        elif mutation=='prediction':first['predictions'][0]=1-first['predictions'][0]
        elif mutation=='val_prediction':first['candidates'][0]['validation_predictions'][0]=.123456
        elif mutation=='test_ids':first['test_ids'].reverse()
        elif mutation=='target':first['targets'][0]=1-first['targets'][0]
        elif mutation=='overlap':
            ids=next(iter(z['data']['panels'].values()))['ids'];ids['test'][0]=ids['train'][0]
        elif mutation=='time':z['data']['panels']['ecom-offers/temporal']['ordered_full_split']=False
        elif mutation=='epoch':next(t for t in z['records'] if t['arm']=='FT-Transformer')['candidates'][0]['best_epoch']=999
        try:audit_result(z)
        except ValueError:rejected.append(mutation)
        else:raise AssertionError('Accepted invalid evidence: '+mutation)
    return {'records':len(r['records']),'rejected_mutations':rejected}

if __name__=='__main__':
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parent
    report={'status':'PASS','ft_source_parity':check_source()}
    if (root/'_verify_l080_results.json').exists(): report['evidence']=check_evidence(root/'_verify_l080_results.json')
    (root/'_check_l080_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
