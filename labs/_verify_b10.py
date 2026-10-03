"""Complete B10 mechanism matrix: source-shaped CPU parity, interventions, gradients."""
import hashlib,json,types
from pathlib import Path
import torch
import torch.nn.functional as F
from relkit.rt_b10 import RelationalTransformer,eligible_rows
from _test_b10 import checks,fixture,oracle
P=Path(__file__).resolve().parent;E=P/'evidence/b10';torch.set_num_threads(1)

def typed_fixture(seed=0):
    torch.manual_seed(seed);b=fixture()
    b['sem_types']=torch.tensor([[3,2,0,1,0,2,3,2,0,0]])
    b['masks']=torch.tensor([[True,False,False,False,True,True,False,False,False,False]])
    for t,d in [('number',1),('text',6),('datetime',1),('boolean',1),('col_name',6)]:b[t+'_values']=torch.randn(1,10,d,dtype=torch.float64)
    return b

def source_module():
    path=P/'sources/b10/upstream/rt/model.py'
    ledger=json.loads((P/'sources/b10/source-ledger.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ledger['files']['upstream/rt/model.py']
    # Keep original model, masks, encoders, residual blocks and loss unchanged.
    # Only replace compiled sparse attention construction/kernel with CPU SDPA math.
    source=path.read_text().replace('flex_attention = torch.compile(flex_attention)','')
    mod=types.ModuleType('pinned_rt_v1');exec(compile(source,str(path),'exec'),mod.__dict__)
    mod._make_block_mask=lambda mask,**kw:mask
    mod.flex_attention=lambda q,k,v,block_mask:F.scaled_dot_product_attention(q,k,v,attn_mask=block_mask[:,None])
    return mod

def verify():
    contracts=checks();up=source_module();rows=[]
    for seed in (0,1,2):
      b=typed_fixture(seed);torch.manual_seed(seed+10)
      source=up.RelationalTransformer(2,16,6,4,32).double()
      model=RelationalTransformer(2,16,6,4,32).double();model.load_state_dict(source.state_dict(),strict=True)
      left={k:v.clone().requires_grad_(v.is_floating_point()) for k,v in b.items()}
      right={k:v.clone().requires_grad_(v.is_floating_point()) for k,v in b.items()}
      loss,pred=model(left);ref_loss,ref=source(right)
      out_error=max((pred[k]-ref[k]).abs().max().item() for k in pred)
      loss_error=abs(loss.item()-ref_loss.item());assert out_error<1e-9 and loss_error<1e-9
      loss.backward();ref_loss.backward()
      grad_error=max((p.grad-dict(source.named_parameters())[k].grad).abs().max().item() for k,p in model.named_parameters())
      input_error=max((left[k].grad-right[k].grad).abs().max().item() for k in left if left[k].is_floating_point())
      assert grad_error<1e-9 and input_error<1e-9
      # Sequence order carries no positional feature. Permute all cell-shaped tensors.
      order=torch.tensor([7,3,9,0,5,8,1,4,6,2]);perm={k:v[:,order] for k,v in b.items()}
      pl,pp=model(perm);perm_error=max((pp[k]-pred[k].detach()[:,order]).abs().max().item() for k in pp)
      assert perm_error<1e-9 and abs(pl.item()-loss.item())<1e-9
      # Change hidden truth: predictions fixed; supervised loss must respond.
      changed={k:v.clone() for k,v in b.items()};changed['number_values'][0,4]=1000;changed['boolean_values'][0,0]*=-1
      cl,cp=model(changed);hidden_error=max((cp[k]-pred[k].detach()).abs().max().item() for k in cp)
      assert hidden_error==0 and abs(cl.item()-loss.item())>1
      # Names are input values; permutation invariance does not mean semantic-name invariance.
      renamed={k:v.clone() for k,v in b.items()};renamed['col_name_values']=-renamed['col_name_values']
      _,rp=model(renamed);name_effect=float((rp['boolean']-pred['boolean'].detach()).abs().max().detach())
      assert name_effect>1e-6
      rows.append(dict(seed=seed,output_max_error=out_error,loss_error=loss_error,parameter_gradient_max_error=grad_error,input_gradient_max_error=input_error,permutation_max_error=perm_error,hidden_target_max_error=hidden_error,name_intervention_max_delta=name_effect))
    # Random graph variants protect against a checker tailored to the worked example.
    from relkit.rt_b10 import relational_masks
    for seed in range(20):
      torch.manual_seed(seed);b=fixture();b['f2p_nbr_idxs']=torch.randint(9,22,(1,10,3));b['is_padding'][0,torch.randint(0,10,(2,))]=True
      for k,v in oracle(b).items():assert torch.equal(relational_masks(b)[k],v)
    # After strict input filtering, changing a future record cannot change selected values.
    event=torch.tensor([2.,5.,12.]);arrival=event.clone();lab=torch.zeros(3,dtype=torch.bool)
    selected=eligible_rows(event,arrival,10.,lab,3.);assert selected.tolist()==[True,True,False]
    # Test a complete prediction, not just the filter: excluded row never reaches the tokenizer.
    def forecast(future):
      amounts=torch.tensor([2.,6.,future]);visible=amounts[selected]
      x=typed_fixture(0);x['number_values'][0,4]=visible.mean().double()
      return model(x)[1]['boolean'][0,0].detach()
    assert torch.equal(forecast(30.),forecast(30000.))
    # Student mutation checks call the live functions through the checker's global names.
    import _test_b10 as tests
    mutations={
      'relational_masks':lambda b:{k:torch.ones_like(v) for k,v in oracle(b).items()},
      'safe_attention':lambda q,k,v,m:torch.zeros_like(v),
      'eligible_rows':lambda t,a,c,l,h:torch.ones_like(l)}
    rejected=[]
    for key,wrong in mutations.items():
      original=getattr(tests,key);setattr(tests,key,wrong)
      try:
        try:tests.checks()
        except AssertionError:rejected.append(key)
        else:raise AssertionError('Mutation survived '+key)
      finally:setattr(tests,key,original)
    result={'status':'COMPLETE_MECHANISM_CHECKS','contracts':contracts,'seeds':rows,'random_mask_cases':20,'mask_pair_checks':8400,'mutation_rejections':rejected,'dtype':'float64','tolerance':1e-9,'fixture_model':{'blocks':2,'width':16,'heads':4,'ff':32,'text':6},'backend_adapter':'Original sparse kernel replaced by PyTorch CPU SDPA; course uses explicit softmax. CUDA/bfloat16 parity NOT_CHECKED.','fresh_training':'NOT_RUN','paper_inference':'NOT_RUN','learner':'PENDING_WRITTEN_DEFENSE','versions':{'torch':torch.__version__}}
    (E/'mechanism.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
