"""Numerical correspondence to untouched pinned official release, every full model stage."""
import sys,types,json,hashlib,contextlib,io,os
from pathlib import Path
import torch
from relkit import driftpfn_l068_v2 as core
ROOT=Path(__file__).resolve().parent
from _sources_l068_v2 import ensure_source
OFFICIAL=Path(os.environ['L068_OFFICIAL']) if 'L068_OFFICIAL' in os.environ else ensure_source()
def original_model(variant,checkpoint=1):
    p=types.ModuleType('tabpfn');p.__path__=[str(OFFICIAL/'tabpfn')];sys.modules['tabpfn']=p
    from tabpfn.scripts.model_builder import load_model
    family='dist_ablation_no_t2v' if variant=='noT2V' else variant
    with contextlib.redirect_stdout(io.StringIO()):m,c=load_model(str(core.ensure_file(ROOT,f'tabpfn/model_cache/tabpfn_{family}_model_{checkpoint}.cpkt',core.CHECKPOINTS[f'tabpfn_{family}_model_{checkpoint}.cpkt'])),'cpu',verbose=False)
    return m[2],c

def run():
    torch.set_num_threads(1);rows=[];fixtures={}
    for variant in ['base','dist','noT2V']:
      for dtype in [torch.float32,torch.float64]:
        for features in [1,3,6]:
          torch.manual_seed(68);x=torch.randn(1,11,features,dtype=dtype);x[:,2,0]=float('nan')
          if features>1:x[:,:,1]=3
          y=torch.tensor([[0,1,2,0,1,2]],dtype=dtype);c=torch.tensor([[0,0,1,1,2,2,3,3,4,5,6]],dtype=dtype)
          model,meta=core.load_pretrained(ROOT,variant);model=model.to(dtype)
          official,config=original_model(variant);official=official.to(dtype);official.generator_device=torch.device("cpu");official.generator.manual_seed(17)
          official_trace={};hooks=[]
          def keep(name):
            def hook(module,args,value):official_trace[name]=value.detach().clone()
            return hook
          for i,layer in enumerate(official.transformer_encoder.layers):hooks.append(layer.register_forward_hook(keep('block'+str(i))))
          hooks.append(official.transformer_encoder.register_forward_pre_hook(lambda m,args:official_trace.update(input=args[0].detach().clone())))
          xt=x.transpose(0,1);inp={'main':xt.clone()}
          if variant!='base':inp['dist_shift_domain']=c.T[...,None]
          with torch.no_grad():
            expected=official((inp,y.T.clone()),single_eval_pos=6).transpose(0,1)
            actual,trace=model(x,y,c,seed=17,return_trace=True)
          errors={k:float((trace[k]-v).abs().max()) for k,v in official_trace.items()}
          errors['logits']=float((actual-expected).abs().max())
          tolerance=1e-3 if dtype==torch.float32 else 1e-9
          assert max(errors.values())<tolerance,(variant,dtype,features,errors)
          for h in hooks:h.remove()
          rows.append(dict(variant=variant,dtype=str(dtype),features=features,errors=errors,checkpoint=meta['checkpoint'],tensors=meta['tensors'],parameters=meta['parameters']))
          if dtype==torch.float64 and features==3:fixtures[variant]=dict(x=x,y=y,c=c,seed=17,trace=trace,expected=expected,state_dict=model.state_dict())
    out=ROOT/'data/cache/l068-source-fixtures.pt';torch.save(fixtures,out)
    result=dict(status='PASS',scope='18 complete original checkpoint forwards; input tokenization and all12 blocks and logits. Uncached numeric CPU path, no optimized wrapper parity.',commit=core.COMMIT,rows=rows,fixture_path=str(out),fixture_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (ROOT/'_paper_audit_l068_v2_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':run()

def check_live_source(namespace,root):
    """Certify the notebook's actual model/operators against original source now."""
    identity=namespace['kernel_identity'](namespace,root);rows=[]
    for variant in ['base','dist','noT2V']:
      torch.manual_seed(680);x=torch.randn(1,15,3);x[:,:,0]=7;x[2:3]=x[2:3]
      x[:,3,2]=float('nan');y=torch.tensor([[0.,1,2,0,1,2,1,2]])
      c=torch.tensor([[0.,0,1,1,2,2,3,3,4,4,5,6,7,8,30]])
      actual,meta=namespace['load_pretrained'](root,variant,1);official,_=original_model(variant)
      official.generator_device=torch.device('cpu');official.generator.manual_seed(17)
      runtime=namespace['model_runtime_identity'](actual);weights=namespace['model_digest'](actual)
      with torch.no_grad():
        inp={'main':x.transpose(0,1).clone()}
        if variant!='base':inp['dist_shift_domain']=c.T[...,None]
        expected=official((inp,y.T),single_eval_pos=8).transpose(0,1)
        observed=actual(x,y,c,seed=17)
      error=float((expected-observed).abs().max());assert error<.001,(variant,error)
      rows.append(dict(variant=variant,max_logit_error=error,weights_sha256=weights,runtime_sha256=runtime,checkpoint_sha256=meta['sha256']))
    return dict(status='PASS',kernel_identity=identity,models=rows,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((ROOT/'_sources_l068_v2.json').read_bytes()).hexdigest())
