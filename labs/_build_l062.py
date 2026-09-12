"""Scoped canonical L062: historical full pretrained inference with five live tasks."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _lesson_depth import enrich_html,enrich_notebook
from _check_l062_v2 import CHECKS
ROOT=Path(__file__).resolve().parent;SLUG='0062-tabpfn-v1';TITLE='Trace the original TabPFN through a real prediction'
TASKS=[
('normalize_context','def normalize_context(x,n):','Fit normalization only on context','Return standardized N×F rows using context sample moments, epsilon 1e-6, and the source clamp. Reject fewer than two context rows.','An apparently correct attention mask cannot undo query-dependent preprocessing.','masked_moments is provided. The first n rows alone determine moments.'),
('scale_pad','def scale_pad(x,max_features=100):','Adapt the fixed feature encoder','Scale k active columns using the historical K/k convention, then append exact zeros to reach K. Reject k outside 1..K.','The released weights learned this magnitude convention; changing it changes the function.','Compute k before padding. Padding has no trainable parameters.'),
('attention_mix','def attention_mix(q,k,v):','Read the context memory','Return scaled dot-product attention for B×H×receivers×headwidth projections, with keys/values already restricted to context.','Wrong softmax axes produce plausible shapes while mixing the wrong conditional distributions.','The final score axis enumerates sending context rows; the scale uses one head’s width.'),
('postnorm_update','def postnorm_update(h,attention_output,block):','Preserve the two residual pathways','Use the provided block modules to perform attention add/norm, then GELU feed-forward add/norm.','Pre-normalization and post-normalization are different functions with the same parameter shapes.','block exposes norm1,norm2,ff1,ff2. Track which state enters each addition.'),
('aggregate_views','def aggregate_views(logits,shifts,classes,temperature=.8):','Align classes before combining views','Restrict to active classes, undo each cyclic label shift, average logits, temperature-scale, and softmax. Validate dimensions and positive temperature.','Averaging unaligned outputs combines probabilities for different events. Averaging probabilities is also a different recipe.','A class originally named j was renamed (j+s)%K. Align output coordinates using that relation.')]
FIGURES={name:(name+'-v2.png',caption) for name,caption in [
('attention','Synthetic width-2 head: context scores 0, 1 produce weights .268941, .731059 and output (.268941,1.462117). Query-key columns are blocked in every v1 layer.'),
('preprocessing','Context-only standardization trace plus a separate K=6 scaling/padding fixture. The actual checkpoint uses K=100; preprocessing and unknown-label semantics are part of the copied-weight function.'),
('ensemble','Synthetic three-class fixture with temperature 1: undo class rotation before averaging logits. Historical wrapper temperature is .8. Probability averaging gives a different predictor.'),
('results','New author-reference pretrained inference: paired 1/4-view log losses on full data, three stratified 50/50 splits per dataset. Lines connect the same split; shared y-axis. These are overlapping split variations, not independent-dataset confidence intervals.') ]}

def piece(name):
 source=(ROOT/'relkit/tabpfn_l062_v2.py').read_text()
 return next(ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)
def parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/f'{SLUG}.md').read_text())
def figure(name,notebook=False):
 if name=='architecture':return ''
 filename,caption=FIGURES[name];path='figures/l062/'+filename
 if notebook:return f'<div style="overflow-x:auto;max-width:100%" role="region" aria-label="Scrollable computational figure"><img alt="{html.escape(caption)}" src="data:image/png;base64,{base64.b64encode((ROOT/path).read_bytes()).decode()}" style="width:760px;min-width:640px;max-width:none;height:auto"></div>\n\n{caption} [Full-size figure]({path}).'
 return '<figure><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+path+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 prefix='../' if prepared else '../labs/'
 links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(prefix+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(prefix+'_verify_l062_v2_results.json','Measured evidence'),(prefix+'l062-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 062 · five live code tasks + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'"'+(' download' if title=='Download notebook' else '')+'>'+title+'</a>' for u,title in links)+'</nav><p>The preview is read-only. Complete the runnable notebook in Colab or Jupyter; student functions are intentionally blank.</p></aside>'

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 062 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Measured author evidence](_verify_l062_v2_results.json) · [Sources](_sources_l062_v2.json) · [Reproduction contract](l062-reproduction.md)

**Mirror scope:** full historical v1 architecture and numeric inference recipe, with actual copied released weights and whole-wrapper source checks. Data tier A: real small paper datasets. Five live tasks feed a genuine pretrained prediction experiment. The offline SCM/BNN prior and 9.2 million-task training are explained but not newly run here. No random reduced encoder is passed off as TabPFN. The default lab predicts on full diabetes rows with one and four views plus a shuffled-context-label intervention; it does not optimize any Transformer weights. The larger author panel remains separately labeled reference evidence. Original paper-result reproduction: INCOMPARABLE.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:
   if m[1]!='architecture':md(figure(m[1],True))
  elif part.strip():md(part.replace('../labs/',''))
 md('''## Start the live implementation
Every provided class resolves its helper functions in this notebook. The pretrained weight loader receives your model class; the experiment receives that actual model instance. The reference package checks the result but does not replace the mechanism you are implementing. Read each coherent chunk before running it.''')
 code('''# PROVIDED — imports, paths and explicit local artifact folder
import hashlib,inspect,importlib.metadata,itertools,json,math,random,time,warnings,functools,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.model_selection import train_test_split
from IPython.display import display
for path in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (path/'relkit').is_dir():ROOT=path.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start in labs')
__file__=str(ROOT/'relkit/tabpfn_l062_v2.py')  # reference source identity only, never live-kernel identity
output=ROOT/'data/cache/l062-student';output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
VERSIONS={k:importlib.metadata.version(k) for k in ['numpy','torch','scipy','scikit-learn']}
print(VERSIONS)''')
 from relkit import tabpfn_l062_v2 as core
 code('# PROVIDED — immutable checkpoint and declared configuration\n'+'\n'.join(name+'='+repr(getattr(core,name)) for name in ['CHECKPOINT_SHA','CHECKPOINT_URL','MODEL_CONFIG','PRESETS']))
 md('''### PROVIDED · Sample moments with missing entries
Count valid values per feature, compute their mean, then sum squared deviations and divide by count−1. The helper returns sample standard deviation. All-missing or one-valid-value columns have undefined moments; later handling follows the declared numeric wrapper and is not a general missing-data model.''')
 code('# PROVIDED — context moment arithmetic\n'+piece('masked_moments'))
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece(name) if solution else sig+'\n    raise NotImplementedError("Implement this live TabPFN operation")'))
  if i==4:
   md('''### PROVIDED · Complete repeated block and row model
Follow B×N×512 through a packed qkv projection into four 128-wide heads. The key/value slice supplies only context rows to your attention function. All receiver states still enter your residual update. There is no query-label encoder call. The model's public forward receives context labels only, returns query logits only, and constructs 12 distinct blocks for the checkpoint.''')
   for item in ['V1Block','TabPFNv1']:code('# PROVIDED — '+item+'\n'+piece(item))
  code(CHECKS[name])
  if i==1:
   md('''### PROVIDED · Outlier compression
The first moment pass finds an inlier subset of context; the second establishes its bounds. Two maximum/minimum operations compress extreme tails logarithmically. Query values are transformed using those context-fitted bounds. This function calls the same visible moment helper.''')
   code('# PROVIDED — two-pass outlier softening\n'+piece('soften_outliers'))
  if i==2:
   md('''### PROVIDED · Prepare each numeric view
This function calls your context normalization, removes context-constant columns and optionally fits a Yeo–Johnson power transform to each context column. It then softens outliers. Scaling/padding occurs after the view rotates feature indices. Exception handling preserves the input column when that power transform fails, as on the named released path.''')
   code('# PROVIDED — whole numeric transform\n'+piece('preprocess_numeric'))
  if i==4:
   md('''### PROVIDED · Strict copied-weight mapping and view order
The loader verifies checkpoint bytes before reading its state. Mapping names does not change tensor values; load_state_dict(strict=True) rejects missing/unexpected model parameters. The criterion buffer is not an inference parameter. view_configurations reproduces the historical random ordering with a local generator; the seed changes views, not weights.''')
   for item in ['load_pretrained','view_configurations']:code('# PROVIDED — '+item+'\n'+piece(item))
 md('''## PROVIDED · Connect the full predictor
Read the complete path: context/query concatenate → each transform → rotations → your scale/pad → your model → your class-aligned aggregation. Query chunks limit memory but this implementation recomputes context states per chunk; it does not claim caching. Raw per-view logits and configurations are returned so the final probabilities can be reconstructed independently.''')
 for item in ['predict_numeric','ensure_checkpoint','load_dataset','run_experiment']:code('# PROVIDED — '+item+'\n'+piece(item))
 md('''## CHECK · Validate this kernel against released source
This downloads the hash-pinned 0.1.11 wheel into an isolated cache and uses it only for comparison. Full copied pretrained forward and all mapped parameter/input gradients are checked; five complete wrapper cases cover binary/multiclass data, constants, missing values and extreme queries. Your notebook namespace is the subject of the check. Expected float32 tolerance reflects different tensor operation ordering; float64 tolerances are much tighter.''')
 code('''# CHECK — live source parity, not an imported completed implementation
from _check_l062_v2 import check as validate_live_v1
parity=validate_live_v1(globals(),save=False)
print(json.dumps(parity,indent=2))''')
 names=[n.name for n in ast.parse((ROOT/'relkit/tabpfn_l062_v2.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
 md('''## PROVIDED · Capture actual kernel identity
The identity includes every visible function and model method, defaults/keyword defaults, constants/configuration/presets and library versions. It is distinct from the repository hash and from the pretrained checkpoint hash. There is no resume: each experiment starts fresh. The EXIT filename is the latest student submission, while run records use unique names.''')
 code('# PROVIDED — inspect the actual functions used\nKERNEL_NAMES='+repr(names)+'''
def stable_value(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,(list,tuple)):return [stable_value(v) for v in value]
    if isinstance(value,dict):return {str(k):stable_value(v) for k,v in value.items()}
    if inspect.isfunction(value):return {'source':inspect.getsource(value),'defaults':stable_value(value.__defaults__),'kwdefaults':stable_value(value.__kwdefaults__)}
    raise TypeError('Unsupported identity value: record it explicitly')
def capture_kernel_identity():
    sources={};defaults={}
    for name in KERNEL_NAMES:
        obj=inspect.unwrap(globals()[name])
        methods={key:value for key,value in vars(obj).items() if inspect.isfunction(value)} if inspect.isclass(obj) else {'function':obj}
        sources[name]={key:inspect.getsource(value) for key,value in methods.items()}
        defaults[name]={key:{'defaults':stable_value(value.__defaults__),'kwdefaults':stable_value(value.__kwdefaults__)} for key,value in methods.items()}
    record={'sources':sources,'defaults':defaults,'constants':{k:stable_value(globals()[k]) for k in ['MODEL_CONFIG','PRESETS','CHECKPOINT_SHA','CHECKPOINT_URL']},'versions':VERSIONS}
    record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return record
kernel_identity=capture_kernel_identity()
print('Actual live kernel:',kernel_identity['sha256'])''')
 code('''# CHECK — implementation/default/config mutations must change identity
base=capture_kernel_identity()['sha256']
original_helper=masked_moments
def altered_moments(x,mask):return original_helper(x,mask)[0]+1,original_helper(x,mask)[1]
masked_moments=altered_moments
assert capture_kernel_identity()['sha256']!=base
masked_moments=original_helper
original_forward=V1Block.forward
def altered_forward(self,h,n):return original_forward(self,h,n)*0
V1Block.forward=altered_forward
assert capture_kernel_identity()['sha256']!=base
V1Block.forward=original_forward
old_defaults=aggregate_views.__defaults__;aggregate_views.__defaults__=(1.,)
assert capture_kernel_identity()['sha256']!=base
aggregate_views.__defaults__=old_defaults
old_kw=aggregate_views.__kwdefaults__;aggregate_views.__kwdefaults__={'identity_probe':1}
assert capture_kernel_identity()['sha256']!=base
aggregate_views.__kwdefaults__=old_kw
PRESETS['smoke']['views'].append(2)
assert capture_kernel_identity()['sha256']!=base
PRESETS['smoke']['views'].pop()
MODEL_CONFIG['layers']+=1
assert capture_kernel_identity()['sha256']!=base
MODEL_CONFIG['layers']-=1
assert capture_kernel_identity()['sha256']==base
print('Helper, model method, defaults, keyword defaults, preset and model config mutations detected; restored')''')
 md('''## Predict, then run actual pretrained inference
The default uses full diabetes rows, one stratified 50/50 split with seed 7, and the same held-out rows for one versus four views. This is a fresh local experiment through your functions. It uses a previously trained checkpoint; no optimizer or synthetic pretraining runs. Loading/source checks are excluded from the recorded per-prediction times. Before running, write which view budget you expect to win and whether shuffling context labels should improve log loss.''')
 code('''# PROVIDED — instrument the exact five live tasks and load your full model
call_counts={name:0 for name in ['normalize_context','scale_pad','attention_mix','postnorm_update','aggregate_views']}
def counted(name,fn):
    @functools.wraps(fn)
    def call(*args,**kwargs):
        call_counts[name]+=1
        return fn(*args,**kwargs)
    return call
for name in call_counts:globals()[name]=counted(name,globals()[name])
model,checkpoint_config=load_pretrained(ensure_checkpoint(ROOT),TabPFNv1(**MODEL_CONFIG))
run_config=dict(datasets=['diabetes'],seeds=[7],cap=None,views=[1,4])
result=run_experiment(ROOT,'lab',model=model,config=run_config)
assert all(count>0 for count in call_counts.values()), 'One or more TODO functions did not execute downstream'
print('Live task calls',call_counts)
display(pd.DataFrame([{k:r[k] for k in ['dataset','seed','views','log_loss','auc','seconds']} for r in result['records']]))''')
 md('''### CHECK · Remove the context-label association
Keep context/query features, row split, class counts, model weights and view settings fixed. Shuffle only context labels using seed 6207. Score against the same hidden query labels after prediction. This is a mechanism intervention, not a new trained baseline. A shuffled-label prediction can accidentally help one small split, so inspect the measured outcome instead of asserting the preferred story.''')
 code('''# PROVIDED — measured shuffled-context-label intervention
x,y=load_dataset('diabetes');record=result['records'][-1];tr=np.array(record['train_ids']);te=np.array(record['test_ids'])
shuffled=y[tr][np.random.default_rng(6207).permutation(len(tr))]
p_shuffled,trace_shuffled=predict_numeric(model,x[tr],shuffled,x[te],views=4,seed=7)
intervention={'kind':'context_label_shuffle','shuffle_seed':6207,'train_ids':tr.tolist(),'test_ids':te.tolist(),'context_labels':shuffled.tolist(),'targets':y[te].tolist(),'probabilities':p_shuffled.tolist(),'trace':trace_shuffled,'log_loss':log_loss(y[te],p_shuffled.numpy()),'auc':roc_auc_score(y[te],p_shuffled[:,1].numpy())}
assert np.array_equal(np.bincount(shuffled),np.bincount(y[tr]))
print('Correct context labels:',record['log_loss'],'Shuffled labels:',intervention['log_loss'])
print('Paired difference (shuffled − correct):',intervention['log_loss']-record['log_loss'])''')
 code('''# CHECK — query isolation on the same complete live wrapper
p_alone,_=predict_numeric(model,x[tr],y[tr],x[te[:1]],views=4,seed=7)
p_joined,_=predict_numeric(model,x[tr],y[tr],x[te[:6]],views=4,seed=7)
query_delta=float((p_alone-p_joined[:1]).abs().max())
assert query_delta<3e-5, 'Check preprocessing scope, dropout and query-key access'
print('Alone versus joined probability max delta',query_delta)''')
 md('''### Predict a failure path before executing it
The historical wrapper fits power parameters only on context, but its try/except also encloses transformation of query rows. Predict what happens when an added query makes that transformation overflow: which earlier rows fall back with it? The saved fixture was independently confirmed against the official 0.1.11 wrapper. This cell reruns your live predictor, not the saved probabilities. The one-view none-transform control separates the failure from attention. The strict numeric trigger below is checked for the author library versions; another environment can expose a different trigger without invalidating the control-flow diagnosis.''')
 code('''# CHECK — live historical power-fallback counterexample
fixture=json.loads((ROOT/'_query_fallback_l062_results.json').read_text())
fx=torch.tensor(fixture['context'])[:,None];fy=torch.tensor(fixture['labels']);fq=torch.tensor(fixture['query']);fe=torch.tensor(fixture['extra_query'])
fa,_=predict_numeric(model,fx,fy,fq,views=4,seed=0)
fb,_=predict_numeric(model,fx,fy,torch.cat([fq,fe]),views=4,seed=0)
fna,_=predict_numeric(model,fx,fy,fq,views=1,seed=0)
fnb,_=predict_numeric(model,fx,fy,torch.cat([fq,fe]),views=1,seed=0)
fallback_delta=float((fa-fb[:1]).abs().max());none_delta=float((fna-fnb[:1]).abs().max())
assert none_delta<3e-5, 'The none-transform control should remain independent'
if VERSIONS==fixture['versions']:
    assert fallback_delta>.05, 'Recorded-environment fallback counterexample should be visible'
else:print('Environment differs: inspect whether this numerical trigger still overflows')
fallback_observed={'alone':fa.tolist(),'batched':fb.tolist(),'probability_max_delta':fallback_delta,'none_view_max_delta':none_delta,'fixture_reference':'_query_fallback_l062_results.json'}
print(json.dumps(fallback_observed,indent=2))''')
 md('''## Read author evidence separately
This table is the committed author panel, not your current kernel result. Three split/view-seed settings per dataset reuse the same pretrained weights; their SD describes overlapping split variability. Compare the sign of the 1→4-view change per split. A three-dataset two-budget table does not justify a full benchmark ranking or a paper speedup claim.''')
 code('''# PROVIDED — readable committed author-reference summary
author=json.loads((ROOT/'_verify_l062_v2_results.json').read_text())
author_table=pd.DataFrame(author['records'])[['dataset','seed','views','log_loss','auc','seconds']]
display(author_table.groupby(['dataset','views']).agg(log_loss_mean=('log_loss','mean'),split_sd=('log_loss','std'),auc_mean=('auc','mean')))
print('Author reference only; current kernel row count:',len(result['records']))''')
 md('''## EXIT TICKET · Interpret the actual output
Write three short sentences: where context labels enter; what the one/four-view and shuffled-label results show on this split; and why this is pretrained inference rather than new pretraining or full paper-result reproduction. Include the measured direction even if it contradicts your prediction. A successful code run without this explanation is not mastery.''')
 code('''# TODO — written interpretation; the EXIT validates and saves it
interpretation = '''+('f'+repr('Context labels enter the linear label encoder before all twelve blocks. On diabetes split 7, one-view log loss is {result["records"][0]["log_loss"]:.4f}, four-view loss is {record["log_loss"]:.4f}, and shuffled-context-label loss is {intervention["log_loss"]:.4f}; these are local measured directions, not universal guarantees. The power-fallback counterexample changes the original query probability by {fallback_delta:.4f}, while the none-transform control stays independent. This executes inherited pretrained weights through my completed mechanism; it does not repeat original pretraining or the eighteen-dataset benchmark.') if solution else repr(''))+'''
assert len(interpretation.split())>=35, 'Explain the mechanism, measured outcome and evidence limit in your own words'
result['intervention']=intervention
result['fallback_counterexample']=fallback_observed
result['kernel_identity']=capture_kernel_identity()
result['call_counts']=dict(call_counts)
result['parity']=parity
result['query_batch_max_delta']=query_delta
result['interpretation']=interpretation
result['run_config']=run_config
result['model_config']=MODEL_CONFIG
assert result['kernel_identity']['sha256']==kernel_identity['sha256'], 'Restore all mutations before saving'
run_path=output/f'run-{time.time_ns()}.json'
run_path.write_text(json.dumps(result,indent=2)+'\\n')
exit_record={'status':'MEASURED_PRETRAINED_INFERENCE','paper_reproduction':'INCOMPARABLE','run_path':str(run_path),'kernel_sha256':result['kernel_identity']['sha256'],'checkpoint_sha256':CHECKPOINT_SHA,'call_counts':call_counts,'query_batch_max_delta':query_delta,'interpretation':interpretation}
(output/'exit.json').write_text(json.dumps(exit_record,indent=2)+'\\n')
print(json.dumps(exit_record,indent=2))''')
 md('''## NEXT STEP · Closer inference protocol
The same visible model can run the three paper datasets with all rows, five local 50/50 split seeds and up to 32 views. This is an inference reproduction track, not synthetic prior training. It still omits fifteen main-table datasets, tuned competitors and exact official split recovery; the reproduction contract lists the boundaries. The gate is off. The standalone CPU/Colab runner and optional Modal operator use the same versioned code; no cloud job has been run. A `paper` preset is intentionally unsupported rather than falsely labeling this subset faithful.''')
 code('''# PROVIDED — post-EXIT explicit scale-up gate, off by default
RUN_PAPER_REPRO=False
if RUN_PAPER_REPRO:
    larger=run_experiment(ROOT,'closer',model=model)
    larger['kernel_identity']=capture_kernel_identity()
    path=output/f'closer-{time.time_ns()}.json';path.write_text(json.dumps(larger,indent=2)+'\\n')
    print('Closer subset inference measured; full paper benchmark INCOMPARABLE:',path)
else:print('Closer subset track NOT_RUN in this kernel; full original pretraining NOT_RUN.')''')
 nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}}),62)
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for previous,current in zip(before,after):current.outputs=previous.outputs;current.execution_count=previous.execution_count
   if 'execution_verification' in old.metadata:nb.metadata['execution_verification']=old.metadata['execution_verification']
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  key='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[key])
  if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'))
 soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def evidence_table():
 import numpy as np
 result=json.loads((ROOT/'_verify_l062_v2_results.json').read_text())
 out='<p>Author reference · mean ± sample SD over three overlapping split/view seeds; no independent-dataset confidence interval.</p><div class="table-scroll" role="region" aria-label="Scrollable measured inference results" tabindex="0"><table><thead><tr><th>Dataset</th><th>Views</th><th>Log loss</th><th>AUC</th></tr></thead><tbody>'
 for name in ['diabetes','blood_transfusion','wdbc']:
  for views in [1,4]:
   rows=[r for r in result['records'] if r['dataset']==name and r['views']==views];loss=[r['log_loss'] for r in rows];auc=[r['auc'] for r in rows]
   out+='<tr>'+''.join('<td>'+v+'</td>' for v in [name.replace('_',' '),str(views),f'{np.mean(loss):.4f} ± {np.std(loss,ddof=1):.4f}',f'{np.mean(auc):.4f} ± {np.std(auc,ddof=1):.4f}'])+'</tr>'
 return out+'</tbody></table></div>'

def build_package(notebooks=True,render=True):
 body=''
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:
   name=m[1];body+=('<div id="v1-attention-viz" class="pfn-widget"><p>Interactive controls require JavaScript; the worked example and portable figure retain the default arithmetic.</p></div>' if name=='attention' else '')+figure(name)
   if name=='results':body+=evidence_table()
  else:body+=markdown2html_mistune(part)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 062 · {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','foundation-course','lab-access','l062-tabpfn'])+'</head><body class="l062"><article>'
 opening='<nav><a href="../index.html">Course</a> · <a href="0061-prior-data-fitted-networks.html">← Lesson 061</a> · <a href="0063-synthetic-scm-prior.html">Lesson 063 →</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 062</p><h1>'+TITLE+'</h1>'+launcher()+'<section id="retrieval"><h2>Retrieve before reading</h2><div id="warmup"></div></section><div id="prediction"></div>'
 cfg=dict(lesson=62,mode='pfn',answer='Context labels enter the row embedding. Every block reads context-only keys/values and preserves each query through residuals; fixed successful preprocessing preserves that isolation, but the measured power-transform fallback can break whole-wrapper independence. This is actual pretrained inference, not fresh prior fitting.',quiz=['What supplies keys and values to a v1 prediction query?',['Only labeled context row states','All other unlabeled query states','Its own unlabeled query states'],0,'The released v1 removes query-self attention. Its residual and attention query projection preserve its own features.'])
 scripts='<script id="foundation-config" type="application/json">'+json.dumps(cfg)+'</script>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','foundation-lesson','l062-tabpfn-viz'])
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(head+opening+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>')
 ref=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 062</a></nav><h1>Historical TabPFN v1 computational reference</h1>'+launcher()+figure('ensemble')+markdown2html_mistune('## Five live operations\n\n'+'\n'.join('- `'+task[1]+'` — '+task[3] for task in TASKS))+markdown2html_mistune('''

## Identity and evidence

Full released model:100 features, 512 width, 4 heads, 1024 FFN,12 postnorm GELU blocks,10 outputs; no query-label embedding. Copied original weights are validated with full float64 forward/input/all parameter gradients and whole numeric wrapper cases. New full-data local inference uses three paper datasets, three 50/50 split seeds, one/four views. It does not reproduce the complete original benchmark or pretraining.

[Measured predictions and view logits](../labs/_verify_l062_v2_results.json) · [Source parity](../labs/_check_l062_v2_results.json) · [Primary source inventory](../labs/_sources_l062_v2.json) · [Reproduction contract](../labs/l062-reproduction.md) · [Paper v6](https://arxiv.org/html/2207.01848v6) · [Historical source](https://github.com/PriorLabs/TabPFN/tree/v1.0.0/tabpfn).
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(ref+'</article></body></html>');enrich_html(62)
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L062 v2 package')
if __name__=='__main__':build_package()
