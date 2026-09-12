"""Canonical complete L065 package; full model visible, four live extraction tasks."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
ROOT=Path(__file__).resolve().parent;SLUG='0065-tabpfn-query-embeddings';TITLE='Extract query representations without their own labels'
TASKS=[
('extract_target_states',"def extract_target_states(hidden,n_context,role='query'):",'Read the correct token and row role','Return B×Q×192 for query or B×C×192 for context. Validate rank, context boundary and role.','The last token is a target token; the last row is only one row.','Use separate slices for row role and target-token position.'),
('scatter_fold_embeddings','def scatter_fold_embeddings(n_rows,folds,extractor):','Restore original row identities','Call extractor with context/query positions for every fold; return N×L×D plus a list of context/query position traces. Validate exactly one write per row.','A fold concatenation can preserve shape while changing which label belongs to each vector.','Allocate after the first result. Use original query positions as destinations.'),
('concatenate_layers','def concatenate_layers(embeddings,layers):','Preserve distinct layer coordinates','Select one to three distinct one-based layers from N×L×D; return N×(rD) in the requested order.','An average cannot give the head independent coefficients for each layer.','Validate layer IDs before selecting and concatenate on the coordinate axis.'),
('choose_candidate','def choose_candidate(candidates):','Freeze a validation-only decision','Choose highest validation_accuracy; ties fewer layers, smaller C, then lexicographic layers.','The test set must never select the layer or its regularization.','Use a deterministic comparison key; each candidate contains only validation fields.')]
CAPTIONS={'gaps':'Fresh local fitted-training versus untouched-test accuracy for the context-role, query-role and raw-feature heads. Each connected pair is one seed; fitted training accuracy is not a generalization estimate.','architecture':'Actual experiment: ten query-role training folds; full-training evaluation context; all 12 target states; validation-only layer/C selection; untouched test comparison. The frozen network is complete historical v2.','roles':'Left: synthetic target-encoder inputs with context labels [0,0,1]. Right: measured own-label flip on the fixed release-check fixture, with all identities and query batches held fixed.','scatter':'Synthetic identity trace. Concatenation has the correct length but the wrong row-label alignment; scatter restores original destinations.','selection':'Synthetic selection example. A tie is resolved without looking at test accuracy; the best test score is not a permitted selection rule.','results':'Fresh author measurement on full diabetes, blood transfusion and WDBC: three overlapping stratified split seeds, ten extraction folds, one numeric view. Dots are individual runs, not confidence intervals or independent datasets.'}
def source(module):return (ROOT/'relkit'/(module+'.py')).read_text()
def nodes(module):return [n for n in ast.parse(source(module)).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
def piece(module,name):return next(ast.get_source_segment(source(module),n) for n in nodes(module) if n.name==name)
def parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/(SLUG+'.md')).read_text())
def figure(name,notebook=False):
 path='figures/l065/'+name+'-v2.png';caption=CAPTIONS[name];minimum=1100 if name in ('results','gaps') else 900 if name=='roles' else 720
 if notebook:return '<div style="max-width:100%;overflow-x:auto" role="region" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/path).read_bytes()).decode()+'" style="width:100%;min-width:'+str(minimum)+'px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+path+').'
 return '<figure id="l065-'+('architecture' if name=='architecture' else 'figure-'+name)+'"><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img style="min-width:'+str(minimum)+'px" src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+path+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 p='../' if prepared else '../labs/';links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(p+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(p+'_verify_l065_v2_results.json','Measured evidence'),(p+'l065-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 065 · four live tasks + pretrained ten-fold experiment + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'">'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Run the notebook in Colab or Jupyter.</p></aside>'
def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 065 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Author measurements](_verify_l065_v2_results.json) · [Reproduction](l065-reproduction.md)

**Skill:** extract role-correct pretrained representations, preserve row identity and freeze a validation-selected head. **Four TODOs** drive the actual full 12-layer v2 checkpoint, with PROVIDED model code visible below. **CHECK** diagnoses errors; **EXIT** saves a measured artifact and interpretation. Built with TabPFN · [Prior Labs License](sources/foundation/v2-LICENSE). CPU supported; checkpoint download and 894 head candidates make this a substantial lab. Default run: all 768 diabetes rows, seed 7, ten folds; the author uses three full datasets and three seeds. No encoder training occurs. Full 29-dataset paper benchmark: NOT_RUN.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:md(figure(m[1],True))
  elif p.strip():md(p.replace('../labs/',''))
 md('''## PROVIDED · Environment and full backbone
The next chunks implement the complete historical model from Lesson 064. Read them as the dependency of your extraction procedure. Grouping and missing flags preserve the release recipe; blocks do not call the original package. The original package runs only in an isolated source-check process. A learned group projection is added after value encoding; the schematic multiplication in Ye §5 Eq. 2 is not the exact released grouped-feature computation.''')
 code('''# PROVIDED — deterministic local imports and output directory
import hashlib,importlib.metadata,inspect,itertools,json,math,time,types,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
for p in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (p/'relkit').is_dir():ROOT=p.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start Jupyter in labs')
torch.set_num_threads(1)
output=ROOT/'data/cache/l065-student';output.mkdir(parents=True,exist_ok=True)''')
 from relkit import tabpfn_l064_v2 as back,query_embeddings_l065_v2 as core
 code('# PROVIDED — pinned identity and explicit protocol\n'+'\n'.join(n+'='+repr(getattr(back,n)) for n in ['CHECKPOINT_SHA','CHECKPOINT_URL','MODEL_CONFIG','RECIPE'])+'\nPROTOCOL='+repr(core.PROTOCOL)+'\nPRESETS65='+repr(core.PRESETS65))
 groups=[('Feature and target inputs',['group_features','impute_with_flags','soften_outliers','encode_groups','encode_targets']),('Attention and residual updates',['attention_mix','PackedAttention','row_attention','postnorm_update']),('All twelve complete blocks and native head',['V2Block','TabPFNv2']),('Strict copied weights and complete numeric data',['load_pretrained','ensure_checkpoint','load_dataset'])]
 for title,names in groups:
  md('### PROVIDED · '+title+'\n\nTrace the tensor axes against the architecture figure. These definitions execute in your notebook namespace; a changed helper changes the model used by source checks and the experiment.')
  for name in names:code('# PROVIDED — '+name+'\n'+piece('tabpfn_l064_v2',name))
 checks={
 'extract_target_states':"h=torch.arange(1*5*3*4.).reshape(1,5,3,4)\nassert torch.equal(extract_target_states(h,3),h[:,3:,2,:])\nassert extract_target_states(h,4).shape==(1,1,4), 'Keep a singleton query row axis'\nassert torch.equal(extract_target_states(h,3,'context'),h[:,:3,2,:])",
 'scatter_fold_embeddings':"folds=np.array([1,0,1,0,2,2]);seen=[]\ndef identity_probe(c,q):\n    assert not set(c)&set(q);seen.extend(q.tolist())\n    return np.broadcast_to(q[:,None,None],(len(q),12,3)).copy()\nz,trace=scatter_fold_embeddings(6,folds,identity_probe)\nassert np.array_equal(z[:,0,0],np.arange(6)) and sorted(seen)==list(range(6))\nassert len(trace)==3",
 'concatenate_layers':"z=np.arange(3*12*2).reshape(3,12,2)\nc=concatenate_layers(z,[6,9,12])\nassert c.shape==(3,6) and np.array_equal(c,np.concatenate([z[:,5],z[:,8],z[:,11]],1))",
 'choose_candidate':"candidates=[dict(layers=[6,9],C=1.,validation_accuracy=.8),dict(layers=[12],C=1.,validation_accuracy=.8),dict(layers=[6],C=.1,validation_accuracy=.7)]\nassert choose_candidate(candidates)==candidates[1]\nassert all('test_accuracy' not in c for c in candidates)"}
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece('query_embeddings_l065_v2',name) if solution else sig+'\n    raise NotImplementedError("Implement this live query-embedding operation")'))
  code('# CHECK — '+name+'\n'+checks[name]+'\nprint("CHECK passed: '+name+'")')
 md('''## PROVIDED · Extract every layer and fit a visible logistic head
The hook captures each complete block output. Your token selector is called for both roles at all twelve layers. The hook handles are removed even if a forward fails. Constant filtering uses the context only. The solver below receives only train-standardized embeddings; validation is transformed with the training mean and scale. It fits actual coefficients rather than reading a stored score. Head optimization is the peripheral library operation; your live selection and concatenation functions determine every candidate.''')
 for name in ['extract_embeddings','fit_head','fit_candidates','metric_record']:code('# PROVIDED — '+name+'\n'+piece('query_embeddings_l065_v2',name))
 md('''## PROVIDED · Evidence identity and the full experiment
Read run_experiment before running it: outer IDs split first, training folds next, scatter next, validation candidates next, test scoring last. Validation and test use all outer training rows as context in separate calls. The native, raw-feature, vanilla and fixed-layer arms are explicit. The identity traverses global dependencies inside nested code, so editing a selector called within a list comprehension is detectable. It records defaults, model methods, configuration, versions and the data source. The weight digest catches post-measurement model edits.''')
 for name in ['model_digest','stable_value','kernel_identity','run_experiment']:code('# PROVIDED — '+name+'\n'+piece('query_embeddings_l065_v2',name))
 md('''## CHECK · Real historical embeddings, row roles and stale evidence
This checker calls your current functions and constructs your visible model. It obtains all intermediate target states from the original 2.0.9 release, and separately calls its get_embeddings API for final query/context states. It also performs the fixed-membership own-label flip, the identity-coded scatter, the layer-order check and a changed-helper fingerprint check. No source output replaces your experiment's predictions.''')
 code('''# CHECK — actual live source identity
from _check_l065_v2 import check as source_check
before=kernel_identity(globals(),ROOT)
parity=source_check(globals(),save=False)
assert parity['kernel_sha256']==before['sha256']==kernel_identity(globals(),ROOT)['sha256']
model,checkpoint_config=load_pretrained(ensure_checkpoint(ROOT),TabPFNv2(**MODEL_CONFIG))
weights_before=model_digest(model)
print(json.dumps(parity,indent=2))''')
 code('''# CHECK — stale helper evidence must be rejected, including a nested call
original=extract_target_states
old_identity=kernel_identity(globals(),ROOT)['sha256']
def altered_selector(*args,**kwargs):return original(*args,**kwargs)*0
extract_target_states=altered_selector
try:
    assert kernel_identity(globals(),ROOT)['sha256']!=old_identity
    try:
        assert parity['kernel_sha256']==kernel_identity(globals(),ROOT)['sha256']
    except AssertionError:print('Stale source verification correctly rejected')
    else:raise AssertionError('Stale source result accepted')
finally:extract_target_states=original
assert parity['kernel_sha256']==kernel_identity(globals(),ROOT)['sha256']''')
 md('''## Predict, then run the full pretrained experiment
Before running, write whether the selected combination must beat native v2 on test, and why cross-fitting does not make the fitted head's training score unbiased. This cell uses all diabetes rows and all 894 validation candidates. It may take several minutes on CPU. Model download/loading and source checks are outside the extraction timer. The current kernel's output stays separate from the author's three-dataset evidence.''')
 code('''# PROVIDED — genuine local measurement, through all four live TODOs
run_config=dict(PRESETS65['lab'])
assert parity['kernel_sha256']==kernel_identity(globals(),ROOT)['sha256']
result=run_experiment(ROOT,model,run_config,globals())
assert result['kernel_identity']['sha256']==parity['kernel_sha256']
for r in result['records']:
    print(r['dataset'],r['seed'],{k:round(v['accuracy'],4) for k,v in r['methods'].items()})
    print('Selected:',r['heads']['combined']['selected'])''')
 md('''## EXIT · Save a measured, identity-bound result and interpretation
Explain the own-label boundary; give the native and selected-head test scores from your run; identify the selected layers; explain the 90% versus 100% support-size shift. A loss on test is an acceptable result, but selecting another head on that test is not. Replace the interpretation below with your own account. The completed teacher notebook contains a checked example, not a mastery judgment for you.''')
 interpretation="Own-label exclusion holds with fixed folds and query batch; context-role states retain an answer route. The selected layer set and native/head scores are recorded above, and a test loss does not invalidate validation-only selection. Training vectors see roughly 90% of training support; validation/test see all training support, so roles align while contexts still differ. A fitted head training score is not a held-out estimate."
 code('# EXIT — required explanation\ninterpretation='+repr(interpretation if solution else 'WRITE YOUR INTERPRETATION HERE')+'''\nassert len(interpretation)>100 and 'WRITE YOUR' not in interpretation
assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_sha256']==result['kernel_identity']['sha256']
assert model_digest(model)==weights_before==result['weights_sha256']
assert parity['checker_sha256']==hashlib.sha256((ROOT/'_check_l065_v2.py').read_bytes()).hexdigest()
assert parity['reference_worker_sha256']==hashlib.sha256((ROOT/'_reference_l065_v2.py').read_bytes()).hexdigest()
for r in result['records']:
    assert r['embedding_shape']==[len(r['train_ids']),12,192]
    assert len(set(r['folds']))==10 and len(r['heads']['combined']['candidates'])==894
    assert not (set(r['train_ids'])&set(r['valid_ids']) or set(r['train_ids'])&set(r['test_ids']) or set(r['valid_ids'])&set(r['test_ids']))
result['parity']=parity;result['interpretation']=interpretation
path=output/f'run-{time.time_ns()}.json';path.write_text(json.dumps(result,indent=2)+chr(10))
exit_record=dict(status='MEASURED_PRETRAINED_QUERY_EMBEDDINGS',paper_reproduction='INCOMPARABLE',run_path=str(path),run_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kernel_sha256=parity['kernel_sha256'],checkpoint_sha256=CHECKPOINT_SHA,interpretation=interpretation)
(output/'student-l065-v2-exit.json').write_text(json.dumps(exit_record,indent=2)+chr(10))
print(json.dumps(exit_record,indent=2))''')
 md('''## NEXT STEP · Same implementation, broader local evidence
The optional closer preset repeats this exact extraction/selection operator on all three numeric datasets and three seeds. It is a replication of our local panel, not the paper's full 29-task benchmark. The reproduction contract identifies the missing paper metadata. Keep your measurement separate; never update an old record's fingerprint to silence a check.''')
 code('''# PROVIDED — optional local/Colab scale-up, no cloud service is launched
RUN_BROADER=False
if RUN_BROADER:
    assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_sha256']
    larger=run_experiment(ROOT,model,PRESETS65['closer'],globals())
    (output/f'closer-{time.time_ns()}.json').write_text(json.dumps(larger,indent=2)+chr(10))
else:print('Broader current-kernel run NOT_RUN; full paper benchmark NOT_RUN.')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:v2:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  k='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[k])
  if not u.scheme and not u.netloc and u.path:tag[k]='../'+tag[k]
  elif k=='href' and tag[k].startswith('#'):tag[k]=unquote(tag[k])
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'));soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))
def build_package(notebooks=True,render=True):
 body=''
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:
   body+=figure(m[1])
   if m[1] in ('scatter','selection'):body+='<div class="l065-widget" id="l065-'+m[1]+'-widget"><p>Interactive controls require JavaScript; the figure retains the baseline trace.</p></div>'
  else:body+=markdown2html_mistune(p)
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 065 · '+TITLE+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','foundation-course','lab-access','l065-query-embeddings'])+'</head><body class="l065"><article>'
 scripts=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','l065-query-embeddings-viz'])
 page=head+'<nav><a href="../index.html">Course</a> · <a href="0064-tabpfn-v2.html">← Lesson 064</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 065</p><h1>'+TITLE+'</h1>'+launcher()+'<h2>Retrieve before reading</h2><div id="warmup"></div><div id="prediction"></div>'+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>'
 soup=BeautifulSoup(page,'html.parser')
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(str(soup))
 ref=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 065</a></nav><h1>Query embeddings: information and selection card</h1>'+launcher()+figure('architecture')+markdown2html_mistune('''
## Operation card

1. Split outer train/validation/test IDs. Save them before extraction.
2. Ten folds within outer training. Other nine folds supply known targets; held fold supplies features only.
3. Capture every complete block output H, shape 1×(C+Q)×(G+1)×192. Select H[:,C:,-1,:].
4. Scatter Q×12×192 vectors to original training positions.
5. Validation/test use query role with all outer training rows as context. Their labels never enter extraction.
6. Concatenate one to three distinct one-based layers; width 192r. Fit scaler and logistic head on training vectors only.
7. Select from 298 layer subsets ×3 C values by validation accuracy, then fewer layers/smaller C/lexicographic ties. Test is untouched until scoring.

## Critical distinctions

Frozen weights can consume labels during conditioning. Context states contain own-label information; query states must be generated without it. Role matching is not identical context distribution: training vectors see about 90% support, evaluation vectors see all training support. Cross-fitting does not make fitted head training scores unbiased. An attention mask does not ensure wrapper query independence when preprocessing inspects all supplied rows.

The 2.0.9 get_embeddings X argument supplies query rows even for data_source='train'; the selector returns context states. One call is not cross-fitting. Layer labels refer to complete block outputs, before the native head.

## Evidence boundaries

[Paper §6 and Appendix Table 10](https://arxiv.org/html/2502.17361v1#S6) · [Fresh local predictions](../labs/_verify_l065_v2_results.json) · [Actual source extraction checks](../labs/_check_l065_v2_results.json) · [Source inventory](../labs/_sources_l065_v2.json) · [Paper table audit](../labs/_paper_l065_v2_results.json) · [Reproduction contract](../labs/l065-reproduction.md).

The new local experiment includes full historical pretrained layers, ten-fold extraction and exhaustive validation-selected combinations. The original 29-task benchmark is NOT_RUN; local results are INCOMPARABLE to its published average ranks. Historical three-fold/final-layer/context-average evidence remains unchanged in _verify_l065_results.json.
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(ref+'</article></body></html>')
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L065 pretrained query-embedding package')
if __name__=='__main__':build_package()
