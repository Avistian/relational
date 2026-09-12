"""Canonical L068 full-checkpoint lesson, standalone live lab and portable figures."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_foundation import markdown2html_mistune
ROOT=Path(__file__).resolve().parent;SLUG='0068-pfns-under-temporal-shift';TITLE='Drift-Resilient TabPFN: learn how relationships change'
TASKS=[
('normalize_time','def normalize_time(c,n):','Preserve future time','Map T×B×1 times using only the first n context rows, including constant contexts and the source clipping bounds.','Fitting on future endpoints changes what extrapolation means.','Check where a query beyond the source maximum lands; handle a zero range without destroying offsets.'),
('time2vec','def time2vec(c,weight,bias):','Encode trend and periodic phase','Apply all learned affine phases and the correct activation to each coordinate.','The first coordinate and remaining 99 play different roles in every feature group.','Keep the first output linear; preserve arbitrary leading dimensions.'),
('shifted_weights','def shifted_weights(base,edge_to_relation,selected,shifts):','Shift relationships, not arbitrary edges','Return time-dependent functional weights using the causal-relation mapping and H outputs.','This live function generates the measured SCM features and labels; sparse causal selection can activate several weights.','A relationship identifier can occur multiple times; fixed edges must retain their base weights at every time.'),
('temporal_split','def temporal_split(domains,source_count,seed,cap_per_domain=None):','Freeze a historical context','Return train/id/ood integer row IDs using whole-domain boundaries, optional label-blind caps and source-domain ID holdouts.','No future labels may become context labels, even at later query horizons.','Use only domains and a local RNG; preserve the full selected row universe and disjoint partitions.'),
('dataset_summary','def dataset_summary(records):','Keep datasets as units','Return per-dataset/arm/split metric means and sample SD, with separate available-AUC counts.','Repeated cutoffs and checkpoints belong to one dataset. A later rank calculation weights datasets equally.','Group before averaging; unavailable AUC is not zero. Do not mix ID and OOD records.')]
CAPTIONS={
'architecture':'Complete released drift checkpoint: 12 alternating feature/row blocks, width 192, six heads, group size 2. Time is repeated within every feature group. Queries provide features/time and a censored target channel; the prediction head reads the query target token.',
'time':'Illustrative arithmetic, not actual checkpoint weights: source endpoints 0 and 4 map query 6 to 1.5; one linear and three sine channels give [1.5,0.7071,−1,0]. The actual model learns 100 time coordinates.',
'scm':'Seed 2 paper-grounded reconstruction: one nonlinear second-order graph produces correlated edge trajectories. Only causal relations 0 and 3 shift; the shown relation-1 edge remains fixed. This is not the unavailable original pretraining sampler.',
'split':'Actual Electricity row counts for three predeclared cutoffs. ID rows are withheld inside each source week; all OOD weeks share the frozen historical context. These are new deterministic cutoffs, not a replay of original random split selection.',
'results':'Fresh full-model OOD log loss on three real tasks and the released synthetic Blobs task. Dots are joint cutoff/checkpoint repetitions; bars are sample SD, not independent-dataset confidence intervals.',
'horizon':'Actual per-domain OOD losses in repetition 0. Every line retains one historical context and checkpoint. Domain difficulty varies; adjacent points are not independent datasets.',
'boundary':'Original Blobs source, domains 0–3 as fixed context, checkpoint 1; actual predictions on a 36×36 query grid. Panels vary requested future domain only. Background is predicted class, points are class-labeled observations; optimized paper Figure 5 was not reproduced.',
'ranks':'Average repetitions within each real dataset, rank arms, then average three dataset ranks. Exploratory Friedman/Nemenyi checks show no resolved pairwise difference; this neither proves equivalence nor reproduces the paper’s Wilcoxon-Holm diagram.'}
def source(name):
 s=(ROOT/'relkit/driftpfn_l068_v2.py').read_text()
 for n in ast.parse(s).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name:return ast.get_source_segment(s,n)
 raise KeyError(name)
def parts():return re.split(r'(<!--figure:\w+-->|<!--results-table-->|<!--analysis-text-->)',(ROOT.parent/'lessons/content'/(SLUG+'.md')).read_text())
def table_markdown():
 a=json.loads((ROOT/'_analysis_l068_v2_results.json').read_text());lines=['| Dataset | Arm | OOD accuracy ± SD | OOD AUC ± SD | OOD log loss ± SD |','|---|---|---:|---:|---:|']
 for r in a['summary']:
  if r['split']=='ood':lines.append('| '+r['dataset']+' | '+r['arm']+' | '+' | '.join(f"{r[m+'_mean']:.4f} ± {r[m+'_sd']:.4f}" for m in ['accuracy','auc','log_loss'])+' |')
 return '\n'.join(lines)
def analysis_text():return json.loads((ROOT/'_analysis_l068_v2_results.json').read_text())['interpretation']
def figure(name,notebook=False):
 p='figures/l068/'+name+'-v2.png';caption=CAPTIONS[name];minimum=1150 if name in ['results','boundary'] else 900 if name in ['architecture','scm','horizon'] else 800
 if notebook:return '<div style="max-width:100%;overflow-x:auto" role="region" tabindex="0" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/p).read_bytes()).decode()+'" style="width:100%;min-width:'+str(minimum)+'px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+p+').'
 return '<figure id="l068-'+('architecture' if name=='architecture' else 'figure-'+name)+'"><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img style="min-width:'+str(minimum)+'px" src="../labs/'+p+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+p+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 p='../' if prepared else '../labs/';links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(p+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(p+'_verify_l068_v2_results.json','Measured evidence'),(p+'l068-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 068 · five live operations + complete released checkpoints + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'">'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Run the notebook in Colab or Jupyter. Built with TabPFN.</p></aside>'
def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 068 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Evidence](_verify_l068_v2_results.json) · [Reproduction](l068-reproduction.md)

**Built with TabPFN** · [License](sources/l068-v2/LICENSE.txt). Your goal is to implement five live operations and measure a future-domain comparison using every original checkpoint tensor. The model and generation algorithm are visible below. Default lab: full Electricity and original three-class Blobs datasets, first predeclared temporal cutoff, original base/drift checkpoint 1 and the separately pretrained NoT2V checkpoint 1. The author-reference table additionally covers Parking/Chess and three cutoffs/checkpoints. It is clearly separate from your current-kernel output. Downloads reuse seven original ~29 MB files as needed; default needs three. CPU is sufficient. Original pretraining and the optimized 18-task paper benchmark are NOT_RUN; this diagnostic is INCOMPARABLE to paper-result reproduction.

**Workflow:** retrieve and predict first, read the connected explanation, implement TODOs, run CHECKs, execute actual checkpoint predictions, and submit the EXIT JSON plus a written diagnosis. This notebook is self-contained; the following recap and figures do not require reopening the lesson.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:md(figure(m[1],True))
  elif p=='<!--results-table-->':md('### Author-reference evidence · not current-kernel output\n\n'+table_markdown())
  elif p=='<!--analysis-text-->':md(analysis_text())
  elif p.strip():md(p.replace('../labs/',''))
 md('## PROVIDED · The live implementation begins here\n\nDefinitions below execute in your namespace. Model construction, all checkpoint loading, every prediction, the sparse prior diagnostic and summaries call these functions. The independent original implementation is used only to verify correspondence. The original data-loader functions retain attribution and are adapted only to the local collector/path interface.')
 code('''# PROVIDED — runtime imports and a fresh destination
import copy,hashlib,importlib.metadata,inspect,json,math,os,time,types,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score
from IPython.display import display
for directory in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (directory/'relkit').is_dir():ROOT=directory.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start Jupyter in labs')
torch.set_num_threads(1)
output=Path(os.environ.get('L068_OUTPUT',str(ROOT/'data/cache/l068-student')))
output.mkdir(parents=True,exist_ok=True)''')
 from relkit import driftpfn_l068_v2 as core
 code('# PROVIDED — immutable source/data/model contracts\n'+'\n'.join(k+'='+repr(getattr(core,k)) for k in ['COMMIT','BASE_URL','MODEL_CONFIG','CHECKPOINTS','PROTOCOL','PRESETS68','DATA_SHA','TASK_TYPE_MULTICLASS']))
 checks={
 'normalize_time':"c=torch.tensor([0.,2.,4.,6.,40.])[:,None,None]\ntorch.testing.assert_close(normalize_time(c,3).flatten(),torch.tensor([0.,.5,1.,1.5,6.]))\ntorch.testing.assert_close(normalize_time(torch.tensor([3.,3.,4.,100.])[:,None,None],2).flatten(),torch.tensor([0.,0.,1.,6.]))\nassert normalize_time(c,3)[3]>1,'Do not clip ordinary extrapolation to1'",
 'time2vec':"c=torch.tensor([1.5])[:,None,None];w=torch.tensor([[1.],[math.pi/2],[math.pi],[2*math.pi]]);b=torch.zeros(4)\ntorch.testing.assert_close(time2vec(c,w,b).flatten(),torch.tensor([1.5,2**-.5,-1.,0.]),atol=1e-6,rtol=0)\nassert time2vec(torch.zeros_like(c),w,torch.tensor([2.,1.,0.,0.]))[0,0,0]==2,'Affine bias is part of the encoding'",
 'shifted_weights':"base=np.array([1.,2.,3.,4.]);mapping=np.array([0,0,1,-1]);shifts=np.array([[.1,.2,.3,.4],[.5,.6,.7,.8]])\nnp.testing.assert_allclose(shifted_weights(base,mapping,[0],shifts),[[1.1,2.2,3,4],[1.5,2.6,3,4]])\nnp.testing.assert_array_equal(base,[1,2,3,4])",
 'temporal_split':"c=np.repeat([0,2,5,9],20);s=temporal_split(c,2,3,12)\nassert {k:len(v) for k,v in s.items()}=={'train':22,'id':2,'ood':24}\nassert c[s['train']].max()<c[s['ood']].min()\nassert not set(s['train'])&set(s['id'])\nassert all(np.array_equal(v,temporal_split(c,2,3,12)[k]) for k,v in s.items()),'Fixed RNG, stable identities'",
 'dataset_summary':"rows=[dict(dataset=d,arm='drift',split='ood',accuracy=a,auc=u,log_loss=l) for d,a,u,l in [('A',.4,None,1.),('A',.8,.7,.5),('B',.9,.9,.2)]]\ns=dataset_summary(rows);a=next(v for v in s if v['dataset']=='A');assert abs(a['accuracy_mean']-.6)<1e-12 and abs(a['accuracy_sd']-2**.5*.2)<1e-12\nassert a['auc_n']==1 and a['auc_mean']==.7 and len(s)==2,'Do not make missing AUC zero or pool datasets'"}
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint:** {hint}')
  code('# TODO — '+name+'\n'+(source(name) if solution else sig+'\n    raise NotImplementedError("Implement this live operation")'))
  code('# CHECK — '+name+'\n'+checks[name]+'\nprint("CHECK passed: '+name+'")')
 chunks=[
 ('Feature groups and attention',['group_features','attention_mix','PackedAttention','row_attention','postnorm_update','V2Block'],'Trace both axes: feature attention mixes G+1 tokens inside one row; row attention uses context keys only. Query heads share the first key/value projection. The active original SDPA scale is exactly 1/sqrt(32). Residual addition precedes normalization.'),
 ('Numeric and target preprocessing',['numeric_groups','target_channels'],'Packing occurs inside each two-feature group before imputation and source sample standardization. Constant detection uses supplied query features as in the release, but normalization moments use only context. Targets contain query missing flag −2 and the source-imputed ordinal value.'),
 ('The complete pretrained network',['DriftPFN','ensure_file','load_pretrained'],'Read forward from raw matrices to 10 logits. The 102-column encoder confirms repeated time concatenation. Strict one-to-one checkpoint mapping loads every tensor; no random layer replaces a pretrained layer.'),
 ('Original real-data transformations',['dataframe_to_distribution_shift_ds','get_electricity_data','get_parking_birmingham_data'],'These attributed original transformations produce the measured task definitions. Electricity is weekly; Parking keeps one car park and quartile labels. Our collector preserves ordinal category codes and processed row order. Every resulting array is independently source-checked.'),
 ('Chronology and the released synthetic task',['get_chess_data','get_intersecting_blobs','load_dataset'],'Chess uses twenty-game groups after date sorting. Blobs has three classes; its published binary description is an internal typo. We preserve the original generator seed and movement/noise recipe.'),
 ('The visible second-order SCM reconstruction',['sample_scm'],'Follow twelve edge IDs through the selected relation mask into eight scalar nodes. The shared H parameters persist across domains. Your shifted_weights determines actual generated X and y, then the full model predicts their held-out future domains. This sampler is explicitly a reconstruction, not original pretraining.'),
 ('Identity of what actually executes',['model_digest','stable_value','model_runtime_identity','kernel_identity'],'The identity walks nested generator bytecode, its global helpers, numeric and function-valued defaults. It also binds real model settings and weights and rejects hooks or instance overrides. Source-check identities and notebook identities remain separate.'),
 ('Prediction and the declared experiment',['predict','metrics','run_experiment'],'The run holds context fixed across future domains and uses the same query IDs across five arms. Training labels alone enter model calls. The result saves probabilities, per-domain metrics, original row IDs, source digests and live identities. NoT2V always uses its available checkpoint 1; the other two variants use three paired checkpoints in the full panel.')]
 for title,names,why in chunks:
  md('## PROVIDED · '+title+'\n\n'+why);code('# PROVIDED — '+title+'\n\n'+'\n\n'.join(source(n) for n in names))
 md('## CHECK · The actual current model agrees with original source\n\nThe checker downloads hash-pinned source into an ignored cache and runs the original complete network. It compares all three live variants with original checkpoint weights on a constant/missing-feature and extrapolated-time fixture. This certifies these live definitions, not a packaged replacement for your TODOs.')
 code('''# CHECK — bind current namespace to independent original inference
from _paper_audit_l068_v2 import check_live_source
parity=check_live_source(globals(),ROOT)
assert parity['kernel_identity']['sha256']==kernel_identity(globals(),ROOT)['sha256']
display(pd.DataFrame(parity['models'])[['variant','max_logit_error']])
print('Current namespace identity',parity['kernel_identity']['sha256'])''')
 md('## CHECK · Changing a hidden generator helper must invalidate evidence\n\nThe original notebook identity must survive an unchanged recheck, and changing a helper called inside a generator expression must change it. We restore the actual implementation before measurement.')
 code('''# CHECK — nested generator dependencies and live sparse prior
saved=sample_scm;before=kernel_identity(globals(),ROOT)['sha256']
def nested68(value):return value+1
def sample_scm(seed=0):return sum(nested68(i) for i in [1,2])
changed=kernel_identity(globals(),ROOT)['sha256']
def nested68(value):return value+2
assert kernel_identity(globals(),ROOT)['sha256']!=changed
sample_scm=saved
assert kernel_identity(globals(),ROOT)['sha256']==before
scm=sample_scm(2);w=np.array(scm['trace']['weights']);mask=np.isin(scm['trace']['edge_to_relation'],scm['trace']['selected'])
assert np.all(w[:,~mask]==np.array(scm['trace']['base'])[~mask])
display(pd.DataFrame(w[:,[0,1,2,8,9]],columns=['U→X edge0','U→X edge1','fixed V→X','X→Y edge8','X→Y edge9']))''')
 md('## Run · Predict the exceptions, then measure\n\nWrite down a predicted benefit and one plausible failure before running. This measures full Electricity and Blobs datasets with the first declared cutoff and all five arms. It also measures the live multi-node reconstruction. The author panel’s stronger synthetic result does not guarantee a real-task benefit.')
 code('''# PROVIDED — fresh current-kernel experiment
assert parity['kernel_identity']['sha256']==kernel_identity(globals(),ROOT)['sha256']
result=run_experiment(ROOT,PRESETS68['lab'],globals())
display(pd.DataFrame(result['summary'])[['dataset','arm','split','accuracy_mean','auc_mean','log_loss_mean']])''')
 md('## CHECK · Same rows and probabilities, separate identities\n\nCompare your fresh output with the matching author repetition; preserve the distinction between module and notebook code identities. This is a numerical comparison, not a reassignment of the author’s hash to your kernel.')
 code('''# CHECK — paired author-reference predictions
reference=json.loads((ROOT/'_verify_l068_v2_results.json').read_text())
maximum=0.
for row in result['records']:
    author=next(a for a in reference['records'] if (a['dataset'],a['seed'],a['arm'],a['split'])==(row['dataset'],row['seed'],row['arm'],row['split']))
    assert row['context_ids']==author['context_ids'] and row['query_ids']==author['query_ids']
    error=float(np.max(np.abs(np.array(row['probabilities'])-author['probabilities'])));maximum=max(maximum,error)
    assert error<3e-5
assert result['kernel_identity']['sha256']==parity['kernel_identity']['sha256']
print('Maximum probability difference',maximum)
print('Your identity',result['kernel_identity']['sha256'])
print('Author module identity',reference['kernel_identity']['sha256'])''')
 md('## EXIT TICKET · A measured explanation\n\nExplain one real-task failure, an accuracy/AUC disagreement, the difference between time sensitivity and NoT2V pretraining, and how sparse relationships changed several functional edges in your generated task. Name the variance unit and original work still unrun. The word-count check only requests a substantive attempt; the tutor assesses the reasoning.')
 interpretation='The full author panel shows a Chess failure: drift mean OOD accuracy 0.6762 and loss 0.7721 are worse than base-with-time 0.7158 and 0.7033. Electricity has a metric disagreement: drift accuracy improves slightly but AUC decreases, so ranking and decision accuracy should not be conflated. On Parking, the same drift checkpoint improves when timestamps are zeroed. That is an inference sensitivity result, not the separately pretrained NoT2V ablation. My live second-order SCM uses one shared nonlinear H across domains, masking causal relations 0 and 3 into functional edges 0, 1, 8, 9; fixed V-to-X weights remain unchanged. Repetitions jointly vary cutoff and checkpoint and are not independent datasets. Three real datasets receive equal weight only after within-dataset summaries. Original 30.7-million-task pretraining, exact original prior sampling, the optimized 32-view wrapper and full 18-task paper benchmark remain unrun. I would next hold checkpoint and future test rows fixed while varying source history length.'
 code('# TODO — written interpretation\ninterpretation='+repr(interpretation if solution else '')+'\nassert len(interpretation.split())>=60,"Explain actual evidence, the mechanism and limits"')
 code('''# EXIT — certify current code and source; fresh artifact only
assert result['kernel_identity']['sha256']==parity['kernel_identity']['sha256']==kernel_identity(globals(),ROOT)['sha256']
assert parity['checker_sha256']==hashlib.sha256((ROOT/'_paper_audit_l068_v2.py').read_bytes()).hexdigest()
assert parity['source_manifest_sha256']==hashlib.sha256((ROOT/'_sources_l068_v2.json').read_bytes()).hexdigest()
result['source_check']=parity;result['interpretation']=interpretation
path=output/'exit-v2.json'
with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
print('EXIT saved',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
print('Paper results:',result['paper_reproduction'])''')
 md('## NEXT STEP · All tasks and original paper route\n\nThe gate below uses these live definitions for the full four-task, three-cutoff/checkpoint panel, retaining all rows. It reproduces the declared local experiment, not the original optimized 18-task study. The reproduction document describes the original source evaluator, preprocessing and remaining unavailable prior generator. No paid/cloud action is needed for this supported CPU track.')
 code('''# PROVIDED — optional full live panel, OFF by default
RUN_BROADER=False
if RUN_BROADER:
    assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_identity']['sha256']
    broader=run_experiment(ROOT,PRESETS68['full_local'],globals())
    with (output/f'full-local-{time.time_ns()}.json').open('x') as stream:json.dump(broader,stream,indent=2,allow_nan=False)
else:print('Broader current-kernel run NOT_RUN; author full local panel is separately measured; paper benchmark NOT_RUN.')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:v2:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb');path.parent.mkdir(exist_ok=True)
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
   if old.metadata.get('execution_verification'):nb.metadata['execution_verification']=old.metadata['execution_verification']
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for t in soup.select('[id]'):t['id']=unquote(t['id'])
 for t in soup.select('[href],[src]'):
  key='href' if t.has_attr('href') else 'src';u=urlsplit(t[key])
  if not u.scheme and not u.netloc and u.path:t[key]='../'+t[key]
  elif key=='href' and t[key].startswith('#'):t[key]=unquote(t[key])
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'style':'max-width:100%;overflow-x:auto','role':'region','tabindex':'0','aria-label':'Scrollable evidence table'}))
 style=soup.new_tag('style');style.string='th,td{padding:.6rem .8rem!important;white-space:nowrap;word-break:normal;overflow-wrap:normal}';soup.head.append(style)
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'));soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def build_package(notebooks=True,render=True):
 body=''
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:
   body+=figure(m[1])
   if m[1] in ['time','scm']:body+='<div class="l068-widget" id="l068-'+m[1]+'-viz"><p>Interactive controls need JavaScript. The static figure preserves a worked example.</p></div>'
  elif p=='<!--results-table-->':body+=markdown2html_mistune(table_markdown())
  elif p=='<!--analysis-text-->':body+=markdown2html_mistune(analysis_text())
  else:body+=markdown2html_mistune(p)
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 068 · '+TITLE+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','foundation-course','lab-access','l068-drift'])+'</head><body class="l068"><article>'
 scripts=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','l068-drift-viz'])
 page=head+'<nav><a href="../index.html">Course</a> · <a href="0067-local-pfn-retrieval-finetuning.html">← Lesson 067</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 068</p><h1>'+TITLE+'</h1>'+launcher()+'<h2>Retrieve before reading</h2><div id="warmup"></div><div id="prediction"></div>'+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>'
 soup=BeautifulSoup(page,'html.parser')
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(str(soup))
 ref=BeautifulSoup(head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 068</a></nav><h1>Temporal PFN: computation and evidence card</h1>'+launcher()+figure('architecture')+markdown2html_mistune((ROOT/'l068-reference.md').read_text())+markdown2html_mistune(table_markdown())+'</article></body></html>','html.parser')
 for t in ref.find_all('table'):t.wrap(ref.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(str(ref))
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built complete L068 checkpoint package')
if __name__=='__main__':build_package()
