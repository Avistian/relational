"""Canonical scoped L061 v2: visible original-style GP PFN and five live tasks."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _lesson_depth import enrich_html,enrich_notebook
from _check_l061_v2 import CHECKS
ROOT=Path(__file__).resolve().parent
SLUG='0061-prior-data-fitted-networks';TITLE='Learn Bayesian prediction from sampled tasks'
TASKS=[
('sample_gp',"def sample_gp(batch,total,features,generator,lengthscale=.6,noise=1e-4,device='cpu'):",'Draw one shared function','Return x:B×N×F and y:B×N. Sample uniform features, construct the noisy RBF covariance, factor in float64 and multiply a standard normal vector; return float32 on device. Reject nonpositive sizes/noise.','The off-diagonal covariance is the information the context should reveal. Independent row targets remove it.','rbf_kernel is provided. Keep all rows of a task in the same joint factorization and use the passed generator for both draws.'),
('gp_posterior','def gp_posterior(x,context_y,n,lengthscale=.6,noise=1e-4):','Condition without an inverse','Return float64 mean and marginal variance of observed query targets. Handle empty context and reject incompatible shapes. Use a Cholesky solve; add observation noise only to the appropriate diagonal blocks.','The independently computable oracle distinguishes an incorrect model from a poor approximation.','Separate x into context/query blocks. The squared norm of a triangular solve gives the diagonal uncertainty reduction.'),
('context_attention_mask',"def context_attention_mask(total,n,device='cpu'):",'Enforce the information boundary','Return a total×total additive attention mask: zero where allowed, negative infinity elsewhere. Follow the original source including query self edges; reject n outside0…total−1.','A wrong mask can leak across queries through an intermediate context token in later layers.','Think in sending columns. Preserve the diagonal even when context is empty.'),
('attention_mix','def attention_mix(q,k,v,mask):','Perform one attention aggregation','For B×H×N×Dh projections, compute scaled dot products, masked row softmax and value aggregation. Return B×H×N×Dh.','Softmax across receiving rows would normalize the wrong conditional weights while still producing plausible shapes.','The final key dimension is the sender axis. Scaling uses one head’s width, not the full model width.'),
('riemann_nll','def riemann_nll(logits,y,borders):','Score a full-support density','Return elementwise NLL matching y. Validate shapes and increasing borders, identify bins, combine log-softmax masses with interior log widths and normalized half-normal tails.','A probability mass is not a density. Gaussian targets outside nominal borders must remain possible.','Use stable log-softmax and searchsorted. The two tail components replace the uniform components; they do not add extra mixture mass.')]
FIGURES={
 'conditioning':('conditioning-v2.png','Analytic one-observation fixture: context (.2, 1), length scale .6. Query .8 gives mean .606470 and noisy-target variance .632257. This is an exact GP computation, not a trained prediction.'),
 'density':('density-v2.png','Synthetic full-support Riemann fixture. The interior target 1 has density .15 and NLL 1.89712. The first and last softmax masses multiply normalized half-normal tails.'),
 'results':('results-v2.png','New author reference: three training-seed points, shared 256 test tasks per context/regime. Exact GP, marginal prior and analytic masses in the same fixed head are diagnostic oracles. Both panels use the same y scale. No paper result reproduction.'),
 'predictions':('predictions-v2.png','Fresh trained seed 2, two chosen observations (.2,1) and(.8,−1). Central 95% predictive intervals are distribution intervals, not confidence intervals for a mean metric. Lower panel traces density at query .5.')}

def piece(name):
 source=(ROOT/'relkit/pfn_l061_v2.py').read_text()
 return next(ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)
def parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/f'{SLUG}.md').read_text())
def figure(name,notebook=False):
 if name=='architecture':return ''
 filename,caption=FIGURES[name];path='figures/l061/'+filename
 if notebook:return f'<div style="overflow-x:auto;max-width:100%" role="region" aria-label="Scrollable computation figure"><img alt="{html.escape(caption)}" src="data:image/png;base64,{base64.b64encode((ROOT/path).read_bytes()).decode()}" style="width:760px;min-width:640px;max-width:none;height:auto"></div>\n\n{caption} [Open full-size figure]({path}).'
 return '<figure><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computation figure"><img src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+path+'">Open full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 prefix='../' if prepared else '../labs/'
 links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(prefix+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(prefix+'_verify_l061_v2_results.json','Measured evidence'),(prefix+'l061-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 061 · 5 live code tasks + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'"'+(' download' if title=='Download notebook' else '')+'>'+title+'</a>' for u,title in links)+'</nav><p>The preview is read-only. Complete the runnable notebook in Colab or Jupyter. Student functions are intentionally blank.</p></aside>'

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 061 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [New measured evidence](_verify_l061_v2_results.json) · [Primary sources](_sources_l061_v2.json) · [Reproduction contract](l061-reproduction.md)

**Mirror scope:** original row-token Transformer, query-self mask, postnorm blocks and full-support Riemann head; paper fixed-RBF-GP objective and prior. This is no longer the historical CountPFN-only lab. **Data tier C:** synthetic functions are the paper's selected mechanism study, not substitutes passed off as real table benchmarks. Five live TODOs, immediate CHECKs, independently validated source computation, a fresh CPU smoke, and an explanatory EXIT. The larger three-seed author run is reference evidence; your kernel has not run it yet. Full Figure4 reproduction remains INCOMPARABLE. No paid/cloud job is launched. Read and predict before executing.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for part in parts():
  match=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if match:
   if match[1]!='architecture':md(figure(match[1],True))
  elif part.strip():md(part.replace('../labs/',''))
 md('''## Start the live implementation
Every mathematical operation below resolves in this notebook's namespace. Provided classes call your mask and attention functions; the trainer calls your sampler and loss; evaluation calls your GP oracle. Do not import replacement implementations to make the CHECKs pass. Each visible chunk is extracted from the versioned teaching module when this notebook is built.''')
 code('''# PROVIDED — imports, path discovery and explicit local artifact folder
import base64,zlib,copy,hashlib,inspect,importlib.metadata,json,math,time,functools,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from IPython.display import display
for path in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (path/'relkit').is_dir():ROOT=path.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run the course bootstrap or start inside labs')
__file__=str(ROOT/'relkit/pfn_l061_v2.py')  # repository reference identity only
output=ROOT/'data/cache/l061-student';output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
VERSIONS={k:importlib.metadata.version(k) for k in ['numpy','torch','scipy']}
print(VERSIONS)''')
 md('''### PROVIDED · The signal covariance kernel
The last axis is features. Unsqueezing two row axes broadcasts each context location against each candidate location, yielding B×N×M squared distances. This is signal covariance; the sampler and oracle add noise to their own appropriate diagonal blocks.''')
 code('# PROVIDED — signal covariance\n'+piece('rbf_kernel'))
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece(name) if solution else sig+'\n    raise NotImplementedError("Implement this live PFN operation")'))
  code(CHECKS[name])
  if i==1:
   md('''### CHECK · Break the shared task deliberately
Predict the context/query covariance after replacing the query by a label from an independent GP task. This intervention preserves the marginal target distribution while removing shared-task information. The covariance check is not a trained negative-control PFN. Its analytic optimum would ignore context because the correct cross-covariance block is zero.''')
   code('''# CHECK — executed independent-task generator negative control
x_joint,y_joint=sample_gp(12000,2,1,torch.Generator().manual_seed(120))
_,y_other=sample_gp(12000,2,1,torch.Generator().manual_seed(121))
joint_cov=float((y_joint[:,0]*y_joint[:,1]).mean())
broken_cov=float((y_joint[:,0]*y_other[:,1]).mean())
expected_cov=float(rbf_kernel(x_joint,x_joint)[:,0,1].mean())
assert abs(joint_cov-expected_cov)<.04 and abs(broken_cov)<.04
negative_control={'joint_covariance':joint_cov,'independent_query_covariance':broken_cov,'expected_joint':expected_cov}
print(negative_control)''')
  if i==4:
   md('''### PROVIDED · Full row encoder and repeated postnorm blocks
Follow h:B×N×d through the qkv projection and head reshape. The concat after attention restores the original width; the two normalization calls occur after the additions. RowPFN constructs context tokens with actual labels and query tokens with features only. The head returns logits for query rows only. Both classes below use your live functions, including in copied-weight checks.''')
   code('# PROVIDED — complete attention/FFN block\n'+piece('PFNBlock'))
   code('# PROVIDED — end-to-end row model\n'+piece('RowPFN'))
 md('''## PROVIDED · From logits to means and central coverage
make_borders uses an independent marginal-prior sample with seed 731. It never fits a boundary using held-out query labels. riemann_mean accounts for the means of the two half-normal tails; riemann_cdf integrates each component before weighting. A plain midpoint average would use the wrong tail means. Read these functions even though they are provided.''')
 for name in ['make_borders','riemann_mean','riemann_cdf']:code('# PROVIDED — '+name+'\n'+piece(name))
 md('''## CHECK · Validate the live model against the original implementation
The validation harness downloads pinned source files into an isolated cache; it does not install a model package. It copies nonzero weights into the model you just defined, checks float32 and float64 outputs, mapped parameter/input gradients, target-density values/gradients, context permutation and query independence. A same-valued float64 mask is passed explicitly to the reference in float64 tests; production float32 also uses the original implicit mask. This validates the named computation, not original optimizer trajectories or benchmark scores.''')
 code('''# CHECK — reference library/source is a validation point; your live definitions are the subject
from _check_l061_v2 import check as validate_live_pfn
parity=validate_live_pfn(globals(),save=False)
print({key:value for key,value in parity.items() if key!='official_sources'})''')
 md('''## PROVIDED · Fresh-task pretraining and paired evaluation
The trainer contains no call to gp_posterior: it sees sampled labels only. The evaluator freezes the network, then uses your oracle for each matched or shifted task. The task batches, targets, logits and analytic moments are preserved in compressed arrays; this keeps independent reconstruction possible. Each selected evaluation context has one query and the same task stream across weight seeds.''')
 code('# PROVIDED — exact declared budgets\nPRESETS='+repr(__import__('relkit.pfn_l061_v2',fromlist=['PRESETS']).PRESETS))
 for name in ['train_pfn','encode_tensor','evaluate_pfn','summarize_records','state_hash','run_experiment']:
  code('# PROVIDED — '+name+'\n'+piece(name))
 names=[node.name for node in ast.parse((ROOT/'relkit/pfn_l061_v2.py').read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
 md('''### PROVIDED · Capture the code actually executed
A repository hash labels author reference code. This kernel can contain edited functions and class methods that differ from it. The identity below records all visible sampler/model/density/evaluator/helper source texts, including methods and PRESETS, plus library versions. There is no checkpoint resume; an existing run filename is rejected. The mutation check changes both a helper and a class method, then restores them.''')
 code('# PROVIDED — actual kernel source identity\nKERNEL_NAMES='+repr(names)+'''
def normalize_default(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,(list,tuple)):return [normalize_default(v) for v in value]
    if isinstance(value,dict):return {str(k):normalize_default(v) for k,v in value.items()}
    if inspect.isfunction(value):return {'function_source':inspect.getsource(value),'defaults':normalize_default(value.__defaults__),'kwdefaults':normalize_default(value.__kwdefaults__)}
    raise TypeError('Unsupported live default; record it explicitly rather than using an unstable repr')

def capture_kernel_identity():
    sources={};defaults={}
    for name in KERNEL_NAMES:
        obj=inspect.unwrap(globals()[name])
        if inspect.isclass(obj):
            sources[name]={method:inspect.getsource(value) for method,value in vars(obj).items() if inspect.isfunction(value)}
            defaults[name]={method:{'defaults':normalize_default(value.__defaults__),'kwdefaults':normalize_default(value.__kwdefaults__)} for method,value in vars(obj).items() if inspect.isfunction(value)}
        else:
            sources[name]=inspect.getsource(obj)
            defaults[name]={'defaults':normalize_default(obj.__defaults__),'kwdefaults':normalize_default(obj.__kwdefaults__)}
    content={'sources':sources,'defaults':defaults,'default_normalizer_source':inspect.getsource(normalize_default),'presets':copy.deepcopy(PRESETS),'versions':dict(VERSIONS),'capture_source':inspect.getsource(capture_kernel_identity)}
    content['sha256']=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    return content

def attach_kernel_identity(result,path):
    result['repository_source_hashes_are_reference_only']=True
    result['kernel_identity']=capture_kernel_identity()
    Path(path).write_text(json.dumps(result,indent=2,allow_nan=False)+'\\n')
    return result
''')
 code('''# CHECK — source/default/class-method edits are visible; restore everything afterward
before=capture_kernel_identity()['sha256']
original_kernel=rbf_kernel
original_forward=RowPFN.forward
original_defaults=rbf_kernel.__defaults__
original_steps=PRESETS['smoke']['steps']
def altered_kernel(x,z,lengthscale=.6):
    return original_kernel(x,z,lengthscale)+.1
def altered_forward(self,x,context_y):
    return original_forward(self,x,context_y)+.1
try:
    rbf_kernel=altered_kernel
    assert capture_kernel_identity()['sha256']!=before
    rbf_kernel=original_kernel;rbf_kernel.__defaults__=(.7,)
    assert capture_kernel_identity()['sha256']!=before
    rbf_kernel.__defaults__=original_defaults;RowPFN.forward=altered_forward
    assert capture_kernel_identity()['sha256']!=before
    RowPFN.forward=original_forward;PRESETS['smoke']['steps']+=1
    assert capture_kernel_identity()['sha256']!=before
finally:
    rbf_kernel=original_kernel;rbf_kernel.__defaults__=original_defaults;RowPFN.forward=original_forward;PRESETS['smoke']['steps']=original_steps
assert capture_kernel_identity()['sha256']==before
print('Live helper, class method and preset changes detected and restored; no resume supported')''')
 md('''## Run the fresh smoke through all five tasks
Thirty updates test execution and wiring; they cannot establish a learned posterior approximation. The saved author panel uses 2000 updates per seed and is examined next. The wrapped call counters instrument the actual names used inside the provided model, trainer and evaluator. Only after restoring the wrappers do we capture the final source identity.''')
 code('''# PROVIDED — bounded fresh fit, actual live call counts, no result overwrite
names=['sample_gp','gp_posterior','context_attention_mask','attention_mix','riemann_nll']
originals={name:globals()[name] for name in names};calls={name:0 for name in names}
def counted(name,function):
    @functools.wraps(function)
    def wrapper(*args,**kwargs):
        calls[name]+=1
        return function(*args,**kwargs)
    return wrapper
for name,function in originals.items():globals()[name]=counted(name,function)
try:
    fresh_path=output/f'smoke-{time.time_ns()}.json'
    fresh=run_experiment('smoke',fresh_path)
finally:
    for name,function in originals.items():globals()[name]=function
assert all(value>0 for value in calls.values()),'Every live TODO must reach the fresh experiment'
attach_kernel_identity(fresh,fresh_path)
print('Actual calls:',calls)
display(pd.DataFrame([{key:r[key] for key in ['seed','regime','n_context','pfn_nll','gp_nll','excess_nll']} for r in fresh['records']]))''')
 md('''## Train your completed PFN, not just the smoke
Now fit a genuinely learned model through your live implementation: the lab recipe uses 2000 updates, but this student run uses seed 7 only. That is a fresh trained model, not the author seed 0–2 models. Its evaluation tasks are deliberately the same declared panel, making comparisons conditional on that panel; do not select a new recipe by repeatedly inspecting it. Expect roughly a minute or two on the author CPU, with runtime varying by host. The saved model-state hash, training trace, raw predictions and kernel source identify this run.''')
 code('''# PROVIDED — substantive default learning run; one explicit new training seed
student_config=copy.deepcopy(PRESETS['lab']);student_config['seeds']=[7]
student_model,student_borders,student_trace,student_seconds=train_pfn(student_config,7)
student_records=evaluate_pfn(student_model,student_borders,student_config,7)
learned={'status':'COMPLETE','operator':'pfn_l061_v2-student','config':student_config,
 'prior':fresh['prior'],'versions':VERSIONS,'seed':7,'training_seconds':student_seconds,
 'trace':student_trace,'weights_sha256':state_hash(student_model),'records':student_records,
 'summary':summarize_records(student_records),'borders':student_borders.tolist(),'paper_reproduction':'INCOMPARABLE'}
learned_path=output/f'fit-seed7-{time.time_ns()}.json'
attach_kernel_identity(learned,learned_path)
display(pd.DataFrame([{key:r[key] for key in ['seed','regime','n_context','pfn_nll','gp_nll','prior_nll','coverage95']} for r in student_records]))
print('Fresh learned PFN seed 7 saved:',learned_path,'Training seconds:',student_seconds)
print('One training seed: no training-seed uncertainty estimate. Original paper results not reproduced.')''')
 md('''## Read author reference evidence without calling it your result
The table below loads the three-seed measured panel. NLL is lower-is-better; coverage is a calibration diagnostic with a 95% target, not a higher-is-always-better score. Mean squared error compares predictive means to the analytic mean, not to noisy labels. The accompanying diagnostic supplies exact GP region masses to the same fixed head: it isolates head restrictions from learned mass errors, without training another network.''')
 code('''# PROVIDED — transparent author-reference tables, separate from fresh smoke
reference_path=ROOT/'_verify_l061_v2_results.json'
reference=json.loads(reference_path.read_text())
head_diagnostic=json.loads((ROOT/'_analysis_l061_v2_results.json').read_text())
assert head_diagnostic['evidence_sha256']==hashlib.sha256(reference_path.read_bytes()).hexdigest()
rows=[]
for row in reference['summary']:
    rows.append({'regime':row['regime'],'context':row['n_context'],
      'PFN mean':row['pfn_nll']['mean'],'seed SD':row['pfn_nll']['seed_sd'],
      'exact GP':row['gp_nll']['mean'],'excess':row['excess_nll']['mean'],
      'paired task ±95%':row['excess_nll']['conditional_task_t95'],
      'mean MSE':row['mean_squared_error']['mean'],'coverage95':row['coverage95']['mean']})
display(pd.DataFrame(rows).round(5))
display(pd.DataFrame(head_diagnostic['records']).round(5))
print('Training traces are sampled minibatch NLL, not monotonic validation curves.')
print('Reference operator SHA:',reference['repository_reference_sha256'])
print('Current live code SHA:',capture_kernel_identity()['sha256'])''')
 md('''## EXIT TICKET · State what the experiment establishes
In at least 120words, explain the shared-task generator, why self-query attention is legal, why probability masses need density conversion, and why the population objective does not certify this learned PFN. Report one matched and one shifted result with their context sizes. Distinguish the fixed-head oracle from the exact GP and the trained network. Distinguish your fresh 30-step smoke, your fresh 2000-step seed 7 fit, and the author three-seed result. Save the live code identity and the three verdicts. Submit the resulting JSON and your explanation to the teacher.''')
 explanation=('The context and query in the correct sampler share one Gaussian process draw. Their covariance makes a context informative about a query; drawing an independent task for the query removes that information. Query self attention uses only its feature token, so it does not expose a hidden target. The Riemann head predicts probability masses which must be divided by bin widths in interior regions; its normalized tail components retain nonzero density outside nominal borders. The population likelihood targets the posterior predictive distribution in expected forward KL, but finite optimization and fixed bins leave approximation error. In the author matched context16 panel, the learned NLL is about minus1.019 versus exactGP minus3.053; analytic masses in the same head give about minus2.276. Shifted context16 has NLL4.493 and coverage about39percent. My fresh 30-step result only checks execution; my separately saved2000-step seed 7 fit actually learns conditioning through the completed functions. Its one-seed evaluation cannot estimate training-seed uncertainty. Original large Figure4 training and benchmark outcomes were not reproduced.')
 code('# EXIT — replace with your own explanation\nlocal_verdict='+repr('OBSERVED_CONDITIONING_WITH_GAPS' if solution else 'WRITE YOUR OBSERVED VERDICT')+'\ninterpretation='+repr(explanation if solution else 'WRITE YOUR OWN EXPLANATION HERE')+'''
assert 'WRITE' not in local_verdict,'Supply a verdict supported by your actual seed 7 metrics'
assert len(interpretation.split())>=120,'Explain the mechanism, measured gaps and evidence boundary in at least 120 words'
exit_report={'lesson':61,'operator_validation':parity,'live_call_counts':calls,'generator_negative_control':negative_control,
 'fresh_smoke':str(fresh_path),'fresh_learned':str(learned_path),'student_summary':learned['summary'],'student_config':student_config,'author_reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
 'author_summary':rows,'kernel_identity':capture_kernel_identity(),'interpretation':interpretation,
 'verdicts':{'operator':'PASS','local_approximation':local_verdict,'author_reference':'USEFUL_CONDITIONING_WITH_SUBSTANTIAL_GAPS','paper_results':'INCOMPARABLE'}}
exit_path=output/'exit.json'
if exit_path.exists():exit_path=output/f'exit-{time.time_ns()}.json'
exit_path.write_text(json.dumps(exit_report,indent=2,allow_nan=False)+'\\n')
print('Saved EXIT:',exit_path)
print(exit_report['verdicts'])''')
 md('''## NEXT STEP · Train closer to the selected paper study
This required follow-up is an explicit runnable track with the same visible implementation. First run `lab` to reproduce the local three-seed result. Then `closer` uses five features, 6 blocks,128 width,256 bins,65 total rows,10000 updates and 1000 held-out tasks per context. The paper's released large-GP setup is much bigger and has recipe ambiguities documented in the reproduction contract. `paper` raises instead of silently claiming fidelity. The gate below is off; attach a suitable GPU before explicitly enabling the larger run. The Modal wrapper is separately supplied and has not been run. A full original-paper claim requires protocol alignment and a fresh result comparison, not just finishing this cell.''')
 code('''# PROVIDED — explicit post-EXIT Colab scale-up; no automatic cloud work
RUN_PAPER_REPRO=False
REPRO_PRESET='closer'
if RUN_PAPER_REPRO:
    repro_device='cuda' if torch.cuda.is_available() else 'cpu'
    larger_path=output/f'{REPRO_PRESET}-{time.time_ns()}.json'
    larger=run_experiment(REPRO_PRESET,larger_path,repro_device)
    attach_kernel_identity(larger,larger_path)
    print('Verified here: this explicit GP run. Paper claim: cited. Scale-up: INCOMPARABLE to original Figure4.')
else:print('Scale-up NOT_RUN in this kernel; original Figure4 reproduction remains INCOMPARABLE.')''')
 nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}}),61)
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
 result=json.loads((ROOT/'_verify_l061_v2_results.json').read_text())
 heads={(r['regime'],r['n_context']):r['head_oracle_nll'] for r in json.loads((ROOT/'_analysis_l061_v2_results.json').read_text())['records']}
 output='<h3>Measured author panel</h3><p>PFN is mean ± training-seed SD. Excess is PFN−exact GP with a paired-task 95% t interval conditional on the three trained models. Head oracle uses analytic region masses; it is not a trained model.</p><div class="table-scroll" role="region" aria-label="Scrollable measured PFN results" tabindex="0"><table><thead><tr><th>Prior</th><th>Context</th><th>PFN NLL</th><th>GP NLL</th><th>Head oracle</th><th>Excess ±95%</th><th>Coverage</th></tr></thead><tbody>'
 for row in result['summary']:
  output+='<tr>'+''.join('<td>'+value+'</td>' for value in [row['regime'].replace('_',' '),str(row['n_context']),f"{row['pfn_nll']['mean']:.3f} ± {row['pfn_nll']['seed_sd']:.3f}",f"{row['gp_nll']['mean']:.3f}",f"{heads[(row['regime'],row['n_context'])]:.3f}",f"{row['excess_nll']['mean']:.3f} ± {row['excess_nll']['conditional_task_t95']:.3f}",f"{row['coverage95']['mean']:.1%}"] )+'</tr>'
 return output+'</tbody></table></div>'

def build_package(notebooks=True,render=True):
 body=''
 for part in parts():
  match=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if match:
   name=match[1];body+=('<div id="gp-conditioning-viz" class="pfn-widget"><p>Interactive query control requires JavaScript; the worked example and figure below retain its default calculation.</p></div>' if name=='conditioning' else '')+figure(name)
   if name=='results':body+=evidence_table()
  else:body+=markdown2html_mistune(part)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 061 · {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','foundation-course','lab-access','l061-pfn'])+'</head><body class="l061"><article>'
 opening='<nav><a href="../index.html">Course</a> · <a href="0060-broad-model-comparison.html">← Lesson 060</a> · <a href="0062-tabpfn-v1.html">Lesson 062 →</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 061</p><h1>'+TITLE+'</h1>'+launcher()+'<section id="retrieval"><h2>Retrieve before reading</h2><p>Why must validation and test labels have separate roles? What changes when labels enter a forward pass instead of only a loss?</p><div id="warmup"></div></section><div id="prediction"></div>'
 cfg=dict(lesson=61,mode='pfn',answer='Shared synthetic tasks make context informative. Expected held-out NLL targets the PPD, but learned weights, finite density bins and prior mismatch limit the result.',quiz=['Can a query attend to itself without target leakage?',['Yes, if its target is never encoded','No, every self edge reveals a target','Yes, if its target is shuffled first'],0,'The original query token contains features only. Query self edges are legal; other query edges remain blocked at every layer.'])
 scripts='<script id="foundation-config" type="application/json">'+json.dumps(cfg)+'</script>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','foundation-lesson','l061-pfn-viz'])
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(head+opening+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>')
 ref=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 061</a></nav><h1>PFN computational reference</h1>'+launcher()+figure('conditioning')+markdown2html_mistune('''## Five live operations

''')+markdown2html_mistune('\n'.join('- `'+task[1]+'` — '+task[3] for task in TASKS))+markdown2html_mistune('''

## Audit boundaries

Original row Transformer computation and full-support density are validated with copied nonzero weights and gradients. The local GP is one-dimensional with width 64,3 blocks,64 bins and2000 updates per seed. It learns useful conditioning but does not match the exact posterior. A separate oracle puts analytic GP masses into the same fixed head, revealing finite-head approximation error.

[Measured predictions and raw task arrays](../labs/_verify_l061_v2_results.json) · [Fixed-head diagnostic](../labs/_analysis_l061_v2_results.json) · [Validation evidence](../labs/_check_l061_v2_results.json) · [Source inventory](../labs/_sources_l061_v2.json) · [Reproduction commands and exact gaps](../labs/l061-reproduction.md).

[Paper v7 §§3–5.1, Appendices A–F](https://arxiv.org/html/2112.10510v7) · [Original source commit 9c20031](https://github.com/automl/TransformersCanDoBayesianInference/tree/9c20031b355923bdd456d5fcfe4e98092b016b97) · [Gaussian conditioning derivation](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(ref+'</article></body></html>');enrich_html(61)
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L061 v2 package')
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--keep-notebooks',action='store_true');args=parser.parse_args();build_package(notebooks=not args.keep_notebooks)
