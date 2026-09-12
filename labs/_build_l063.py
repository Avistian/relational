"""Scoped canonical L063 historical-generator lesson and portable live lab."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _check_l063_v2 import CHECKS
ROOT=Path(__file__).resolve().parent;SLUG='0063-synthetic-scm-prior';TITLE='Build the prior that teaches TabPFN what a table can be'
TASKS=[
('sparse_weights','def sparse_weights(raw, mask, dropout, first=False):','Sample edges that stay fixed across rows','Return a sparse weight matrix for the released independent-edge branch. raw includes init_std; mask is binary. Preserve the source scaling and special first-layer behavior. Reject invalid shapes/probabilities.','An ordinary dropout rescaling silently changes the prior, even if the graph picture looks identical.','Separate first=True before the sparsification expression. Read the source denominator carefully; the block operator is separately provided.'),
('propagate','def propagate(causes, weights, biases, noises, activation):','Execute a sampled world','Return a list of all layer outputs. Implement the first affine step and then repeated activation→affine→noise using fixed matrices/noise arrays. Validate shapes and reject nonfinite generated values.','Both where noise enters and which values feed the next layer define the generated distribution. A correct output shape does not establish correctness.','activation_value is provided. Matrix axes are child,parent. Vectorize across rows; keep graph layers sequential.'),
('select_nodes','def select_nodes(values, feature_nodes, target_node):','Choose what the learner can observe','Return feature matrix and continuous target from the selectable node pool. Preserve requested column order. Reject duplicate, out-of-range or target-overlapping feature indices.','An observed target copy creates a trivial label leak; hidden versus observed roles are central to SCM task diversity.','The caller constructs the correct selectable pool. This function enforces its observation contract, not graph discovery.'),
('rank_labels','def rank_labels(target, bound_indices, permutation):','Turn one real target into named classes','Return discrete labels and sampled bound values. Count strict exceedances of bounds sampled by row index; then apply one class permutation to all rows. Retain empty intervals from duplicate bounds.','Class balance and ordering are part of the prior. A median threshold or query-specific relabeling changes the supervised task.','A target equal to a bound stays in the lower interval. Bounds need not be sorted to count how many are exceeded.'),
('posterior_weights','def posterior_weights(log_prior, log_context, log_query_x):','Condition on all observed evidence','Return normalized posterior weights for finite candidate worlds using logsumexp. Validate equal vector shapes and reject impossible observations.','SCMs model the feature distribution, so x_query can update beliefs about the world before its label is observed.','Add the three log-evidence terms. Subtract a shared log normalization before exponentiating; do not multiply tiny probabilities directly.')]
FIGURES={
'generator':'Trace the released generator: shared world parameters, row-level noise, intermediate node selection, joint class construction, and the context/query information boundary.',
'sparsity':'Fixed independent-edge fixture: raw matrix times binary mask divided by 1−sqrt(.25); contrast the separately explained blockwise rule.',
'roles':'Illustrative observed/hidden-node graph: a proxy and an effect can carry predictive information without identifying intervention effects.',
'classes':'Strict bound comparisons on one coherent five-row example, followed by shared arbitrary class renaming.',
'posterior':'Two-world exact arithmetic: query-feature likelihood changes posterior weights and the class probability from .58 to 11/14.',
'results':'Fresh author reference: paired ordinary/shuffled-context logistic losses on 12 sampled worlds per family. Individual task gaps remain visible; this is not the paper prior-training ablation.'}

def piece(name):
 source=(ROOT/'relkit/scm_l063_v2.py').read_text()
 return next(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/f'{SLUG}.md').read_text())
def figure(name,notebook=False):
 path='figures/l063/'+name+'-v2.png';caption=FIGURES[name]
 if notebook:return '<div style="overflow-x:auto;max-width:100%" role="region" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/path).read_bytes()).decode()+'" style="width:760px;min-width:640px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+path+').'
 return '<figure class="scm-figure"><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable computational figure"><img src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+path+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 p='../' if prepared else '../labs/'
 links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(p+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(p+'_verify_l063_v2_results.json','Measured evidence'),(p+'l063-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 063 · five live code tasks + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'"'+(' download' if t=='Download notebook' else '')+'>'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Complete the runnable notebook in Colab or Jupyter; student functions are intentionally blank.</p></aside>'

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 063 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Sources](_sources_l063_v2.json) · [Fresh author evidence](_verify_l063_v2_results.json) · [Reproduction contract](l063-reproduction.md)

**Scope: historical numerical generator key parts.** Build the layered sparse SCM/BNN generator, observe nodes, construct multiclass tasks and evaluate a finite posterior. No pretrained checkpoint or new Transformer training is required. The default run creates 12 new worlds per family and 48 small logistic fits (one ordinary and one shuffled probe per world). Original prior-training and benchmark reproduction are INCOMPARABLE / NOT_RUN. PROVIDED cells expose the complete generator/experiment; five TODOs feed that experiment; CHECKs diagnose mechanism errors; EXIT saves your actual kernel identity and explanation. Synthetic data is the subject of this paper component, not a substitute for an unperformed real-data benchmark.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:md(figure(m[1],True))
  elif part.strip():md(part.replace('../labs/',''))
 md('''## Start the live implementation
Every function below is defined in this kernel. The experiment calls those names directly; imported original source is used only as a checker. Read the complete path before running: hyperparameters → world → episode → learner → score. The source-parity test is conditional on fixed weights and noise; it does not replace your functions with the author's generator.''')
 code('''# PROVIDED — imports and local artifact root
import base64,copy,functools,hashlib,inspect,json,platform,sys,time,types,zlib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.special import logsumexp
from scipy.stats import truncnorm,norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from IPython.display import display
for d in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (d/'relkit').is_dir():ROOT=d.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run the bootstrap or start inside the course repository')
output=ROOT/'data/cache/l063-v2';output.mkdir(parents=True,exist_ok=True)
VERSIONS={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
print('CPU generator lab; no model checkpoint download or cloud run')''')
 md('''### PROVIDED · Activation and blockwise sparsity
Elementwise activation acts on parent coordinates before the next affine map. ELU has slope 1 on the positive side and exp(x)−1 below 0; LeakyReLU uses slope .01 below 0. block_weights keeps floor-sized diagonal blocks. Leftover coordinates remain zero, matching the historical branch.''')
 for n in ['activation_value','block_weights']:code('# PROVIDED — '+n+'\n'+piece(n))
 for i,(name,signature,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal.** {goal}\n\n**Why it matters.** {why}\n\n**Hint boundary.** {hint}')
  code(f'# TODO {i} — {name}\n'+(piece(name) if solution else signature+'\n    raise NotImplementedError("Implement the stated operation")'))
  code(f'# CHECK {i} — {name}\n'+CHECKS[name]+f'\nprint("PASS: {name}")')
 md('''## PROVIDED · Sample a world once, then draw all its rows
sample_meta_scale follows the released relative-standard-deviation sampler, not the simplified Table5 description. sample_world keeps matrices, activation and noise distributions fixed across the episode. The caps, Gaussian root-only branch and missing wrapper refinements are deliberate scope choices. sample_episode follows both SCM and BNN observation routes; it rejects accidental target/feature overlap, samples full-episode target bounds, and applies a shared class renaming. Read where new random numbers enter. Context and query do not receive independently sampled graphs.''')
 for n in ['sample_meta_scale','sample_world','sample_episode']:code('# PROVIDED — '+n+'\n'+piece(n))
 md('''### CHECK · Fixed noise and shared world are executable claims
Generating with the same frozen causes, weights and noises must replay the same values. The checker below executes the historical MLP class and records its actual noise terms before comparing your propagation. It also executes the original MulticlassRank on fixed random draws. The full wrapper's support repair/contiguous remapping/label shift is outside this parity claim.''')
 code('''# CHECK — pinned original source evaluates your live functions
from _check_l063_v2 import check as check_original
source_check=check_original(globals(),save=False)
print('Original forward max error:',source_check['max_forward_error'],'Classification cases:',source_check['multiclass_source_cases'])
world=sample_world(np.random.default_rng(633),'SCM')
episode=sample_episode(world,np.random.default_rng(6331),64)
replayed=propagate(episode['causes'],world['weights'],world['biases'],episode['noises'],world['activation'])
assert all(np.array_equal(a,b) for a,b in zip(replayed,episode['outputs']))
print('Frozen world/noise replay exact; observed roles:',episode['roles'])''')
 md('''## PROVIDED · Measurements with explicit information boundaries
fitted_probe fits both scaling and a C=1 logistic model on context rows only. The shuffled arm permutes only those labels. It preserves missing-class difficulties through tiny probabilities rather than rejecting inconvenient tasks. paired_intervention changes one named final-layer edge with the same root/noise draws; its known-parent effect has an exact algebraic oracle. It does not estimate a causal effect from observational data. finite_oracle_experiment is a separate, exactly specified two-world model; no claim of tractable exact Bayesian inference over the full TabPFN prior is made.''')
 for n in ['fitted_probe','paired_intervention','finite_oracle_experiment','serializable','decode_array','summarize']:code('# PROVIDED — '+n+'\n'+piece(n))
 from relkit.scm_l063_v2 import PRESETS
 code('# PROVIDED — explicit fresh budgets\nPRESETS='+repr(PRESETS)+'\n\n'+piece('run_experiment'))
 md('''## CHECK · Identity of the code this kernel actually uses
The author source hash identifies a repository file; it cannot identify an edited notebook function. We capture the semantic bytecode/constants (including nested code objects), defaults, function sources, all explicitly listed generator/evaluator helpers, active configuration and library versions. The helper names below are a complete inventory of this visible core, including functions called inside comprehensions. Wrappers are removed before capture. There is no checkpoint resume: fresh run paths reject existing files. EXIT is a latest-submission artifact, not a resumable training cache.''')
 names=[n.name for n in ast.parse((ROOT/'relkit/scm_l063_v2.py').read_text()).body if isinstance(n,ast.FunctionDef)]
 code('# PROVIDED — actual executable kernel identity\nKERNEL_NAMES='+repr(names)+'''
def stable_value(value):
    if isinstance(value,types.CodeType):
        return {'bytecode':value.co_code.hex(),'constants':[stable_value(c) for c in value.co_consts],'names':value.co_names,'varnames':value.co_varnames,'freevars':value.co_freevars,'cellvars':value.co_cellvars,'argcount':value.co_argcount,'kwonly':value.co_kwonlyargcount,'posonly':value.co_posonlyargcount,'flags':value.co_flags,'exceptiontable':value.co_exceptiontable.hex()}
    if inspect.isfunction(value):
        fn=inspect.unwrap(value)
        return {'code':stable_value(fn.__code__),'defaults':stable_value(fn.__defaults__),'kwdefaults':stable_value(fn.__kwdefaults__),'closure':stable_value(tuple(c.cell_contents for c in fn.__closure__)) if fn.__closure__ else None}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,bytes):return {'bytes':value.hex()}
    if isinstance(value,(tuple,list)):return [stable_value(v) for v in value]
    if isinstance(value,dict):return {str(k):stable_value(v) for k,v in value.items()}
    raise TypeError('Unsupported live identity value '+type(value).__name__)
def kernel_identity(config):
    content={'functions':{name:stable_value(inspect.unwrap(globals()[name])) for name in KERNEL_NAMES},'sources':{name:inspect.getsource(inspect.unwrap(globals()[name])) for name in KERNEL_NAMES},'presets':copy.deepcopy(PRESETS),'config':copy.deepcopy(config),'versions':dict(VERSIONS),'identity_code':stable_value(stable_value)}
    content['sha256']=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    return content
''')
 code('''# CHECK — global helper, function default, nested code and preset mutations are detected
before=kernel_identity(PRESETS['lab'])['sha256'];original=activation_value;defaults=sparse_weights.__defaults__;original_tasks=PRESETS['smoke']['tasks']
def altered_activation(x,activation):return original(x,activation)+.01
try:
    activation_value=altered_activation
    assert kernel_identity(PRESETS['lab'])['sha256']!=before
    activation_value=original;sparse_weights.__defaults__=(True,)
    assert kernel_identity(PRESETS['lab'])['sha256']!=before
    sparse_weights.__defaults__=defaults;PRESETS['smoke']['tasks']+=1
    assert kernel_identity(PRESETS['lab'])['sha256']!=before
finally:
    activation_value=original;sparse_weights.__defaults__=defaults;PRESETS['smoke']['tasks']=original_tasks
assert kernel_identity(PRESETS['lab'])['sha256']==before
# A helper called inside a generator expression is still covered by explicit dependency inventory.
def nested_call():return sum(float(activation_value(np.array([0.]),'identity')[0]) for _ in range(2))
assert nested_call()==0
activation_value=altered_activation
assert nested_call()==.02 and kernel_identity(PRESETS['lab'])['sha256']!=before
activation_value=original
assert any(isinstance(c,types.CodeType) for c in nested_call.__code__.co_consts)
assert 'bytecode' in stable_value(nested_call)['code']
print('Identity mutations detected and restored; no resume supported')''')
 md('''## Run a new panel through your five functions
Before running, write which direction you expect the ordinary/shuffled gap to have and identify one reason a finite task might violate it. This run uses new seeds 1630–1641. The author panel uses 630–641 and remains separate. Counts instrument real downstream calls: a call from a CHECK does not count toward the evidence that these tasks feed the experiment. Neither block mode nor any provided function calls a TODO merely to discard its output.''')
 code('''# PROVIDED — fresh student experiment with live TODO call counters
config=copy.deepcopy(PRESETS['lab']);config['seed']=1630
originals={name:globals()[name] for name in ['sparse_weights','propagate','select_nodes','rank_labels','posterior_weights']};call_counts={name:0 for name in originals}
def counted(name,fn):
    @functools.wraps(fn)
    def call(*args,**kwargs):
        call_counts[name]+=1
        return fn(*args,**kwargs)
    return call
for name,fn in originals.items():globals()[name]=counted(name,fn)
try:
    path=output/f'fresh-lab-{time.time_ns()}.json'
    fresh=run_experiment('lab',path,config)
finally:
    for name,fn in originals.items():globals()[name]=fn
assert all(v>0 for v in call_counts.values()),'Every TODO must feed the measured experiment'
fresh['kernel_identity']=kernel_identity(config);fresh['live_call_counts']=call_counts
path.write_text(json.dumps(fresh,indent=2,allow_nan=False)+chr(10))
display(pd.DataFrame(fresh['summary']));print('Actual task calls:',call_counts);print('Saved fresh student run:',path)
print('Finite oracle NLL:',fresh['finite_oracle']['correct_nll'],'Without query-x evidence:',fresh['finite_oracle']['ignored_query_x_nll'])''')
 md('''## CHECK · Reconstruct saved evidence rather than trust a summary
The artifact preserves the world, row draws, labels, probabilities and intervention oracle. Large arrays use lossless zlib/base64; decode_array restores their exact dtype/shape. We recompute loss from probabilities and hidden targets, reconstruct generated features/labels through your functions, and audit intervention errors. Compression is packaging, not a change of measurements. Some sampled worlds have affine values near 10⁸: subtracting two such values to recover an edge effect of about 59 can lose low-order floating-point digits. A discrepancy around 10⁻⁸ is then compatible with the exact algebraic identity. The CHECK bounds rounding error using the sum of absolute affine contributions and machine precision (64ε), rather than scaling tolerance only by the small final difference. A wrong operation that exceeds this justified bound still fails.''')
 code('''# CHECK — independently reconstruct every fresh task's probability loss and generator path
for r in fresh['records']:
    target=np.asarray(r['targets']);p=decode_array(r['probabilities']);ps=decode_array(r['shuffled_probabilities'])
    np.testing.assert_allclose(-np.log(p[np.arange(len(target)),target]).mean(),r['nll'])
    np.testing.assert_allclose(-np.log(ps[np.arange(len(target)),target]).mean(),r['shuffled_nll'])
    w=r['world'];e=r['episode'];c=decode_array(e['causes']);eps=[decode_array(v) for v in e['noises']]
    h=propagate(c,[decode_array(v) for v in w['weights']],[decode_array(v) for v in w['biases']],eps,w['activation'])
    values=np.concatenate(h[1:],1) if r['family']=='SCM' else np.column_stack([c,h[-1][:,0]])
    x,z=select_nodes(values,e['roles']['feature_nodes'],e['roles']['target_node'])
    y,bounds=rank_labels(z,np.array(e['bound_indices']),np.array(e['class_permutation']))
    np.testing.assert_array_equal(y,e['y']);np.testing.assert_allclose(x,decode_array(e['x']))
    assert r['intervention']['unchanged_upstream']
    child=r['intervention']['child'];last_w=decode_array(w['weights'][-1])[child]
    # Cancellation error scales with the summed affine contributions, not only the small delta.
    scale=np.abs(activation_value(h[-2],w['activation']))@np.abs(last_w)+abs(w['biases'][-1][child])+np.abs(eps[-1][:,child])
    roundoff_bound=64*np.finfo(float).eps*np.maximum(1,scale)
    residual=np.abs(np.array(r['intervention']['delta'])-r['intervention']['oracle_delta'])
    assert np.all(residual<=roundoff_bound),'Edge identity exceeds the scale-aware floating-point bound'
print('All fresh saved probabilities, generator outputs, class labels and paired edge oracles reconstructed')''')
 code('''# CHECK — reject an edit made after measurements, even if EXIT alone is re-executed
measured_identity=fresh['kernel_identity']['sha256']
assert kernel_identity(config)['sha256']==measured_identity
original_activation=activation_value
def edited_after_run(x,activation):return original_activation(x,activation)+.02
try:
    activation_value=edited_after_run
    try:
        assert kernel_identity(config)['sha256']==measured_identity,'Stale result: rerun measurement after code edits'
    except AssertionError:print('PASS: post-measurement helper edit rejected')
    else:raise AssertionError('Stale-result guard did not fire')
finally:activation_value=original_activation
assert kernel_identity(config)['sha256']==measured_identity
assert json.loads(path.read_text())['kernel_identity']['sha256']==measured_identity
print('Fresh result parses as JSON and still matches the live kernel')''')
 md('''## Read the author panel as a separate experiment
The author panel is a fixed reference, not output from this kernel. Its means are conditional on 24 worlds and a deliberately limited logistic learner. SCM and BNN seeds do not generate matched worlds. Use the saved individual gaps to find a counterexample to the claim that shuffling must always hurt. Then inspect whether one class, weak predictive information or linear-model misspecification could explain it; the score alone cannot select among these explanations.''')
 code('''# PROVIDED — separately labeled author reference
reference=json.loads((ROOT/'_verify_l063_v2_results.json').read_text())
print('AUTHOR REFERENCE only:',reference['operator'],reference['config'])
display(pd.DataFrame(reference['summary']))
failures=[{'family':r['family'],'seed':r['seed'],'ordinary':r['nll'],'shuffled':r['shuffled_nll'],'classes_in_context':len(set(r['episode']['y'][:r['n_context']]))} for r in reference['records'] if r['shuffled_nll']<=r['nll']]
display(pd.DataFrame(failures))''')
 md('''## EXIT · Submit a prior card with a failed case
Write at least 60 words that cover world-level versus row-level randomness; the released activation/affine/noise order; observed/hidden roles and the classification rule; why query x may reweight worlds; a counterexample in your measurements; two distribution deviations; and one relational dependency this prior omits. State what controlled experiment could falsify your proposed improvement. Code checks and teacher example text are not evidence of personal mastery.''')
 verdict=('One world fixes its graph, matrices, biases, activation and noise distributions for all rows; only row causes and structural noise draws vary. The released repeated step activates parent values, applies the affine map and adds noise. Feature and target roles hide unselected nodes; strict sampled bounds and one class permutation create labels. Query features can reweight joint worlds. Some shuffled logistic probes improve, so a finite score is not a universal causal law. Our Gaussian root-only branch and architecture caps differ from the complete prior. A customer-to-transactions aggregation prior should be tested with frozen temporal splits and equal predictor training budgets; this lab does not reproduce the original PFN ablation.')
 code('# TODO — written interpretation, then save the latest EXIT submission\n'+(('example=next(r for r in fresh[\'records\'] if r[\'shuffled_nll\']<r[\'nll\'])\nverdict='+repr(verdict)+'+f\" Concrete measured counterexample: {example[\'family\']} seed {example[\'seed\']}, ordinary loss {example[\'nll\']:.4f}, shuffled loss {example[\'shuffled_nll\']:.4f}; finite logistic fitting and sample noise do not guarantee a positive gap.\"') if solution else 'verdict=\'\'')+'''
assert len(verdict.split())>=60,'Write the complete defensible prior card before submitting'
assert kernel_identity(config)['sha256']==fresh['kernel_identity']['sha256'],'Code or configuration changed after measurement; rerun the fresh experiment'
assert json.loads(path.read_text())['kernel_identity']['sha256']==fresh['kernel_identity']['sha256'],'Saved run identity mismatch'
ticket={'lesson':63,'checks':'passed in this kernel','fresh_run':str(path),'fresh_run_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'call_counts':call_counts,'summary':fresh['summary'],'finite_oracle_nll':fresh['finite_oracle']['correct_nll'],'kernel_identity':kernel_identity(config),'verdict':verdict,'paper_reproduction':'INCOMPARABLE','full_pretraining':'NOT_RUN'}
exit_path=output/'student-l063-exit.json';exit_path.write_text(json.dumps(ticket,indent=2,allow_nan=False)+chr(10))
print('EXIT artifact:',exit_path)''')
 md('''## After EXIT · Larger generator audit through the same live functions
The closer gate increases worlds/rows and the exact finite-oracle sample count. It is still a logistic generator audit, not a matched PFN training ablation. Expect more class-support difficulties and broader world variation, not an automatic move toward Table4's numbers. Full historical prior fitting has its own source/protocol route in the reproduction contract and remains NOT_RUN.''')
 code('''# PROVIDED — explicit local gate, OFF by default; no cloud job
RUN_CLOSER=False
if RUN_CLOSER:
    closer_path=output/f'closer-{time.time_ns()}.json'
    closer=run_experiment('closer',closer_path)
    closer['kernel_identity']=kernel_identity(PRESETS['closer'])
    closer_path.write_text(json.dumps(closer,indent=2,allow_nan=False)+chr(10))
    print('Larger generator audit saved:',closer_path,'Full original prior-fitting NOT_RUN')
else:print('Closer NOT_RUN in this kernel; no paper reproduction preset exists')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in old.cells if c.cell_type=='code'];b=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   for previous,current in zip(a,b):current.outputs=previous.outputs;current.execution_count=previous.execution_count
   if 'execution_verification' in old.metadata:nb.metadata['execution_verification']=old.metadata['execution_verification']
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  key='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[key])
  if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO1') or h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'))
 soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def evidence_table():
 result=json.loads((ROOT/'_verify_l063_v2_results.json').read_text());out='<div class="table-scroll" role="region" aria-label="Measured synthetic probe results" tabindex="0"><table><thead><tr><th>Family</th><th>Ordinary NLL</th><th>Shuffled NLL</th><th>Gap mean ± SD</th><th>Positive gaps</th></tr></thead><tbody>'
 for r in result['summary']:
  values=[r['family'],f"{r['nll']:.4f}",f"{r['shuffled_nll']:.4f}",f"{r['paired_gap_mean']:.4f} ± {r['paired_gap_sd']:.4f}",f"{r['positive_gaps']}/{r['tasks']}"]
  out+='<tr>'+''.join('<td class="score">'+v+'</td>' for v in values)+'</tr>'
 return out+'</tbody></table></div><p>Sample SD across generated worlds; no real-dataset or training-seed confidence claim.</p>'

def build_package(notebooks=True,render=True):
 body=''
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:
   name=m[1];body+=figure(name)
   if name=='generator':body+='<div id="scm-edge-viz" class="scm-widget"><p>The static figure retains the baseline; interactive edge controls require JavaScript.</p></div>'
   if name=='posterior':body+='<div id="scm-posterior-viz" class="scm-widget"><p>The static figure retains the baseline; interactive posterior controls require JavaScript.</p></div>'
   if name=='results':body+=evidence_table()
  else:body+=markdown2html_mistune(part)
 soup=BeautifulSoup(body,'html.parser')
 for i,h in enumerate(soup.select('h2,h3')):
  if not h.get('id'):h['id']=re.sub('[^a-z0-9]+','-',h.get_text().lower()).strip('-')
 for table in soup.find_all('table'):
  if table.parent.get('class')==['table-scroll']:continue
  wrap=soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable prior specification'});table.wrap(wrap)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 063 · {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','foundation-course','lab-access','l063-scm'])+'</head><body class="l063"><article>'
 opening='<nav><a href="../index.html">Course</a> · <a href="0062-tabpfn-v1.html">← Lesson 062</a> · <a href="0064-tabpfn-v2.html">Lesson 064 →</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 063</p><h1>'+TITLE+'</h1>'+launcher()+'<section id="retrieval"><h2>Retrieve before reading</h2><div id="warmup"></div></section><div id="prediction"></div>'
 cfg=dict(lesson=63,answer='One sampled world is shared across context and query rows. The released generator activates parents before affine maps and then adds noise. Node observation and strict sampled class bounds define the supervised task. Query features can update posterior world weights. Synthetic interventions are exact only inside the known world; neither generator parity nor logistic probes reproduce the paper PFN prior ablation.',quiz=['In a joint SCM prior, what can query features change?',['The posterior weights over worlds','The already observed context labels','The fixed pretrained model weights'],0,'A world that makes the query features more likely can gain posterior mass, even before the query label is observed.'])
 scripts='<script id="foundation-config" type="application/json">'+json.dumps(cfg)+'</script>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','foundation-lesson','l063-scm-viz'])
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(head+opening+str(soup)+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div><p>Ask the teacher about any operation or failed case you cannot defend.</p></section></article>'+scripts+'</body></html>')
 ref=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 063</a></nav><h1>SCM prior: execution and evidence card</h1>'+launcher()+figure('generator')+markdown2html_mistune('''
## Two levels of randomness

World-level: graph, weights, biases, activation, noise distributions and observation/class rule. Row-level: root values and structural noises. Share one world across context and queries.

## Released operator

First affine H0=CW0ᵀ+b0. Later Hl=a(Hl−1)Wlᵀ+bl+εl. W indexes child,parent. SCM observes nodes from H1,H2,…; BNN observes roots and final scalar output. Source C.1's abbreviated equation differs; this card follows historical mlp.py.

Independent-edge scaling with the checkpoint sqrt flag: divisor1−√p, first affine undropped. Block mode uses√keep_fraction and includes first affine. Strict target-bound exceedances → interval index → one shared class permutation. Duplicate bounds may leave empty classes. Target/feature overlap is rejected locally.

## Joint posterior

p(φ|x,D) ∝ p(φ)p(D|φ)p(x|φ) for conditionally independent rows of a joint world. The finite fixture (.5,.5)×(.4,.6)×(.2,.8) normalizes to(1/7,6/7); averaging class probabilities(.1,.9) gives11/14. The independent-row factorization is not automatically valid for a jointly constructed episode label rule.

## Five live tasks
'''+ '\n'.join('- `'+t[1]+'` — '+t[2] for t in TASKS))+evidence_table()+markdown2html_mistune('''
## Exact scope and sources

[Paper v6 §4/C/E.4/B.4](https://arxiv.org/html/2207.01848v6) · [Historical mlp.py](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/priors/mlp.py) · [Source inventory](../labs/_sources_l063_v2.json) · [Source checks](../labs/_check_l063_v2_results.json) · [Fresh measurements](../labs/_verify_l063_v2_results.json) · [Visible core](../labs/relkit/scm_l063_v2.py) · [Reproduction route](../labs/l063-reproduction.md).

The lab caps graph sizes, conditions on Gaussian roots and omits several categorical/missingness/preprocessing/support-repair refinements. Conditional operator parity is not RNG or full-prior parity. Logistic probes are not PFNs trained under alternative priors. Full original prior fitting and Table4 reproduction remain NOT_RUN / INCOMPARABLE. A known synthetic edge intervention is not real causal discovery or an estimate of do-effects from arbitrary observations.
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(ref+'</article></body></html>')
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L063 v2 package')
if __name__=='__main__':build_package()
