"""Build lesson, field guide and portable notebooks from canonical B07a code."""
import ast,base64,copy,hashlib,io,json,zipfile,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b07a';S='b07a-hypernetworks'
r=json.loads((E/'course-audit.json').read_text());run=json.loads((E/'runs/results.json').read_text())
weights='''<div class="b07a-board" data-b07a="weights"><h3>Keep the generated weights; change the query path</h3><p>Toy hidden query z=[1,t]. Predict the winning class before enabling retrieval.</p><label>Hidden coordinate t: <strong data-q>2</strong><input type="range" min="-1" max="3" step="1" value="2"></label><label><input type="checkbox"> Add illustrative class 0 neighbor correction (+10)</label><button type="button">Reset weights example</button><output aria-live="polite">At z=[1,2], logits=[11.2,20.5]. One class 0 correction of10 changes logits to[21.2,20.5]. Generated weights stay fixed.</output></div>'''
costs='''<div class="b07a-board" data-b07a="costs"><h3>Construction versus repeated work</h3><p>Illustrative assumptions: A builds in 2 s and costs 1 ms/query; B builds in 0 s and costs 5 ms/query. Hold these costs fixed.</p><label>Total queries<input data-volume type="number" min="0" max="1000000" value="500"></label><label>Support rebuilds<select data-refresh><option value="0">0 rebuilds</option><option value="1">1 rebuild</option><option value="2">2 rebuilds</option></select></label><button type="button">Reset cost example</button><output aria-live="polite">At 500 queries and 0 rebuilds, both cost 2.5 s. A is strictly cheaper from 501 queries. Two rebuilds move that threshold to 1501.</output></div>'''
def figure(name,caption):return f'<figure class="b07a-figure"><img src="../labs/figures/b07a/{name}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
captions={'mothernet':'MotherNet v2: class-pooled transformer output generates rank32 factors for a two-hidden-layer child. Fixed factors were learned during meta-training. Architecture only.','hyperfast':'Full released HyperFast path: 1024 repeated support rows, 32768 random features, 784 PCA coordinates and generated layers. Retrieval retains support.','iltm':'iLTM paper path: fitted tree/robust features, fixed-size projection, generated MLP and optional cosine-weighted context. Architecture only.'}
table='| Dataset | Weights only | With retrieval | Paired change |\n|---|---:|---:|---:|\n'
for d in ['banknote','phoneme','diabetes']:
 vals=[next(x for x in r['aggregate'] if x['dataset']==d and x['arm']==a) for a in ['weights_only','retrieval']];delta=next(x for x in r['paired'] if x['dataset']==d)
 table+='| '+d+' | '+' | '.join(f"{100*x['mean']:.2f} ± {100*x['sd']:.2f}%" for x in vals)+f" | {100*delta['mean']:+.2f} pp |\n"
table+='\nMean balanced accuracy ± sample SD over three paired split/model seeds. Paired change is retrieval minus weights-only, in percentage points. SD is not a confidence interval.\n'
construction=[x['construction_seconds'] for x in run['records'] if x['arm']=='weights_only'];import statistics
nn=next(x for x in run['records'] if x['dataset']=='banknote')
discussion=f"""All **{r['runs']} prediction arms / {r['predictions']:,} query predictions** passed independent row/label, output-layer, retrieval and scalar-score reconstruction. Source-wrapper prediction parity was checked for all nine generated predictors.

Construction took **{min(construction):.2f}–{max(construction):.2f}s** across nine predictors (median {statistics.median(construction):.2f}s). Checkpoint object loading took {run['load_seconds']:.2f}s; authentication/download is separate. The banknote seed0 refresh took**{run['refresh']['seconds']:.2f}s**, changed the generated-state hash, and was not scored as held-out quality.

For banknote, pure predictor arrays require about**{nn['retained_bytes']['weights_only_minimum']/2**20:.2f}MiB**; adding stored raw support/labels gives**{nn['retained_bytes']['retrieval_minimum']/2**20:.2f}MiB**. The shared hypernetwork's parameters occupy about**{run['hypernetwork_storage_bytes']/2**30:.2f}GiB**; do not confuse the builder with its output. Python/temporary/allocator/peak memory is excluded. The adapter retains support in both arms; discarding it in the pure path is a possible deployment change, not something measured here.

Retrieval raises mean balanced accuracy from 80.38% to 100.00% on banknote and 74.61% to 77.34% on phoneme, but lowers it from 69.59% to 65.72% on diabetes. Phoneme has one negative split effect; diabetes has three. Banknote happens to equal the published 100% number, but the original split/subsample/search identities are still missing: this numerical coincidence is not reproduction.

The quality plot connects the same split across paths. Read each line before trusting a mean: retrieval is an intervention, not a guaranteed improvement. The timing plot excludes construction and shows the release's support recomputation cost. With only these three tables and one checkpoint, no broad model ranking is justified. Nearest-neighbor-only and random-generator controls were not run, so the contribution of meta-training is not isolated. Full per-split log loss and raw timings are in the audit and run report.
"""
replacements={'[[WEIGHTS_WIDGET]]':weights,'[[COST_WIDGET]]':costs,'[[RESULTS]]':table,'[[RESULT_DISCUSSION]]':discussion,'[[QUALITY]]':figure('quality.png','Measured paired balanced accuracy. Lines connect the same split; all panels share a common vertical scale.'),'[[TIMING]]':figure('timing.png','Measured CPU call time: medians of9 observations per point across3 seeds and3 repeats. Construction is excluded; panel scales differ.')}
for n,c in captions.items():replacements['[['+n.upper()+']]']=figure(n+'.svg',c)
def doc(title,body):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/hypernetworks.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','hypernetworks'])+'</body></html>'
text=(R/'lessons/content'/(S+'.md')).read_text()
for a,b in replacements.items():text=text.replace(a,b)
assert '[[' not in text
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B07a — Hypernetworks: generate a predictor from a table',render(text)))
reference='''# Hypernetworks · field guide

**One question:** after support becomes weights, what still runs for a query?

| Model | Generator | Retained query path |
|---|---|---|
| MotherNet | Support transformer → class means → low-rank factors | Preprocessing and child MLP; ensembles add children |
| HyperFast | RF/PCA → shared MLP summaries → sequential layers | RF/PCA and generated MLP; optional two-space1-NN |
| iLTM | Fitted tree/robust embedding → sequential hypernetwork | Representation and MLP; optional cosine/label context |

**Two timescales:** θ is learned across tasks; φ=hθ(S) is generated for one task. No downstream optimizer does not mean no pretraining. fφ(x,S) still needs support; fφ(x) may discard it.

**Generated class head:** average per-row generated vectors by class; add mean main-hidden state to weight coordinates only; the final coordinate is the bias. If a class has no sampled row, the release uses a global mean. Repeat rows are not new observations.

**Worked arithmetic:** class 0 mean[2,3,.2]+hidden[1,1]→[3,4,.2]; class 1→[6,7,.5]. Query[1,2] yields[11.2,20.5]. A class 0 correction10 changes the winner without changing weights.

**Cost:** T=(U+1)B+Qc, where U counts support rebuilds. Add fitted preprocessing, tuning, ensembling, retrieval and caching at the appropriate stage. A build2s/query1ms versus B build0s/query5ms ties at500queries and favors A from501. With2rebuilds, threshold1501. Real batching invalidates a universal per-query slope.

**Evidence:** full-dimensional selected checkpoint course inference;3datasets×3 seeds×2paths. Original Table7 banknote requires10 mini-test repetitions at300s each and remains INCOMPLETE_SOURCE_PROTOCOL. Its split/subsample/search identities are missing. Paper target100±0% is cited, not reproduced. Full pretraining/MotherNet/iLTM runs NOT_RUN. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/b07a-hypernetworks.html) · [Lab](../labs/b07a-hypernetworks.ipynb) · [Contract](../labs/b07a-reproduction.md) · [MotherNet §3](https://arxiv.org/html/2312.08598v2#S3) · [HyperFast](https://arxiv.org/html/2402.14335v1) · [iLTM §3](https://arxiv.org/html/2511.15941v1#S3).

Retrieve at1/7/30days after completion. Ask the agent to assess your causal explanation and evidence boundary.
'''
(R/'reference/b07a-hypernetworks.html').write_text(doc('B07a · Field guide',render(reference)))
# Portable small archive, explicitly excluding the 5 GB checkpoint and caches.
files={}
for name in ['relkit/hyper_b07a.py','relkit/hyperfast_b07a.py','relkit/serving_b07a.py','_run_b07a.py','_audit_b07a.py','_source_b07a.py','_reproduce_b07a.py','_test_b07a.py','_budget_b07a.py','b07a-reproduction.md']:
 files['labs/'+name]=(P/name).read_bytes()
files['labs/relkit/__init__.py']=b''
for root in [P/'sources/b07a',P/'data/b07a',E/'runs']:
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.ckpt','.partial'] and p.name!='partial.json':files['labs/'+str(p.relative_to(P))]=p.read_bytes()
for n in ['course-protocol.json','source-gate.json','course-audit.json','source-parity.json','execution-environment.json']:files['labs/evidence/b07a/'+n]=(E/n).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(files.items()):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
archive=buf.getvalue();(E/'reproducer.zip').write_bytes(archive)
md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
cells=[md('''# B07a · Generate a predictor from a table

**Skill:** trace support→generated parameters→query prediction and count the complete serving cost.

PROVIDED exposes the full released HyperFast architecture, optional downstream optimizer and course inference loop. TODO contains 3 live functions; CHECK gives immediate hand-computed feedback; EXIT requires a written defense. Tier-A real data: banknote, phoneme and diabetes. The archive includes authenticated saved evidence for offline audit. Fresh execution additionally needs the 5.09 GB original checkpoint; no paid service is required. Live Colab NOT_CHECKED.

## Concept recap
A support row includes a known label; a query does not. A hypernetwork hθ generates weights φ=hθ(S) for another network fφ. Meta-training learns θ across tasks. Generating φ with frozen θ is adaptation without an optimizer. The produced MLP applies matrix multiplications, biases and ReLU (negative→zero). Softmax turns output logits into class probabilities.

A pure generated predictor can retain only its preprocessing and parameters. HyperFast's optional nearest-neighbor correction and iLTM's retrieval still require support. MotherNet uses a support transformer and low-rank child factors; HyperFast uses feedforward modules over RF/PCA features; iLTM adds tree/robust embeddings and cosine-weighted context. They are not interchangeable inference paths.

**Predict before coding:** must retrieval improve every split? Does one forward pass eliminate pretraining cost?
''')]
for n,c in captions.items():cells.append(md('## Model architecture · '+n+'\n\n![Architecture](data:image/png;base64,'+base64.b64encode((P/f'figures/b07a/{n}.png').read_bytes()).decode()+')\n\n'+c))
cells.append(md('''## Worked trace and experiment contract
Class0 generated rows[1,2,.1],[3,4,.3] average to[2,3,.2]. Hidden rows[2,0],[0,2] average to[1,1]. Add only to the first2coordinates→[3,4,.2]. Class1 gives[6,7,.5]. Query hidden[1,2] yields logits[11.2,20.5]. A nearest-class 0 correction10 reverses the winner. These are illustrative values, not checkpoint outputs.

Full released dimensions: 32768 random features→784PCA coordinates; two784-wide hidden layers and class output. The generated RF/PCA transformation is reused on queries. Support summaries condition each successive layer. Three full datasets×three fresh 80/20 seeds×retrieval off/on;18 arms share 9 generated predictors. Scaler fits all training rows; 512 sampled rows repeat to 1024 for PCA. No tuning, ensembles or fine-tuning. A tenth construction measures one banknote support refresh only. Timing 1/32/128 query rows, one warm-up and 3 repeats/seed, one CPU thread; retrieval recomputes support hidden vectors like the release.

Balanced accuracy=(recall0+recall1)/2. Log loss=mean(−log probability of true class). Sample SD over3splits is not a confidence interval. The original Table7 target uses 10 mini-test repetitions with a 300 s budget and is a separate source-gated experiment.
'''))
versions=json.loads((E/'course-protocol.json').read_text())['versions']
cells.append(code('''# @colab-bootstrap: install missing dependencies; report version drift.
import importlib.util,importlib.metadata,subprocess,sys,warnings,os
'''+f'versions={versions!r}\n'+'''for module,package in [('numpy','numpy'),('pandas','pandas'),('torch','torch'),('sklearn','scikit-learn'),('scipy','scipy'),('requests','requests')]:
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable,'-m','pip','install',package+('=='+versions[package] if package in versions else '')])
    elif package in versions and importlib.metadata.version(package)!=versions[package]:warnings.warn(package+' version differs; inspect parity')
import ast,base64,hashlib,io,json,math,statistics,zipfile,tempfile,time,gc,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
'''))
payload=code('payload='+repr(base64.b64encode(archive).decode())+'\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=='+repr(hashlib.sha256(archive).hexdigest())+"\nworkspace=Path(tempfile.mkdtemp(prefix='b07a-portable-'))\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())\n    z.extractall(workspace)\nlab=workspace/'labs';evidence=lab/'evidence/b07a'\nprint('Portable evidence/source package authenticated; checkpoint excluded')")
payload.metadata['tags']=['data-payload'];cells.append(payload)
def extract(path,names):
 text=(P/path).read_text();return '\n\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
for name,checker,explain in [('class_weights','check_class_weights','Group support rows by class. Average generated vectors and add mean main-hidden activations to weight coordinates only. Return(hidden+1,classes). Preserve the input tensors. Match the release fallback to the global mean for an absent class.'),('retrieval_bias','check_retrieval','Compute Euclidean nearest-support classes in blocks of128query rows. First support index wins ties. Add the given amount to each matched class in a copied logit matrix. Query labels are not an argument.'),('break_even','check_break_even','Return the first nonnegative integer query count where A is strictly cheaper, or None. Both methods rebuild refreshes+1times. Reject negative/nonfinite costs. Explain the difference between a tie and strict savings.')]:
 cells.append(md('## TODO · '+name+'\n\n'+explain));cells.append(code(extract('relkit/hyper_b07a.py',[name])));cells.append(code(extract('_test_b07a.py',[checker])+'\n'+checker+'('+name+')\nprint("CHECK passed")'))
sections=[(['seed_everything','svd_flip','TorchPCA'],'RF/PCA support transformation','Full SVD produces deterministic-sign PCA coordinates. Transform uses the fitted mean and components. Random feature projection is created inside the generator.'),(['get_main_weights','forward_linear_layer','HyperFast'],'Complete released weight generator','Follow the1024support rows through means,46label slots and sequential layers. The final output calls your class_weights. Full dimensions remain unchanged.'),(['forward_main_network','transform_data_for_main_network','distance_matrix','NN','nn_bias_logits'],'Main network and original neighbor mechanics','The residual connects the input of the hidden pair to its second output. Original nearest-neighbor helpers are visible for comparison; the course adapter uses your live retrieval_bias. The paper ablation labels the first correction as PCA-based; both archived wrappers use standardized original inputs. This course follows the released code.'),(['MainNetworkTrainable','fine_tune_main_network'],'Optional released downstream optimizer — NOT_RUN','Included to make the full release inspectable. This AdamW loop fine-tunes a task model; it is not the missing meta-training loop and is not called by this experiment.')]
for names,title,explain in sections:cells.append(md('## PROVIDED · '+title+'\n\n'+explain));cells.append(code(extract('relkit/hyperfast_b07a.py',names)))
configline=next(line for line in (P/'relkit/serving_b07a.py').read_text().splitlines() if line.startswith('CONFIG='))
cells.append(md('## PROVIDED · Checkpoint loading and complete serving state\n\nMemory mapping avoids a duplicate 5 GB allocation. State keys load strictly with full float32 weights. Fit sees training labels only. Prediction recomputes support states for retrieval, matching the release. Array-byte accounting excludes temporary peak RAM.'))
cells.append(code(configline+'\n\n'+extract('relkit/serving_b07a.py',['load_hypernetwork','GeneratedPredictor'])))
cells.append(md('## PROVIDED · Full course runner and independent audit\n\nThese are the canonical experiment functions. The auditor reconstructs output logits from saved hidden vectors and output weights, then independently recomputes both nearest-neighbor corrections and scalar metrics. It authenticates full row IDs and paired predictor hashes.'))
cells.append(code(extract('_run_b07a.py',['digest','source_prediction_parity','run_course'])))
cells.append(code(extract('_audit_b07a.py',['audit_course'])+'\n\n'+extract('_source_b07a.py',['audit_sources'])))
cells.append(md('## CHECK · Offline saved-evidence replay\n\nThis audit executes without loading the checkpoint. It verifies author artifacts, not fresh inference or your mastery.'))
cells.append(code("author=audit_course(lab,evidence/'runs')\nassert author==json.loads((evidence/'course-audit.json').read_text())\nsource=audit_sources(lab/'sources/b07a')\nprint(pd.DataFrame(author['aggregate']).to_string(index=False))\nprint('Saved evidence audit',author['status'],'; original target',source['status'])"))
cells.append(md('## RUN · Optional fresh selected-checkpoint inference\n\nSet RUN_FRESH=True to regenerate all 9 predictors / 18 arms and one refresh. Download 5.09 GB once if needed; allow several minutes on CPU and sufficient RAM/disk. Original Table7 is not dispatched. The author solution enables this lane; the student default is offline. The two live mechanism functions are passed into the runner.'))
cells.append(code('''RUN_FRESH=True
if RUN_FRESH:
    import requests
    meta=json.loads((lab/'sources/b07a/checkpoint.json').read_text())
    checkpoint=Path(os.environ.get('HYPERFAST_CHECKPOINT',str(Path.home()/'.cache/b07a/hyperfast.ckpt')))
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if not checkpoint.exists() or checkpoint.stat().st_size!=meta['bytes']:
        partial=checkpoint.with_suffix('.partial');h=hashlib.sha256();n=0
        with requests.get(meta['url'],stream=True,timeout=90) as response:
            response.raise_for_status()
            with partial.open('wb') as f:
                for chunk in response.iter_content(8*1024*1024):f.write(chunk);h.update(chunk);n+=len(chunk)
        assert n==meta['bytes'] and h.hexdigest()==meta['sha256'],'Checkpoint download authentication failed'
        partial.replace(checkpoint)
    calls={'head':0,'bias':0}
    def counted_head(*args,**kwargs):
        calls['head']+=1
        return class_weights(*args,**kwargs)
    def counted_bias(*args,**kwargs):
        calls['bias']+=1
        return retrieval_bias(*args,**kwargs)
    fresh=workspace/'fresh-runs'
    report=run_course(lab,fresh,checkpoint,head_fn=counted_head,bias_fn=counted_bias)
    assert calls['head']==10 and calls['bias']>0,'Live learner functions were bypassed'
    course=audit_course(lab,fresh)
    assert course['rows']==author['rows'],'Fresh numerical results differ; inspect environment/protocol'
    print('Fresh source parity and scoring PASS',calls)
else:
    course=author
    print('Saved-evidence replay only; fresh inference NOT_RUN in this kernel')
assert break_even(2,.001,0,.005)==501
Path('b07a-report.json').write_text(json.dumps({'course':course,'source':source,'fresh':RUN_FRESH},indent=2))
'''))
for name,caption in [('quality','Author reference: paired splits on identical generated predictors. This figure is saved evidence, not output from a blank student kernel.'),('timing','Author reference: call-time medians,3 seeds × 3 repetitions. Construction excluded; hardware/cache policy matters.')]:cells.append(md('## Author reference · '+name+'\n\n![Measured result](data:image/png;base64,'+base64.b64encode((P/f'figures/b07a/{name}.png').read_bytes()).decode()+')\n\n'+caption))
cells.append(md(table+'\n\n'+discussion))
cells.append(md('''## EXIT · Written defense
1. Explain which objects must survive predictor construction in each of the3architectures.
2. Explain the class-head residual and why it does not add to the bias.
3. Interpret the paired split differences without claiming a universal retrieval benefit.
4. Derive the501query and1501query thresholds. Explain batching and possible cache changes.
5. Specify query-label and support-intervention falsifiers. Account for every refit.
6. List missing original Table7 inputs. Why would100% on our banknote split not prove reproduction?

Ask the agent for feedback. Review after 1/7/30 days following completion. Author checks are not learner mastery.
'''))
cells.append(code("submission={'status':'PENDING_WRITTEN_DEFENSE','defense':''}\nPath('b07a-submission.json').write_text(json.dumps(submission,indent=2))\nprint(submission['status'])"))
cells.append(md('''## NEXT STEP · Original published target
Original HyperFast v1 Table7 banknote:100±0% balanced accuracy;10 mini-test repetitions;300 seconds each. Original split/subsample IDs and selected search trajectories remain missing. The command below is a tested source preflight, not a complete runnable historical evaluator. It refuses execution rather than substituting our course splits. Full meta-training/all-paper/MotherNet/iLTM execution NOT_RUN. No cloud dispatch for an unresolved scientific protocol.

[MotherNet v2 §3](https://arxiv.org/html/2312.08598v2#S3) · [HyperFast v1](https://arxiv.org/html/2402.14335v1) · [iLTM v1 §3](https://arxiv.org/html/2511.15941v1#S3). Source archive contains originals and licenses. HyperFast source is CC BY-NC4.0. The portable package carries code/data/evidence; the checkpoint remains an authenticated external download.
'''))
cells.append(code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    subprocess.run([sys.executable,str(lab/'_reproduce_b07a.py'),'--run'],check=True)\nelse:\n    print('Original paper NOT_RUN:',source['status'])"))
solution=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
for i,c in enumerate(solution.cells):c.id=f'b07a-{i:03d}'
student=copy.deepcopy(solution)
for c in student.cells:
 if c.cell_type=='code':
  if c.source.startswith(('def class_weights(','def retrieval_bias(','def break_even(')):
   name=ast.parse(c.source).body[0].name;c.source=c.source.split('\n',1)[0]+f'\n    raise NotImplementedError("TODO: {name}")'
  c.source=c.source.replace('RUN_FRESH=True','RUN_FRESH=False')
nb.write(student,P/(S+'.ipynb'));nb.write(solution,P/'solutions'/(S+'.ipynb'))
print('Built',len(cells),'cells; portable ZIP bytes',len(archive))
