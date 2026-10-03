"""Build the B07 lesson and full portable live-code student/solution notebooks."""
import ast,base64,copy,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b07';S='b07-semantic-transfer'
r=json.loads((E/'course-audit.json').read_text())
intervention='''<div class="b07-board" data-b07="intervention"><h3>Predict the center after hiding names</h3><p>Illustrative 2D vectors. Keep row identity and numeric value 2 fixed.</p><div class="b07-pair"><div><strong>Baseline row</strong><p>volume: 2; country: France</p><p>center = [1, 0.5]</p></div><div><strong>Changed row</strong><p data-changed>volume: 2; country: France</p></div></div><label>Information arm<select><option value="meaningful">Meaningful headers</option><option value="anonymous">Anonymous headers</option><option value="numeric_only">Numeric features only</option></select></label><button type="button">Reset intervention</button><output aria-live="polite">Numeric leaf=[2,0]; text leaf=[0,1]. Mean center=[1,0.5]. Anonymous headers produce [0.36,0.94]; numeric-only produces [0.72,1.28]. These are illustrative vectors.</output></div>'''
adaptation='''<div class="b07-board" data-b07="adaptation"><h3>Audit the update and target contract</h3><label>Adaptation path<select data-model><option value="carte">CARTE / course probe</option><option value="contexttab" selected>ConTextTab context</option><option value="tabstar">TabSTAR adapters</option></select></label><label>Candidate-class input<select data-target><option value="all">Every class candidate</option><option value="answer">Only the true answer</option></select></label><button type="button">Reset contract</button><output aria-live="polite">ConTextTab changes context with frozen weights. Every class candidate may identify the task; the true query answer must remain hidden. TabSTAR learns adapters; the CARTE course probe fits a ridge head.</output></div>'''
def figure(name,caption):return '<figure class="b07-figure"><img src="../labs/figures/b07/'+name+'" alt="'+caption+'"><figcaption>'+caption+'</figcaption></figure>'
table='| Table | Meaningful | Anonymous | Numeric-only |\n|---|---:|---:|---:|\n'
for d in ['wine_pl','wine_dot_com_prices','wine_vivino_price']:
 vals=[next(x for x in r['aggregate'] if x['dataset']==d and x['arm']==a) for a in ['meaningful','anonymous','numeric_only']]
 table+='| '+d+' | '+' | '.join(f"{x['r2_mean']:.3f} ± {x['r2_sd']:.3f}" for x in vals)+' |\n'
table+='\nMean test R² ± sample SD across three paired split seeds. Full data: [independent audit](../labs/evidence/b07/course-audit.json).\n'
paired='| Table | Anonymous − meaningful | Numeric-only − anonymous |\n|---|---:|---:|\n'
for d in ['wine_pl','wine_dot_com_prices','wine_vivino_price']:
 vals=[x for x in r['paired'] if x['dataset']==d];paired+='| '+d+' | '+' | '.join(f"{x['mean']:+.3f} ± {x['sd']:.3f}" for x in vals)+' |\n'
paired+='\nMean paired R² change ± sample SD; positive favors the left arm.\n'
captions={'carte':'Course CARTE: visible row graph, edge-conditioned messages, frozen selected encoder and fitted ridge head.','contexttab':'ConTextTab paper concept: labeled support, hidden query target and alternating column/row attention. Original Table 2 run is source-gated.','tabstar':'TabSTAR paper architecture: numeric/text pair fusion, target candidates and six within-row interaction layers. No TabSTAR training executed here.'}
rep={'[[INTERVENTION]]':intervention,'[[ADAPTATION]]':adaptation,'[[RESULTS]]':table,'[[PAIRED]]':paired,'[[RESULT_FIGURE]]':figure('results.png','Measured test R²: three split-seed points per arm, black mean bars and common vertical scale.')}
for n in captions:rep['[['+n.upper()+']]']=figure(n+'.svg',captions[n])
def doc(title,body):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/semantic-transfer.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','semantic-transfer'])+'</body></html>'
text=(R/'lessons/content'/(S+'.md')).read_text()
for k,v in rep.items():text=text.replace(k,v)
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B07 — Semantic transfer: CARTE, ConTextTab and TabSTAR',render(text)))
reference='''# Semantic-transfer field guide

**Ask two questions:** what information enters, and what changes during adaptation?

| Model | Information path | Downstream adaptation |
|---|---|---|
| CARTE | Column edges + value leaves → row graph | Task fitting; B07 uses frozen encoder + ridge |
| ConTextTab | Type-specific cells + headers → column/row attention | Ordinary ICL changes labeled context |
| TabSTAR | Text/numeric pair fusion → features + target candidates | Gradient-based LoRA adapter training |

**Name ablation:** preserve values, targets, split identities and numerical preprocessing. Replace each header consistently. **Text removal:** also removes categorical/free-text predictors and changes graph size. **Candidate targets:** include all possible classes; never include only the query's actual answer.

**Worked trace:** numerical cell2×header(1,0)=(2,0); text leaf(0,1) with edge(0,1); mean edge-conditioned center=(1,.5). Anonymous toy header vectors(.6,.8) and(.8,.6) give(.36,.94). These are illustrative 2D vectors, not FastText outputs.

**Protocol:**3 real wine tables×3paired split seeds×3 arms.64 train/64 validation/256 test. Strict selected YAGO encoder frozen; train-only feature scaling; ridge alpha 1/10/100 selected on validation.27 fits/6,912 predictions. Digit labels 6..18,20,0; an all-missing numeric row uses a zero 300-vector. Original source rows and transformed targets retained.

**Scoring:** R²=1−SSE/SST; RMSE=sqrt(SSE/n). Pair effects by(table,seed). SD over splits is not a confidence interval. No broad significance claim from three related wine sources.

**Observed:** meaningful names help mean R² on wine_pl but hurt on the other two tables. Wine.com has two negative anonymous-name effects and one large positive effect. Text removal is also mixed. General semantic superiority NOT_ESTABLISHED.

**Evidence:** selected course experiment COMPLETE. Original ConTextTab v1 Table 2 INCOMPLETE_SOURCE_PROTOCOL; full pretraining/paper benchmark NOT_RUN. Current rename does not authenticate historical binning checkpoint or splits. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/b07-semantic-transfer.html) · [Student lab](../labs/b07-semantic-transfer.ipynb) · [Contract](../labs/b07-reproduction.md) · [CARTE](https://arxiv.org/html/2402.16785v2#S3) · [ConTextTab](https://arxiv.org/html/2506.10707v1#S3) · [TabSTAR](https://arxiv.org/html/2505.18125v2#S3).

Retrieve after 1/7/30 days following completion. Ask the agent to assess your written defense.
'''
(R/'reference/b07-semantic-transfer.html').write_text(doc('B07 · Semantic-transfer field guide',render(reference)))
# Archive inputs, primary sources and executable code, with deterministic ZIP metadata.
files={}
for name in ['_run_b07.py','_audit_b07.py','_source_b07.py','_reproduce_b07.py','_test_b07.py','_budget_b07.py','relkit/carte_b07.py','relkit/semantic_b07.py','b07-reproduction.md']:
 files['labs/'+name]=(P/name).read_bytes()
files['labs/relkit/__init__.py']=b''
for root in [P/'data/b07',P/'sources/b07']:
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:files['labs/'+str(p.relative_to(P))]=p.read_bytes()
for name in ['course-protocol.json','course-audit.json','source-gate.json']:files['labs/evidence/b07/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(files.items()):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
archive=buf.getvalue();(E/'reproducer.zip').write_bytes(archive)
md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
cells=[md('''# B07 · Semantic transfer

**One skill:** change semantic information while preserving a fair evaluation contract.

PROVIDED exposes the actual graph encoder and full 27-fit course loop; TODO holds three live functions; CHECK catches information/selection errors; EXIT needs your interpretation. Default: fresh frozen-encoder inference and27ridge fits, no cloud/GPU. Full paper pretraining and original ConTextTab Table 2 NOT_RUN. This is not a three-model benchmark.

The portable archive contains authenticated real wine tables, cached FastText vectors, selected YAGO tensors, primary-source evidence and canonical code. No checkout or network is needed when dependencies are installed. The supplied solution is author verification, not learner mastery. Live Colab NOT_CHECKED.

## Concept recap
An embedding turns an input into a numeric vector. Semantic transfer uses meaning learned elsewhere. Anonymous headers preserve text values; numeric-only removes them. Training rows fit the head; validation rows select its penalty; test rows only score it. Every allowed target class may be a candidate token, but the query's true class must stay hidden. ConTextTab uses labeled context with frozen parameters; TabSTAR trains adapters; B07's CARTE probe freezes its encoder and fits ridge.

Predict: must meaningful headers help every table? Write your answer before reading author results.
''')]
for n in captions:cells.append(md('## Model architecture · '+n.upper()+'\n\n![Architecture](data:image/png;base64,'+base64.b64encode((P/f'figures/b07/{n}.png').read_bytes()).decode()+')\n\n'+captions[n]))
cells.append(md('''**Trace before coding.** For a standardized volume2 and a France text leaf, illustrative header vectors(1,0),(0,1) give an edge-conditioned center(1,.5). Replacing headers by(.6,.8),(.8,.6) gives(.36,.94) even though cell values stay unchanged. The actual course uses300-dimensional FastText vectors. CARTE's12 heads have25 coordinates each. ConTextTab's toy3×3×768table distinguishes row and column attention. TabSTAR independently fuses each text/numeric pair, then mixes feature and target tokens with six 384-wide layers.

**Experiment contract:**3 real tables,3 split seeds,3 arms;64 train/64 validation/256 test. Targets remain in released transformed units. Alpha candidates1/10/100, choose lowest validation MSE without a validation refit. R²=1−SSE/SST; negative values are possible. For targets[1,3] and predictions[1,2], R²=.5. Removing text can create all-missing rows: retain them as zero 300-vectors.
'''))
versions=json.loads((E/'course-protocol.json').read_text())['versions']
setup='''# @colab-bootstrap: install only missing packages; warn on version drift.
import importlib.util, importlib.metadata, subprocess, sys, warnings
'''+f"versions={versions!r}\n"+'''for module,package in [('numpy','numpy'),('pandas','pandas'),('torch','torch'),('sklearn','scikit-learn'),('scipy','scipy'),('pyarrow','pyarrow')]:
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable,'-m','pip','install',package+'=='+versions[package]])
    elif importlib.metadata.version(package)!=versions[package]:
        warnings.warn(package+' version differs from author protocol; verify numerical tolerances')
import ast,base64,hashlib,io,json,math,statistics,zipfile,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.linear_model import Ridge
from scipy.stats import rankdata,friedmanchisquare,studentized_range
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
'''
cells.append(code(setup))
c=code('payload='+repr(base64.b64encode(archive).decode())+'\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=='+repr(hashlib.sha256(archive).hexdigest())+"\nworkspace=Path(tempfile.mkdtemp(prefix='b07-portable-'))\nwith zipfile.ZipFile(io.BytesIO(raw)) as bundle: bundle.extractall(workspace)\nlab=workspace/'labs'; evidence=lab/'evidence/b07'\nprint('Portable data/source package authenticated')");c.metadata['tags']=['data-payload'];cells.append(c)
def extract(path,names):
 text=(P/path).read_text();return '\n\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
for name,check,title,body in [('intervene','check_intervention','TODO 1 · Remove only the declared information','Implement the3 arms without mutating the input. Use fixed digit labels 6..18,20,0 by original feature slot. Numeric-only must not renumber surviving columns or remove rows.'),('fit_probe','check_selection','TODO 2 · Fit and select without test exposure','Fit a StandardScaler and each Ridge candidate on training rows only. Choose minimum validation MSE; return prediction, alpha, mean, scale, coefficient, intercept and all candidate validation_mse values. Reject overlapping identities. Never use test labels.'),('paired_delta','check_pairing','TODO 3 · Pair by complete identity','Inputs map(table,seed) to R². Require identical nonempty key sets. Return left-minus-right in sorted key order. Reject missing or different keys.')]:
 cells.append(md('## '+title+'\n\n'+body));cells.append(code(extract('relkit/semantic_b07.py',[name])));cells.append(code(extract('_test_b07.py',[check])+'\n'+check+'('+name+')\nprint("Learner contract PASS")'))
for names,title,body in [(['make_graph','grouped_attention'],'PROVIDED · Row graph and receiver-normalized attention','Read the graph indices: row0 receives; row1 supplies. Edge-conditioned messages retain column meaning. Softmax normalizes independently per receiving node.'),(['Attention','Readout','Encoder'],'PROVIDED · Actual transferred model','These definitions exactly match the inherited L074 selected architecture. Follow all300 coordinates through initial maps,12attention heads and the center readout. There is no hidden API model.'),(['load_encoder','batch_graphs','embed','normalize_numeric'],'PROVIDED · Weight transfer, batches and preprocessing','Strictly load every selected tensor. The parent checkpoint was authenticated before extraction. Fit numeric transforms on train only; preserve missing values and omit wholly unobserved training columns.')]:
 cells.append(md('## '+title+'\n\n'+body));cells.append(code(extract('relkit/carte_b07.py',names)))
cells.append(md('## PROVIDED · Empty rows preserve the evaluation population\n\nText removal may leave no observed leaves. Return a declared zero 300-vector for that row. Do not silently drop hard queries.'))
cells.append(code(extract('relkit/semantic_b07.py',['encode_table'])))
cells.append(md('## PROVIDED · Complete fresh27-fit loop\n\nThe function below accepts your live intervention and head functions. It never reads test labels to fit/select. Saved coefficients, embeddings and predictions make independent reconstruction possible.'))
cells.append(code(extract('_run_b07.py',['run_course'])))
cells.append(md('## RUN · Freeze your prediction, then train\n\nRecord which table you expect to benefit from names. This cell regenerates all 27 fits locally. Repeating it is a repeatability check, not new independent seeds.'))
cells.append(code('''calls={'intervention':0,'probe':0}
def counted_intervention(*args,**kwargs):
    calls['intervention']+=1
    return intervene(*args,**kwargs)
def counted_probe(*args,**kwargs):
    calls['probe']+=1
    return fit_probe(*args,**kwargs)
fresh=workspace/'fresh-runs'
result=run_course(lab,fresh,intervention=counted_intervention,probe=counted_probe)
assert calls=={'intervention':27,'probe':27}
print('All27 fits used your live functions')'''))
cells.append(md('## PROVIDED · Independent audit and source gate\n\nThe auditor uses scalar scoring and an independently solved dual ridge system. It rejects wrong keys, labels, selected alpha and missing runs. It does not import the fitted predictor. The source preflight separately authenticates original ConTextTab evidence.'))
cells.append(code(extract('_audit_b07.py',['audit_course'])+'\n\n'+extract('_source_b07.py',['audit_sources'])))
cells.append(code('''course=audit_course(lab,fresh)
source=audit_sources(lab/'sources/b07')
reference=json.loads((evidence/'course-audit.json').read_text())
assert course==reference, 'Fresh course report differs from author evidence; inspect protocol/environment'
assert source==json.loads((evidence/'source-gate.json').read_text())
for dataset in ['wine_pl','wine_dot_com_prices','wine_vivino_price']:
    for left,right in [('anonymous','meaningful'),('numeric_only','anonymous')]:
        lhs={(r['dataset'],r['seed']):r['r2'] for r in course['rows'] if r['dataset']==dataset and r['arm']==left}
        rhs={(r['dataset'],r['seed']):r['r2'] for r in course['rows'] if r['dataset']==dataset and r['arm']==right}
        actual=paired_delta(lhs,rhs)
        expected=next(r['seed_deltas'] for r in course['paired'] if r['dataset']==dataset and r['contrast']==left+' minus '+right)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
Path('b07-report.json').write_text(json.dumps({'course':course,'source':source},indent=2))
print(pd.DataFrame(course['aggregate']).to_string(index=False))
print(pd.DataFrame(course['paired'])[['dataset','contrast','mean','sd']].to_string(index=False))
print('COMPLETE course; original paper',source['status'])'''))
cells.append(md('## Author reference · measured split variability\n\n![Measured R2](data:image/png;base64,'+base64.b64encode((P/'figures/b07/results.png').read_bytes()).decode()+')\n\nBlack bars are means; points are split seeds. These are author reference results, not a claim that a blank student notebook has run.\n\n'+table.replace('Full data: [independent audit](../labs/evidence/b07/course-audit.json).','Full data: the portable course-audit.json and freshly generated b07-report.json.')+'\n\nMeaningful names help only wine_pl by mean R². Wine.com anonymous-name effects are negative on two splits and strongly positive on one. Numeric-only also has mixed effects. Three related domains, one arbitrary pseudonym scheme and a frozen probe do not establish general semantic-model superiority.'))
cells.append(md('''## EXIT · Defend the causal claim
1. Explain name removal versus text removal and why numeric-only cannot isolate semantic quality.
2. Show the exact(table,seed) paired effects, including disagreeing split signs.
3. Draw all-class target tokens without feeding the unknown query answer.
4. Contrast ConTextTab context adaptation with TabSTAR LoRA updates.
5. Propose two falsifiers and identify required source evidence before original Table 2 dispatch.

Your written defense is not auto-graded. Ask the agent for feedback; author checks do not establish learner mastery. Retrieve after 1/7/30 days following completion.'''))
cells.append(code("submission={'status':'PENDING_WRITTEN_DEFENSE','defense':''}\nPath('b07-submission.json').write_text(json.dumps(submission,indent=2))\nprint(submission['status'])"))
cells.append(md('''## NEXT STEP · Original published target, not a larger course run
B07-CONTEXTTAB-BAGGING targets originalv1 Table 2 binning base versus without bagging across the complete CARTE subset. Missing original binning checkpoint/configuration, full split identities and evaluator mean INCOMPLETE_SOURCE_PROTOCOL. The tested command below refuses dispatch. It is a source preflight, not a complete runnable paper trainer/evaluator. No Modal operator is supplied because dispatch cannot resolve missing scientific inputs. Full pretraining and original benchmark NOT_RUN.

Primary sources: [CARTE](https://arxiv.org/html/2402.16785v2#S3), [ConTextTab](https://arxiv.org/html/2506.10707v1#S5.T2), [TabSTAR](https://arxiv.org/html/2505.18125v2#S3), [SAP model card](https://huggingface.co/SAP/sap-rpt-1-oss). Archived source files and licenses are in the portable source archive. Course does not implement/train TabSTAR or ConTextTab from scratch; it implements and executes the selected CARTE probe.
'''))
cells.append(code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    subprocess.run([sys.executable,str(lab/'_reproduce_b07.py'),'--run'],check=True)\nelse:\n    print('Original paper NOT_RUN: source gate',source['status'])"))
solution=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
for i,c in enumerate(solution.cells):c.id=f'b07-{i:03d}'
student=copy.deepcopy(solution)
for c in student.cells:
 if c.cell_type=='code' and c.source.startswith(('def intervene(','def fit_probe(','def paired_delta(')):
  node=ast.parse(c.source).body[0];signature=c.source.split('\n',1)[0];c.source=signature+'\n    raise NotImplementedError("TODO: '+node.name+'")'
nb.write(student,P/(S+'.ipynb'));nb.write(solution,P/'solutions'/(S+'.ipynb'))
print('Built',len(cells),'cells; archive',len(archive),'bytes')
