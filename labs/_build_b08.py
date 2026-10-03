"""Build B08 lesson/reference and self-contained student/solution notebooks."""
import ast,base64,copy,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b08';S='b08-structured-objectives'
r=json.loads((E/'course-audit.json').read_text())
def figure(name,caption,wide=False):return f'<figure class="b08-figure{" b08-wide" if wide else ""}"><img src="../labs/figures/b08/{name}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>'
widgets={
'SAMPLE_WIDGET':'''<div class="b08-board" data-b08="sample"><h3>Does another query change this readout?</h3><p>Equal-score attention on scalar value states. Baseline support values: 2 and 6.</p><label><input type="checkbox"> Incorrectly permit all query keys</label><label>Q2 value state: <strong data-value>30</strong><input type="range" min="10" max="30" step="10" value="30"></label><div data-matrix></div><button type="button">Reset sample access</button><output aria-live="polite">Legal sample matrix: all four rows read S1 and S2 only. Readout=(2+6)/2=4. Query states 10 and 30 are excluded.</output></div>''',
'FEATURE_WIDGET':'''<div class="b08-board" data-b08="feature"><h3>Which representation can the task read?</h3><p>Hold feature states 2, 6 and task state 20 fixed. T summarizes four task slots.</p><label><input type="checkbox"> Incorrectly permit task-to-task attention</label><div data-matrix></div><button type="button">Reset feature access</button><output aria-live="polite">Feature rows read X1,X2,T. Task row T reads X1,X2 only; equal-score task readout=4.</output></div>''',
'LOSS_WIDGET':'''<div class="b08-board" data-b08="loss"><h3>Keep the input; change what is scored</h3><p>Truth [2,5,9], predictions [1,7,8]. Only the first two cells were hidden. Target squared error=1.</p><label>Reconstruction weight λ: <strong data-value>1</strong><input type="range" min="0" max="2" step="0.5" value="1"></label><label><input type="checkbox"> Incorrectly score the observed third cell</label><button type="button">Reset loss example</button><output aria-live="polite">Scored-only Lx=(1+4)/2=2.5. At λ=1, Ly+λLx=3.5. The observed third cell has zero reconstruction gradient.</output></div>'''}
table='| Objective | Target MSE ↓ | Feature MSE ↓ |\n|---|---:|---:|\n'
for row in r['summary']:table+=f'| {row["objective"]} | {row["target_mse"]:.4f} ± {row["target_mse_sd"]:.4f} | {row["feature_mse"]:.4f} ± {row["feature_mse_sd"]:.4f} |\n'
by=sum(x['target_mse'] for x in r['baselines'])/3;bx=sum(x['feature_mse'] for x in r['baselines'])/3
text=f'''Values are **mean ± sample SD across three seeds**, not confidence intervals. Each seed includes 128 target predictions and 128 scored feature cells per arm. All 9 arms  / 1,152 target predictions  / 1,152 scored feature predictions passed independent scoring. The same inputs and initial weights are paired across arms.

Combined-minus-target-only target MSE is **+0.00384, −0.04761, +0.29332** for seeds 0, 1, 2. Only seed 1 improves. Mean target MSE rises from 0.6704 to 0.7536. Reconstruction-only has the lowest mean feature MSE, but its target head is untrained.

The support-mean baseline has target MSE**{by:.4f}** and feature MSE**{bx:.4f}** averaged across the same three seeds. Combined reconstruction does not beat that feature baseline in every seed. This brief training recipe provides mixed evidence, not a general benefit for joint modeling. A seed bundles initialization and episode streams; its SD includes both. Three seeds from one synthetic family do not justify real-dataset ranks or a model-family ranking. No post-result retuning was performed.
'''
rep={**widgets,'RESULTS':table,'DISCUSSION':text,
'LIMIX16M':figure('limix16m.svg','Original LimiX-16M paper architecture. The Table 23 target consumes the feature output; this is not the course model.'),
'LIMIX2':figure('limix2.svg','LimiX-2 forward path. Four task slots concatenate across rows; shallow feature and final task heads answer different questions.'),
'MASKS':figure('masks.png','Portable access trace: rows are readers, columns are keys. T summarizes the four task slots.'),
'RESULT_FIGURE':figure('results.png','Measured target and feature MSE. Lines connect the same seed; dashed lines are mean support-only baselines. Panel scales differ.'),
'PAIRED':figure('paired.png','Measured target effect for each paired seed. Positive means reconstruction worsened target MSE.')}
def doc(title,body):return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/structured-objectives.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','structured-objectives'])+'</body></html>'
source=(R/'lessons/content'/(S+'.md')).read_text()
for key,value in rep.items():source=source.replace('[['+key+']]',value)
assert '[[' not in source
(R/'lessons'/(S+'.html')).write_text(doc('B08 · Structured-data objectives',render(source)))
reference='''# B08 · Structured-objective field guide

**Ask three separate questions:** What input is hidden? Which attention edges are allowed? Which output errors are scored?

| Axis | Allowed access | Forbidden access |
|---|---|---|
| Rows | Support→support; query→support | Support→query; query→query |
| Columns within a row | Feature→features/tasks; task→features | Task→tasks |

Arrows here mean **reader→source**. Own query features survive through residual connections. Hidden query targets are replaced before embedding; masked features retain column identity. Batch invariance requires fixed preprocessing and context as well as correct attention.

**MSE:** mean squared prediction error. **RMSE:** square root of MSE. Lx averages only scored masked features; Ly averages query target errors. Combined course loss=Ly+Lx. Example: feature errors[1,4,1], first two hidden →Lx=2.5; Ly=1 →total3.5. The observed third cell has zero feature-loss gradient.

**Release map:** LimiX-16M=original selected imputation target; LimiX-2M=smaller original-family model; LimiX-2=new400M-class release,24 blocks,four task slots and5000-bin regression. Course=2 blocks,width 16,scalar heads. It preserves selected access mechanisms, not full architecture or checkpoint parity.

**Readout depth:** LimiX-2 reconstructs from shallow feature states and predicts the designated target from final task states. Shared gradients need not improve both tasks. Feature-only leaves this course’s final target head untrained.

**Observed course result:** nine fits; combined target MSE 0.7536 versus target-only0.6704. Seed differences+0.00384,−0.04761,+0.29332; one improves. Same synthetic generator, no real-dataset ranking. Paper Table 23 Analcatdata RMSE 0.194 remains INCOMPLETE_SOURCE_PROTOCOL. Full pretraining and other benchmarks NOT_RUN. Author checks do not establish learner mastery or causal identification.

**Reproduction checklist:** pin paper/code/weights separately; authenticate dataset version, complete row/cell keys, split, masks, scaler, repeats and metric. Preserve immutable predictions. Matching a rounded score is insufficient.

[Lesson](../lessons/b08-structured-objectives.html) · [Lab](../labs/b08-structured-objectives.ipynb) · [Protocol](../labs/b08-reproduction.md) · [LimiX-2 §§2–3](https://arxiv.org/html/2609.17488v1#S2) · [Original Table 23](https://arxiv.org/html/2509.03505v2#S7.SS3).

Ask the agent to assess your explanation. Retrieve the two masks from memory after 1, 7 and 30 days following completion.
'''
(R/'reference'/f'{S}.html').write_text(doc('B08 · Field guide',render(reference)))
# Portable archive contains frozen evidence, source pins, runnable code and data.
paths=[]
for base in [P/'data/b08',P/'sources/b08',E/'runs']:
 paths.extend(p for p in base.rglob('*') if p.is_file() and p.suffix!='.gz' and '__pycache__' not in p.parts)
paths.extend(E/x for x in ['course-protocol.json','course-audit.json','source-gate.json'])
paths.extend(P/x for x in ['_run_b08.py','_audit_b08.py','_reproduce_b08.py','_verify_b08.py','_test_b08.py','_model_test_b08.py','_budget_b08.py','relkit/limix_b08.py','b08-reproduction.md'])
blob=io.BytesIO()
with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(paths):
  info=zipfile.ZipInfo(str(p.relative_to(P)),date_time=(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
payload=blob.getvalue();(E/'reproducer.zip').write_bytes(payload)
md=nb.v4.new_markdown_cell
code=lambda s,tags=None:nb.v4.new_code_cell(s,metadata={'tags':tags or []})
cells=[md('''# B08 · Trace the evidence, then change the objective

Skill: implement both attention masks and a scored-only loss; explain a complete nine-fit objective ablation.

**PROVIDED** cells expose the model and trainer. **TODO** functions are live: the model calls them. **CHECK** cells give immediate feedback. **EXIT** requires your written defense as well as numerical evidence. Student blanks are intentional; use the separate solution only after attempting them.

This notebook runs offline after dependency setup. It includes frozen arrays and the pinned source packet. Fresh course training takes seconds to minutes on a CPU. No paid service is called. Full historical Table 23 inference is source-gated and full pretraining remains NOT_RUN.

[Lesson](https://avist.github.io/relational/lessons/b08-structured-objectives.html) · [LimiX-2 §§2–3](https://arxiv.org/html/2609.17488v1#S2) · [Original Table 23](https://arxiv.org/html/2509.03505v2#S7.SS3)
'''),md('''## Concept recap and worked trace

Support rows have known features X and target y. Query rows have observed features and an unknown target. A hidden feature is replaced by a missing embedding plus a column code. Its true value belongs to the scorer, not the encoder.

Across rows, every reader sees support keys only. Support cannot read queries either, preventing query→support→query leakage. Within a row, feature queries see feature and task states; task queries see features only. Four task slots represent one target, not four labels. An attention query/key/value denotes a retrieval role, not necessarily a dataset query row.

Equal scores give equal weights: value states 2 and 6 average to4. Letting query states 10 and 30 join changes that average to12. These are illustrative internal states, not labels. A residual connection preserves the receiving row’s own feature state.

MSE averages squared errors; RMSE is its square root. True features [2,5,9], predicted[1,7,8], first two hidden: feature loss Lx=(1+4)/2=2.5. Target truth 4,prediction 3 gives Ly=1. Combined course loss Ly+Lx=3.5. Only scored cells receive direct reconstruction gradients.

The complete course experiment holds inputs, masks, initialization, architecture, optimizer and120 steps fixed within each seed. Three objectives×three seeds = 9 fits. Exactly one feature per query is hidden. Train/test are independent streams of the same synthetic generator; no real-data transfer claim follows. Feature-only leaves the final target head untrained. Sum loss changes gradient magnitude as well as supervision.
''')]
for name,title,caption in [('limix16m','Model architecture · LimiX-16M','Original selected-target model:12 blocks,cell embeddings and feature reconstruction. Original model/encoder/layer source is visible in the appendix.'),('limix2','Model architecture · LimiX-2','24 blocks,width 256,four task slots,independent pathways. Course uses2 blocks,width 16,single-head attention,LayerNorm and scalar heads; released model uses RMSNorm,attention normalization/scaling,adapters and5000-bin regression.'),('masks','Two access matrices','Rows read columns; T compresses four task slots. Own query inputs survive via residuals; preprocessing must also preserve the claimed visibility.')]:
 cells.append(md('## '+title+'\n\n'+caption+'\n\n<img alt="'+caption+'" width="'+'440'+'" src="data:image/png;base64,'+base64.b64encode((P/'figures/b08'/f'{name}.png').read_bytes()).decode()+'">'))
versions=json.loads((E/'course-protocol.json').read_text())['versions']
cells.append(code('''# @colab-bootstrap: install only missing packages; report drift explicitly.
import importlib.util, importlib.metadata, subprocess, sys, warnings, os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
expected_versions = '''+repr(versions)+'''
for package in ['numpy','torch']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable,'-m','pip','install',package+'=='+expected_versions[package]])
    actual=importlib.metadata.version(package)
    if actual != expected_versions[package]: warnings.warn(f'{package}: {actual}; author used {expected_versions[package]}')
import numpy as np
import torch
from torch import nn
import hashlib, json, time
from pathlib import Path
print('CPU course run; no paid service. Numerical drift is reported, not hidden.')
'''))
cells.append(code('''# PROVIDED: authenticate and extract the portable packet, never overwrite evidence.
import base64, io, zipfile, tempfile
encoded = '''+repr(base64.b64encode(payload).decode())+'''
raw = base64.b64decode(encoded)
assert hashlib.sha256(raw).hexdigest() == '''+repr(hashlib.sha256(payload).hexdigest())+'''
packet = Path(tempfile.mkdtemp(prefix='b08-packet-'))
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for item in archive.infolist():
        destination=(packet/item.filename).resolve()
        assert destination.is_relative_to(packet.resolve()), 'Archive traversal'
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(archive.read(item))
protocol=json.loads((packet/'evidence/b08/course-protocol.json').read_text())
for name,digest in protocol['data_files'].items():
    assert hashlib.sha256((packet/name).read_bytes()).hexdigest()==digest
print('Frozen inputs authenticated; originals remain immutable.')
''',['data-payload']))
src=(P/'relkit/limix_b08.py').read_text();tree=ast.parse(src);nodes={node.name:ast.get_source_segment(src,node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
tasks=[('sample_visibility','rows, support','Build a Boolean matrix restricting every reader to the nonempty support prefix. Reject invalid support sizes.','assert sample_visibility(4,2).tolist()==[[True,True,False,False]]*4'),('feature_visibility','features, slots','Build the within-row allowed-access matrix, including all four task slots. Do not confuse reader and source axes.','assert feature_visibility(2,2).tolist()==[[True]*4,[True]*4,[True,True,False,False],[True,True,False,False]]'),('masked_mse','prediction, truth, mask','Average error only over scored cells. Reject empty or misaligned masks. An unscored prediction must have zero loss gradient.','p=torch.tensor([1.,100.,5.],requires_grad=True)\nt=torch.tensor([3.,-100.,4.]);m=torch.tensor([True,False,True])\nl=masked_mse(p,t,m);assert l.item()==2.5\nl.backward();assert p.grad.tolist()==[-2.,0.,1.]')]
solution_sources={}
for name,args,goal,check in tasks:
 cells.append(md('## TODO · '+name+'\n\n**Goal.** '+goal+'\n\n**Why.** This function controls actual information access or gradients in every training arm. Describe the desired rule first; the check is only a small example.'))
 idx=len(cells);cells.append(code(f'def {name}({args}):\n    # TODO: implement the contract described above.\n    raise NotImplementedError("Complete {name}")',['student-todo']));solution_sources[idx]=nodes[name]
 cells.append(code('# CHECK: '+name+'\n'+check+'\nprint("PASS: '+name+'")'))
for names,heading,explain in [(['Attention','SwiGLU'],'Attention and gated transformation','Scaled dot-product attention computes softmax(QKᵀ/√d)V. Disallowed logits become−∞ before softmax. A learned gate multiplies the value transformation.'),(['DualAxis'],'One dual-axis block','Sample-axis maps are separate for X and concatenated task slots. Independent SwiGLU precedes asymmetric within-row attention. The two feature-axis updates use the same pre-update state.'),(['CourseModel'],'Complete course forward pass','Replace hidden values before encoding. Add rank4 column codes. Two blocks route information; block1 emits feature predictions and block2 emits target predictions. This is a declared course architecture, not released-weight parity.'),(['make_episodes','state_hash'],'Episodic generator and identity','A latent z generates correlated features; each episode draws target coefficients. Independent seed ranges separate train and test. Fixed feature masks are shared across objectives.'),(['train_arm'],'Complete course training and inference','Each arm resets the model seed and uses AdamW for120 steps. Only the objective changes. Every test episode is evaluated; no checkpoint selection occurs.')]:
 cells.append(md('## PROVIDED · '+heading+'\n\n'+explain));cells.append(code('\n\n'.join(nodes[n] for n in names)))
cells.append(md('## CHECK · try to leak truth or couple queries\n\nCorrupt masked features and hidden query labels; predictions must not change. Then alter another query’s features. Finally perturb support labels: the target prediction should be allowed to change. These interventions verify an information contract.'))
check=(P/'_model_test_b08.py').read_text().replace('from relkit.limix_b08 import CourseModel','').replace("if __name__=='__main__':unittest.main()",'')
cells.append(code(check+'\nresult=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Visibility))\nassert result.wasSuccessful()'))
cells.append(md('## Predict, then run all nine arms\n\nWrite your prediction about combined-minus-target target MSE before running. Keep all seeds even if one looks poor. The output compares fresh arrays with saved author arrays; any drift is surfaced.'))
cells.append(code('''prediction_before_run = ""  # Write a prediction; this is learner work.
print('Prediction recorded' if prediction_before_run.strip() else 'PENDING prediction explanation')
records=[];fresh_metrics=[]
fresh_dir=Path(tempfile.mkdtemp(prefix='b08-fresh-'))
for seed in protocol['seeds']:
    train=dict(np.load(packet/f'data/b08/train-{seed}.npz'))
    test=dict(np.load(packet/f'data/b08/test-{seed}.npz'))
    for objective in protocol['objectives']:
        arrays,record=train_arm(train,test,seed,objective)
        reference=np.load(packet/f'evidence/b08/runs/{objective}-{seed}.npz')
        for key,value in arrays.items():
            np.testing.assert_allclose(value,reference[key],rtol=0,atol=0,err_msg='Fresh prediction drift')
        yerror=arrays['y_pred'][:,:,24:].astype('float64')-test['y'][:,:,24:]
        xerror=arrays['x_pred'][test['hidden']].astype('float64')-test['x'][test['hidden']]
        fresh_metrics.append(dict(seed=seed,objective=objective,target_mse=float(np.mean(yerror**2)),feature_mse=float(np.mean(xerror**2))))
        np.savez_compressed(fresh_dir/f'{objective}-{seed}.npz',**arrays)
        records.append(record)
        print(seed,objective,'fresh predictions match exactly')
for seed in range(3):
    assert len({r['initial_sha256'] for r in records if r['seed']==seed})==1
from IPython.display import display,HTML
table='<table><tr><th>Seed</th><th>Objective</th><th>Target MSE</th><th>Feature MSE</th></tr>'
for row in fresh_metrics:
    table+=f"<tr><td>{row['seed']}</td><td>{row['objective']}</td><td>{row['target_mse']:.4f}</td><td>{row['feature_mse']:.4f}</td></tr>"
display(HTML(table+'</table>'))
'''))
cells.append(md('## Author-reference evidence · interpret after your run\n\n'+table+'\n\n'+text+'\n\n<img width="510" alt="Measured author course evidence" src="data:image/png;base64,'+base64.b64encode((P/'figures/b08/results.png').read_bytes()).decode()+'">'))
cells.append(md('## NEXT STEP · selected paper result\n\nThe target is original Table 23 Analcatdata BroadwayMult,normalized RMSE 0.194 at 5% masking. Both archived demos use breast-cancer data and 30% masking. Original row/mask/scaler/repetition/checkpoint identities are not authenticated. Current BCCO BroadwayMult train/test candidate bytes are pinned, but the 5% mask and their historical Table 23 identity are not authenticated. Current full released inference is provided but NOT_RUN; historical dispatch fails closed. Read `b08-reproduction.md` in the packet for the authenticated numeric-packet interface. Course heads and masks do not substitute for the released model. Full pretraining trainer/data are unavailable in this packet.'))
cells.append(code('''# PROVIDED: authenticate source and show the real historical blocker.
sys.path.insert(0,str(packet))
from _reproduce_b08 import source_audit
source_status=source_audit()
print(source_status['status'])
for blocker in source_status['blockers']:print('•',blocker)
RUN_PAPER_REPRO=False
if RUN_PAPER_REPRO:
    result=subprocess.run([sys.executable,str(packet/'_reproduce_b08.py'),'--lane','paper'])
    assert result.returncode==2, 'Historical gate unexpectedly bypassed'
'''))
cells.append(md('## EXIT · numerical output plus written defense\n\nExplain both attention axes, why feature-only has an untrained target head, all three paired target effects, and the missing evidence for Table 23. Numerical checks alone do not mark your learning complete. Paste your explanation to the agent for feedback.'))
cells.append(code('''written_defense = ""  # Learner writes this; author leaves it blank.
submission=dict(status='PENDING_WRITTEN_DEFENSE',course_arms=len(records),source=source_status['status'],written_defense=written_defense)
Path('b08-submission.json').write_text(json.dumps(submission,indent=2))
report=dict(fresh=True,metrics=fresh_metrics,source=source_status['status'],all_prediction_arrays='EXACT',learner=submission['status'])
Path('b08-report.json').write_text(json.dumps(report,indent=2))
print(submission)
'''))
cells.append(md('## Appendix · full original release core, readable without imports\n\nThe following archived source is displayed for inspection, not executed as course code. It comes from publication-era commitc6f677f8e884e86b7638e0cda978bbccf7b45e1a. The complete historical snapshot and current release are also in the authenticated packet. Map embeddings→encoder; attention/residuals→layer; backbone/heads→transformer; preprocessing/inference→predictor and inference_method. Original pretraining is not supplied or claimed. The selected inference operator uses the separately pinned current release.'))
for name in ['model/encoders.py','model/layer.py','model/transformer.py','inference/predictor.py','inference/inference_method.py']:
 raw=(P/'sources/b08/historical'/name).read_text();parsed=ast.parse(raw);lines=raw.splitlines();starts=[node.lineno-1 for node in parsed.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))];bounds=sorted(set([0]+starts+[len(lines)]))
 cells.append(md('### Archived source · '+name+'\n\nSHA256 `'+hashlib.sha256(raw.encode()).hexdigest()+'`. Author release code; dependencies and licenses are in the packet.'))
 for lo,hi in zip(bounds,bounds[1:]):
  if hi>lo:cells.append(md('```python\n'+'\n'.join(lines[lo:hi])+'\n```'))
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nb.write(book,P/(S+'.ipynb'));sol=copy.deepcopy(book)
for idx,source in solution_sources.items():sol.cells[idx].source=source;sol.cells[idx].metadata['tags']=['teacher-solution']
# Retain outputs only when every executed code byte is unchanged (prose-only rebuild).
oldpath=P/'solutions'/(S+'.ipynb')
if oldpath.exists():
 old=nb.read(oldpath,4);oc=[c for c in old.cells if c.cell_type=='code'];nc=[c for c in sol.cells if c.cell_type=='code']
 if [c.source for c in oc]==[c.source for c in nc] and all(c.execution_count is not None for c in oc):
  for before,after in zip(oc,nc):after.outputs=before.outputs;after.execution_count=before.execution_count
nb.write(sol,oldpath)
if all(c.execution_count is not None for c in sol.cells if c.cell_type=='code'):
 from nbconvert import HTMLExporter
 from nbconvert.preprocessors import TagRemovePreprocessor
 exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
 exported,_=exporter.from_notebook_node(sol);(P/'html'/(S+'.html')).write_text(exported)
print('Built lesson/reference and',len(book.cells),'notebook cells; portable archive',len(payload),'bytes')
