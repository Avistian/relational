"""Rebuild L174 narrative, reference, report and portable notebooks from evidence."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l174';D=E/'runs';S='0174-fine-tuning-protocol'
r=json.loads((D/'report.json').read_text());baselines=json.loads((D/'baselines.json').read_text());arms=['freeze','full','adapter','scratch']
def results():
 lines=['**Measured: all twelve fits complete.** Lower macro task loss is better; every entry uses all 106,757 test targets.','','| Arm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |','|---|---:|---:|---:|---:|']
 for arm in arms:
  v=[x['test']['macro_loss'] for x in r['runs'] if x['arm']==arm];lines.append('| '+arm+' | '+' | '.join(f'{n:.6f}' for n in v)+f' | {np.mean(v):.6f} ± {np.std(v,ddof=1):.6f} |')
 v=[x['test']['macro_loss'] for x in baselines if x['name']=='unchanged'];lines.append('| Unchanged source | '+' | '.join(f'{n:.6f}' for n in v)+f' | {np.mean(v):.6f} ± {np.std(v,ddof=1):.6f} |')
 lines+=['',f"2006 training-only constant baseline: **{baselines[-1]['test']['macro_loss']:.6f}**. Parameter counts: freeze 9,834; full 28,426; adapter 10,386 (including 552 new); scratch 28,426."]
 return '\n'.join(lines)
vals={a:np.array([x['test']['macro_loss'] for x in r['runs'] if x['arm']==a]) for a in arms}
delta=vals['adapter']-vals['freeze'];scratch=vals['scratch']-vals['full']
interpretation='**Measured conclusion:** scratch has lower test macro loss than every pretrained adaptation arm for each of the three seeds. All pretrained adaptation arms improve on their unchanged source checkpoints, yet remain worse than the constant baseline. These runs do not establish a benefit from pretraining. Adapter minus freeze averages '+f'{delta.mean():+.6f}'+', with paired differences '+', '.join(f'{v:+.6f}' for v in delta)+'. Scratch minus full averages '+f'{scratch.mean():+.6f}'+'. This is a descriptive result for this exposed F1 population and fixed schedule, not a universal ranking of adaptation methods.'
captions={'architecture':'Actual L174 computation and update boundaries. The optional 32→8→32 residual adapter sits between the shared encoder and the original task heads.','parameters':'Trainable and total parameters for the actual course model. Existing heads dominate the adapter arm; parameter count is not a runtime measurement.','curves':'Measured three-seed validation curves (mean ± sample SD) and complete test losses. Open circles mark selected epochs. The test control includes unchanged checkpoints and a 2006 constant baseline.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',results()).replace('[[INTERPRETATION]]',interpretation)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l174'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l174/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0" style="overflow-x:auto"><img style="min-width:740px;width:100%" src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 for tag,text,html in [('WARMUP','Recall: why are target masking and training-only preprocessing separate requirements? What did the L173 constant baseline reveal?','<div id="warmup"></div>'),('PREDICT','Predict: with a zero up projection, which adapter projection can receive a nonzero weight gradient on the first step? Work through the chain rule before reading the answer.','<div id="predict"></div>'),('POLICY_EXPLORER','Intervene on the update policy: freeze trains9,834 parameters; full 28,426; adapter 10,386; scratch 28,426. Which tensors stay byte-identical?','<div id="policy-explorer"></div><noscript>Freeze:9,834 trainable; full:28,426; adapter:10,386; scratch:28,426. In freeze and adapter arms the 18,592 encoder parameters stay fixed.</noscript>'),('TEACHBACK','Write your defense before consulting teacher feedback.','<div id="teachback"></div>')]:s=s.replace('[['+tag+']]',text if portable else html)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','finetune-policy'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','finetune-policy','l174-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0173-multi-task-pretraining.html">Lesson 173</a></nav><header><p class="route-kicker">Year 5 · Quarter 2 · Lesson 174</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Fine-tuning protocol',prose(),True))
reference='''**Decision:** mark trainable tensors before constructing the optimizer. Verify both `requires_grad` and saved parameter changes. `eval()` does not freeze weights.

| Policy | Encoder | Heads | Added path |
|---|---|---|---|
| Freeze | Fixed | Train | None |
| Full | Train | Train | None |
| Adapter | Fixed | Train | Train residual bottleneck |
| Scratch | Random, train | Random, train | None |

**Adapter:** h′=h+U ReLU(Dh+b)+c, 32→8→32. Random D, zero U/c gives identity initialization. The first D gradient is zero; U can learn.552 new parameters;10,386 total trainable including heads.

**Fair comparison:** same input population, preprocessing, seed pairing, example order, loss weighting, update budget and validation selection. Report actual runtime too; same epochs do not mean equal compute. A common learning rate is not per-arm optimal tuning.

**Controls:** unchanged source = improvement from adaptation; scratch = benefit from pretrained initialization under the fixed schedule; training-only constant = simple baseline. Do not select the source or adaptation checkpoint on test results.

**Frozen contract:** L173 cell checkpoints selected on2005 validation; adapt2006; validate2007; score2008+; all21tasks; seeds0/1/2; ten epochs; Adam.001; batch1024; equal-task weights from2006. Existing head weights retained except scratch. Full population; no subsampling.

'''+results()+'\n\n'+interpretation+'''

**Evidence boundary:** prior L173 test exposure; same-task autocomplete; whole-paper reproduction NOT_RUN; cross-database transfer and historical availability NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.

Primary sources: [Houlsby et al.](https://proceedings.mlr.press/v97/houlsby19a.html), [RT v1 §4.2](https://arxiv.org/html/2510.06377v1#S4.SS2). [Lesson](../lessons/0174-fine-tuning-protocol.html) · [Student notebook](../labs/0174-fine-tuning-protocol.ipynb) · [Protocol](../labs/l174-reproduction.md).
'''
(R/'reference/fine-tuning-protocol.html').write_text(doc('Fine-tuning — quick reference',reference))
lines=['# L174 measured temporal adaptation',results(),interpretation,'','## Every task, reported separately','Mean across three seeds. Raw MAE and accuracy have separate units; they are not pooled.','','| Task | Metric | Freeze | Full | Adapter | Scratch | Constant |','|---|---|---:|---:|---:|---:|---:|']
for i,s in enumerate(json.loads((E/'tasks.json').read_text())):
 metric='mae_raw' if s['kind']=='number' else 'accuracy'
 v=[np.mean([x['test']['tasks'][i][metric] for x in r['runs'] if x['arm']==arm]) for arm in arms]
 lines.append('| '+s['name']+' | '+metric+' | '+' | '.join(f'{x:.6f}' for x in v)+f' | {baselines[-1]["test"]["tasks"][i][metric]:.6f} |')
lines+=['','## Per-task losses','| Task | Freeze | Full | Adapter | Scratch | Constant |','|---|---:|---:|---:|---:|---:|']
for i,s in enumerate(json.loads((E/'tasks.json').read_text())):
 v=[np.mean([x['test']['tasks'][i]['loss'] for x in r['runs'] if x['arm']==arm]) for arm in arms]
 lines.append('| '+s['name']+' | '+' | '.join(f'{x:.6f}' for x in v)+f' | {baselines[-1]["test"]["tasks"][i]["loss"]:.6f} |')
lines+=['','## Measured fit runtime','Wall seconds include checkpoint writes, validation and selected test inference. These are operational observations, not deterministic metrics.','| Arm | Seed0 | Seed1 | Seed2 |','|---|---:|---:|---:|']
for arm in arms:lines.append('| '+arm+' | '+' | '.join(f'{x["seconds"]:.3f}' for x in r['runs'] if x['arm']==arm)+' |')
lines+=['','Raw per-seed losses, metrics, trainability, selection and histories are in report.json. Every epoch is saved under its arm/seed directory.','Whole-paper reproduction NOT_RUN; test exposure RETROSPECTIVE_L173; learner PENDING_WRITTEN_DEFENSE.']
(D/'report.md').write_text('\n'.join(lines)+'\n')
# Portable packet contains all targets and the three actual selected source checkpoints.
m=json.loads((E/'manifest.json').read_text());names=sorted(list(m['files'])+['manifest.json'])
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in names:
  info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(E/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
contracts={'configure_trainable':('Set the update boundary','Validate the arm (freeze/full/adapter/scratch). Set every parameter’s requires_grad flag: all original parameters for full/scratch; only heads for freeze; heads and adapter for adapter. The optimizer will use this live policy.'),'adaptation_split':('Separate adaptation, selection and evaluation','Input UTC seconds for each target identity. Return -1 before2006,0 during2006,1 during2007,2 from2008 onward. Do not index the padded date lookup as though it were the target array.'),'select_epoch':('Select without test access','Return the zero-based earliest finite validation minimum. Reject an empty list or any nonfinite value. Test scores must never enter this function.')}
source_paths=['relkit/multitask_l173.py','relkit/finetune_l174.py','_check_l174.py']
functions=[]
for filename in source_paths:
 source=(P/filename).read_text()
 for node in ast.parse(source).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and not (filename=='relkit/multitask_l173.py' and node.name in {'erase_target','load_packet','train_run'}):functions.append((node.name,ast.get_source_segment(source,node)))
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 174 · Fine-tuning protocol\n\n**One skill:** implement and defend the parameter-update boundary. The narrative contains saved author evidence. Executable cells below perform a separate fresh adapter/seed 0 fit. Full paper reproduction NOT_RUN; this is retrospective F1 autocomplete.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_markdown_cell('## PROVIDED · Runtime and visible code\nRequires NumPy, PyTorch and matplotlib. The author runtime is recorded below. No cloud service, account or network is needed after downloading the notebook. Model definitions are inlined; no hidden package imports implement the experiment.'),nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport hashlib,json,time,tempfile,base64,io,zipfile\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nimport matplotlib.pyplot as plt\ntorch.set_num_threads(1)\nprint("Author versions: numpy '+r['numpy']+', torch '+r['torch']+'; other versions may differ numerically.")')]
 for name,code in functions:
  if name in contracts:
   title,contract=contracts[name];cells.append(nb.v4.new_markdown_cell('## TODO · '+title+'\n\n'+contract));cells.append(nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
  else:
   descriptions={'MaskedCellModel':'Inherited encoder: typed tokens, separate row/FK pools, shared MLP and local heads.','ResidualAdapter':'The actual residual 32→8→32 module. Trace the zero up projection before training.','AdaptationModel':'Same computation as L173 with the optional adapter immediately before the original heads.','train_adaptation':'The actual ten-epoch trainer. It calls your live policy, splitter and selector; saves every epoch and refuses existing output directories.','check174':'Behavioral checks perturb trainability, exercise actual optimizer updates and test the temporal boundary.'}
   cells.append(nb.v4.new_markdown_cell('## PROVIDED · '+name+'\n\n'+descriptions.get(name,'Visible support code used by the complete experiment.')));cells.append(nb.v4.new_code_cell(code))
 cells+=[nb.v4.new_markdown_cell('## CHECK · Exercise your live functions\nThe tests perform actual updates, require an identity adapter, check frozen weights and test exact year boundaries and selection ties.'),nb.v4.new_code_cell('assert check174(configure_trainable,adaptation_split,select_epoch,AdaptationModel)=="PASS"\nprint("CHECK: update policy, adapter learning, split and selection PASS")'),nb.v4.new_markdown_cell('## PROVIDED · Authenticated portable packet\nAll 370,024 original targets and their permitted contexts, inherited preprocessing and three selected source checkpoints. Fresh adaptation uses the full 6,429/6,074/106,757 split. Hashes detect corruption; the repository verifier independently reconstructs the population from original tables. Hashes alone do not prove historical availability.')]
 cell=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nROOT=Path(tempfile.mkdtemp(prefix="l174-portable-"))\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n for name in z.namelist():\n  assert not Path(name).is_absolute() and ".." not in Path(name).parts\n z.extractall(ROOT)\narrays,tasks,states,manifest=load_adaptation_packet(ROOT)\nsplit=adaptation_split(arrays["dates"][arrays["cell_ids"]])\nassert [int((split==i).sum()) for i in range(3)]==[6429,6074,106757]\nassert not (arrays["context"]==arrays["cell_ids"][:,None]).any()\nprint("Complete population and actual source checkpoints authenticated")');cell.metadata['tags']=['data-payload'];cells.append(cell)
 expected=next(x for x in r['runs'] if x['arm']=='adapter' and x['seed']==0);expected={k:v for k,v in expected.items() if k!='seconds'}
 cells +=[nb.v4.new_markdown_cell('## RUN · One fresh complete adaptation fit\nAdapter/seed0, ten full epochs, no subsampling. This is a fresh run from the inherited source checkpoint, not replay of the twelve author fits. The live TODO functions are passed into the trainer.'),nb.v4.new_code_cell('RUN=Path(tempfile.mkdtemp(prefix="l174-fresh-"))/"adapter-0"\nfresh=train_adaptation(arrays,tasks,states[0],"adapter",0,RUN,policy=configure_trainable,split_fn=adaptation_split,select_fn=select_epoch)\nPath("l174-fresh-report.json").write_text(json.dumps(fresh,indent=2)+"\\n")\nprint("Fresh selected epoch:",fresh["selected_epoch"],"test macro:",fresh["test"]["macro_loss"])'),nb.v4.new_markdown_cell('## CHECK · Compare fresh execution with saved author evidence\nTiming is deliberately excluded from equality. On other library versions, investigate score differences rather than replacing the author reference.'),nb.v4.new_code_cell('expected='+repr(expected)+'\nif np.__version__=='+repr(r['numpy'])+' and torch.__version__=='+repr(r['torch'])+':\n assert {k:v for k,v in fresh.items() if k!="seconds"}==expected\n print("Fresh full fit matches saved author evidence exactly (excluding runtime)")\nelse:\n print("Different environment; exact parity NOT_ESTABLISHED")\nplt.figure(figsize=(7,3))\nplt.plot(range(1,11),[e["validation"]["macro_loss"] for e in fresh["epochs"]],marker="o")\nplt.xlabel("Epoch");plt.ylabel("2007 validation macro loss");plt.title("Your fresh adapter/seed 0 fit");plt.grid(alpha=.2);plt.show()')]
 cells +=[nb.v4.new_markdown_cell('## RUN · All twelve fresh fits (gated)\nEnable only when ready to run the complete comparison. These are separately initiated notebook runs. The repository operator enforces the author’s shared 3,600-second budget. Never label this cell executed while its gate is false.'),nb.v4.new_code_cell('RUN_ALL_TWELVE=False\nif RUN_ALL_TWELVE:\n full_root=Path(tempfile.mkdtemp(prefix="l174-twelve-"))\n full=[train_adaptation(arrays,tasks,states[seed],arm,seed,full_root/f"{arm}-{seed}",policy=configure_trainable,split_fn=adaptation_split,select_fn=select_epoch) for seed in range(3) for arm in ["freeze","full","adapter","scratch"]]\n (full_root/"report.json").write_text(json.dumps(full,indent=2)+"\\n")\nelse:\n print("Twelve fresh notebook fits NOT_RUN. Complete author runs are separate saved evidence.")'),nb.v4.new_markdown_cell('## EXIT · Defend the result\nWrite 200–400 words: what changed, which weights stayed fixed, why identity initialization can learn, how you selected the checkpoint, whether pretraining helped, and what prior test exposure prevents you from claiming. Ask the teaching agent for feedback on your explanation.'),nb.v4.new_code_cell('submission=dict(defense="",fresh_test_macro_loss=fresh["test"]["macro_loss"],learner="PENDING_WRITTEN_DEFENSE",whole_paper="NOT_RUN",transfer="NOT_ESTABLISHED",test_exposure="RETROSPECTIVE_L173")\nPath("l174-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE")')]
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l174-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
     old=nb.read(path,4);book.metadata=old.metadata
     previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
     if [c.source for c in previous]==[c.source for c in current]:
         for prior,c in zip(previous,current):
             c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
 if solution and all(c.get('execution_count') is not None for c in cells if c.cell_type=='code'):
  from nbconvert import HTMLExporter
  from nbconvert.preprocessors import TagRemovePreprocessor
  exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
  html,_=exporter.from_notebook_node(book);(P/'html'/(S+'.html')).write_text(html)
print('Built L174 lesson, reference, full report and portable notebooks')
