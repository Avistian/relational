"""Build the lesson, reference, evidence tables and portable notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l173';S='0173-multi-task-pretraining'
r=json.loads((E/'report.json').read_text());coverage=json.loads((E/'coverage.json').read_text());tasks=json.loads((E/'tasks.json').read_text());baseline=json.loads((E/'baseline.json').read_text());manifest=json.loads((E/'manifest.json').read_text())
def results():
    lines=[f"**Complete:** {coverage['targets']:,} targets across {coverage['tasks']} tasks. Train / validation / test: "+' / '.join(f'{n:,}' for n in coverage['splits'])+'.',
    '', '| Loss weighting | Seed 0 test | Seed 1 test | Seed 2 test | Mean ± sample SD |','|---|---:|---:|---:|---:|']
    for arm in ['cell','task']:
        vals=[x['test']['macro_loss'] for x in r['runs'] if x['arm']==arm]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.6f}' for v in vals)+f' | {np.mean(vals):.6f} ± {np.std(vals,ddof=1):.6f} |')
    lines+=['',f"All entries are **test macro task loss** (lower is better), even for the cell-weighted training arm. Training-only constant baseline: **{baseline['macro_loss']:.6f}**. Each run scores all {coverage['splits'][2]:,} test targets."]
    return '\n'.join(lines)
cell=np.array([x['test']['macro_loss'] for x in r['runs'] if x['arm']=='cell']);task=np.array([x['test']['macro_loss'] for x in r['runs'] if x['arm']=='task']);delta=task-cell
interpretation=f"**Measured interpretation:** equal-task weighting changes test macro loss by {delta.mean():+.6f} on average (task minus cell). The three paired changes are "+', '.join(f'{x:+.6f}' for x in delta)+'. One seed reverses the direction. **Every trained run has higher test macro loss than the constant baseline.** This short pretraining schedule has not established an aggregate predictive benefit, even against that simple comparator. This supports a description of these six runs, not a general claim that equal-task weighting is better. Test losses also exceed validation losses substantially; do not present the validation curve as future-period performance.'
captions={'architecture':'Course model computation: erase the target; embed visible row and dated FK-parent cells; pool separately; share the encoder; route a local head; aggregate scalar losses. Shapes use batch B.',
'weights':'Illustrative fixed errors: nine A cells at loss 1, one B cell at loss 5. Changing weights changes 1.4 to 3 without changing predictions.',
'curves':'Measured six-run evidence. Circles outline the validation-selected epoch; paired seed points show complete test-population macro loss. Test targets are scored after selection.'}
def prose(portable=False):
    s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',results()).replace('[[INTERPRETATION]]',interpretation)
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l173'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l173/'+name+'.svg'
        s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0" style="overflow-x:auto"><img style="min-width:740px;width:100%" src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
    tags={'WARMUP':('Recall: why must MASKED and MISSING remain separate? Why does a valid FK not prove temporal availability?','<div id="warmup"></div>'),
    'PREDICT':('Predict before computing: do nine errors of 1 and one error of 5 give the same cell and task mean?','<div id="predict"></div>'),
    'LOSS_EXPLORER':('Intervention: increase A from 9 to 99 cells while keeping both task losses fixed. Cell mean becomes 1.04; equal-task mean stays 3. The CHECK cell below calculates both rules.','<div id="loss-explorer"></div><noscript>With A=9, B=1 and losses 1 and 5, cell mean is1.4 and task mean3. With A=99, cell mean becomes1.04 and task mean stays3.</noscript>'),
    'TEACHBACK':('Write your defense before reading any teacher feedback.','<div id="teachback"></div>')}
    for k,(text,html) in tags.items():s=s.replace('[['+k+']]',text if portable else html)
    if portable:
        s=s.replace('](../','](https://avistian.github.io/relational/')
        s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
    return s

def doc(title,body,interactive=False):
    html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
    css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','multitask-loss']
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','multitask-loss','l173-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0172-schema-tokenization.html">Lesson 172</a></nav><header><p class="route-kicker">Year 5 · Quarter 2 · Lesson 173</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Multi-task pre-training',prose(),True))
reference='''**Target contract:** choose an observed feature cell, hold its value separately, erase its identity from every context path. Missing cells are not targets. Keys preserve links, not magnitude. Autocomplete can use same-row fields; that is not forecasting.

| Choice | Formula | Consequence |
|---|---|---|
| Cell mean | Σ Nₜ Lₜ / N | Large tasks dominate |
| Equal-task mean | Σ Lₜ / T | Each task gets one scalar vote |
| Uniform minibatch weight | N / (T Nₜ) | Unbiased equal-task estimate using training counts |
| Numeric loss | Huber(normalized prediction − target), delta1 | Depends on training normalization |
| Category loss | logsumexp(logits) − correct logit | Local classes; UNKNOWN stays explicit |

**Do not** renormalize weights over tasks present in one minibatch. Equal scalar contribution does not imply equal gradient norms. Select the earliest best validation macro loss; score test after selection.

**Course architecture:** B×8 context IDs → 32-wide typed tokens → separate row/FK means + query-column embedding → 96→64→32 shared MLP → local head. No column attention or cross-sample ICL. Untimed parents excluded; dated parents cannot be later than query. Event time is not an ingestion history.

**Four conceptual axes:** row, column, FK and cross-sample describe information flow, not four mandatory losses. [KumoRFM-2 §3](https://arxiv.org/html/2604.12596v1#S3). For a different architecture with numeric Huber and boolean BCE, read [RT v1 §3.3](https://arxiv.org/html/2510.06377v1#S3.SS3).

'''+results()+'\n\n'+interpretation+'''

**Boundary:** complete selected course training; whole-paper reproduction NOT_RUN; transfer and historical availability NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0173-multi-task-pretraining.html) · [Notebook](../labs/0173-multi-task-pretraining.ipynb) · [Protocol](../labs/l173-reproduction.md).
'''
(R/'reference/multi-task-pretraining.html').write_text(doc('Multi-task pre-training — quick reference',reference))
lines=['# L173 measured course pretraining',results(),interpretation,'','## Per-task results','Baseline is training mean or Laplace-smoothed training class frequencies. Loss is the task-specific criterion. MAE and accuracy are kept separate.','', '| Task | Train / valid / test | Baseline loss | Cell mean loss | Task mean loss | Metric | Cell / task |','|---|---:|---:|---:|---:|---|---:|']
for i,spec in enumerate(tasks):
    byarm={arm:[run['test']['tasks'][i] for run in r['runs'] if run['arm']==arm] for arm in ['cell','task']}
    means={arm:np.mean([v['loss'] for v in values]) for arm,values in byarm.items()};metric='mae_raw' if spec['kind']=='number' else 'accuracy'
    scores={arm:np.mean([v[metric] for v in values]) for arm,values in byarm.items()}
    lines.append(f"| {spec['name']} | "+' / '.join(str(n) for n in spec['counts'])+f" | {baseline['tasks'][i]['loss']:.6f} | {means['cell']:.6f} | {means['task']:.6f} | {metric} | {scores['cell']:.6f} / {scores['task']:.6f} |")
lines+=['','[All 21 per-task validation curves](../../figures/l173/per-task-curves.svg) — mean and sample SD across seeds; separate loss scales.','','## Coverage and source gaps',f"{coverage['targets']:,} targets; {coverage['row_context_cells']:,} row context references; {coverage['parent_context_cells']:,} dated parent references. Full exclusion ledger: coverage.json.",'','Unknown category targets: '+', '.join(f"{x['task']}={x['unknown_targets']}" for x in baseline['tasks'] if 'unknown_targets' in x)+'.','', 'Whole-paper reproduction NOT_RUN. Cross-database transfer and historical availability NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.']
(E/'report.md').write_text('\n'.join(lines)+'\n')
# Portable packet: raw inputs, prepared population and author reference summary. No network.
names=set(manifest['inputs'])|{'evidence/l173/'+name for name in manifest['files']}|{'evidence/l173/manifest.json','evidence/l173/report.json','evidence/l173/baseline.json','sources/l173/source-ledger.json'}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(names):
        info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
source=(P/'relkit/multitask_l173.py').read_text();nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
contracts={
'erase_target':('Erase identities, not merely values','Return a new integer context matrix with every entry matching that row’s target ID replaced by padding0. Validate a 2D context, one positive target per row, and do not mutate inputs.', 'assert np.array_equal(erase_target(np.array([[1,2,2,0]]),np.array([2])),[[1,0,0,0]])\nprint("CHECK: duplicate references erased")'),
'task_weights':('Balance by the frozen training population','Accept task IDs, positive finite counts for every training task, and arm cell/task. Cell weights are ones. Task weights must make the uniform-example minibatch mean an unbiased estimator of the equal-task objective. A task absent from a minibatch must not alter another example’s weight.', 'ids=torch.tensor([0]*9+[1]);counts=torch.tensor([9.,1.]);losses=torch.tensor([1.]*9+[5.])\nassert torch.allclose((losses*task_weights(ids,counts,"cell")).mean(),torch.tensor(1.4))\nassert torch.allclose((losses*task_weights(ids,counts,"task")).mean(),torch.tensor(3.))\nprint("CHECK: same predictions, two objectives")'),
'select_epoch':('Select without looking at test','Return the zero-based earliest minimum of a nonempty finite validation score list. Raise ValueError for empty/nonfinite inputs. No test-score argument is allowed.', 'assert select_epoch([.7,.6,.6])==1\nprint("CHECK: earliest validation minimum")')}
explanations={'load_packet':'Hash-check the prepared tensors and metadata. The packet also includes raw tables and their hashes for independent reconstruction.', 'MaskedCellModel':'Model architecture. Read the token construction, padding denominator, concatenation and per-column head routing. Category codes index embeddings; they are never numeric quantities.', 'cell_losses':'Route each example to Huber or multiclass cross-entropy. The outer averaging rule is applied later.', 'tensor_packet':'Convert immutable prepared arrays to tensors. Only integer identities index context.', 'predict':'Score a specified population without gradients; this helper never selects a checkpoint.', 'score_outputs':'Report each task separately, plus an explicitly defined macro loss. MAE keeps original numerical units.', 'train_run':'Three complete epochs, fixed optimizer, paired permutation seed, saved checkpoints, validation-only selection, then test. Your weighting and selection functions are called by this trainer.'}
for solution in [False,True]:
    cells=[nb.v4.new_markdown_cell('# Lesson 173 · Multi-task pre-training\n\n**One skill:** design and implement the loss for a masked-cell training loop. The next narrative contains **saved author-reference evidence**, not results from your kernel. The executable section below runs a fresh full seed-0 fit. Tier B: real F1 database, course-defined autocomplete targets. Whole-paper reproduction NOT_RUN.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport copy,hashlib,json,base64,io,zipfile\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nimport pandas as pd\nimport matplotlib.pyplot as plt\ntorch.set_num_threads(1)\nprint("Author runtime: torch 2.13.0+cpu; exact replay may differ on other versions or devices.")')]
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Portable authenticated population\nAll 370,024 targets, full raw input packet, training-only metadata and saved author results. No network download or account needed. A hash checks integrity, not historical availability. The provided preparation script and independent verifier in the repository reconstruct the arrays from the raw tables.'))
    c=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nROOT=Path("l173-portable");ROOT.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n for name in z.namelist():\n  assert not Path(name).is_absolute() and ".." not in Path(name).parts\n z.extractall(ROOT)\nE=ROOT/"evidence/l173"\nprint("Portable packet authenticated")');c.metadata['tags']=['data-payload'];cells.append(c)
    for node in nodes:
        name=node.name;code=ast.get_source_segment(source,node)
        if name in contracts:
            title,contract,check=contracts[name]
            cells += [nb.v4.new_markdown_cell('## TODO · '+title+'\n\n'+contract),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)]
        else:cells += [nb.v4.new_markdown_cell('## PROVIDED · '+name+'\n\n'+explanations[name]),nb.v4.new_code_cell(code)]
    check_source=(P/'_check_l173.py').read_text();check_node=next(n for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef))
    cells += [nb.v4.new_markdown_cell('## CHECK · Attack the contracts\nPredict the failure of using batch counts instead of training counts; then run these checks on your live implementations.'),nb.v4.new_code_cell(ast.get_source_segment(check_source,check_node)),nb.v4.new_code_cell('assert check173(erase_target,task_weights,select_epoch)=="PASS"\narrays,tasks,manifest=load_packet(E)\nfor name,h in manifest["inputs"].items():\n assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h\narrays["context"]=erase_target(arrays["context"],arrays["cell_ids"])\nassert not (arrays["context"]==arrays["cell_ids"][:,None]).any()\nassert len(arrays["target"])==370024\nprint("Full population authenticated; target identities absent")')]
    cells += [nb.v4.new_markdown_cell('## RUN · Fresh full-population seed-0 training\nThis is one fresh three-epoch course fit, not a six-fit replay or paper reproduction. It calls your weighting and selection functions. Use a new output directory on each rerun; the trainer refuses to overwrite evidence.'),nb.v4.new_code_cell('import tempfile\nRUN=Path(tempfile.mkdtemp(prefix="l173-fresh-"))/"cell-0"\nfresh=train_run(arrays,tasks,"cell",0,RUN,weight_fn=task_weights,select_fn=select_epoch)\nreference=json.loads((E/"report.json").read_text())\nexpected=next(r for r in reference["runs"] if r["arm"]=="cell" and r["seed"]==0)\nif torch.__version__==reference["torch"] and np.__version__==reference["numpy"]:\n assert fresh==expected,"Fresh result differs on matching versions; investigate, do not replace reference"\nPath("l173-fresh-report.json").write_text(json.dumps(fresh,indent=2)+"\\n")\ndisplay(pd.DataFrame(fresh["test"]["tasks"]))\nprint("Fresh selected epoch:",fresh["selected_epoch"],"test macro loss:",fresh["test"]["macro_loss"])'),nb.v4.new_code_cell('plt.figure(figsize=(7,3))\nplt.plot([1,2,3],[e["validation"]["macro_loss"] for e in fresh["epochs"]],marker="o")\nplt.xlabel("Epoch");plt.ylabel("Validation macro loss");plt.title("Your fresh seed-0 cell-weighted fit")\nplt.xticks([1,2,3]);plt.grid(alpha=.2);plt.show()')]
    cells += [nb.v4.new_markdown_cell('## RUN · Complete six-fit course reproduction (gated)\nThe author-reference experiment contains all six fits. Enable this cell to rerun all six in your own runtime. It uses your live functions and the full population, and does not call a paid service. The repository command uses a 3,600-second aggregate watchdog; interactive notebook runs are your separately initiated work. Whole-paper reproduction remains NOT_RUN.'),nb.v4.new_code_cell('RUN_ALL_SIX=False\nif RUN_ALL_SIX:\n full_root=Path(tempfile.mkdtemp(prefix="l173-six-fits-"))\n full_results=[train_run(arrays,tasks,arm,seed,full_root/f"{arm}-{seed}",weight_fn=task_weights,select_fn=select_epoch) for seed in [0,1,2] for arm in ["cell","task"]]\n (full_root/"report.json").write_text(json.dumps(full_results,indent=2)+"\\n")\nelse:\n print("Six fresh notebook fits NOT_RUN; six saved author fits are a separate evidence lane.")')]
    cells += [nb.v4.new_markdown_cell('## EXIT · Defend the objective\nWrite 200–400 words addressing: cell versus task weighting; numerical versus category losses; a target-copy leak; missing cross-sample architecture; and why autocomplete is not transfer. Submit code, fresh results and your defense. A completed training run alone does not pass this gate.'),nb.v4.new_code_cell('submission=dict(defense="",fresh_run_macro_loss=fresh["test"]["macro_loss"],selected_epoch=fresh["selected_epoch"],learner="PENDING_WRITTEN_DEFENSE",whole_paper_reproduction="NOT_RUN",transfer="NOT_ESTABLISHED")\nPath("l173-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE")')]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(cells):c.id=f'l173-{i:03d}'
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
print('Built L173 lesson, reference, evidence tables and portable notebooks')
