"""Build B09 from canonical prose, source-visible contracts and authenticated evidence."""
import ast,base64,copy,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b09';S='b09-cost-frontier'
audit=json.loads((E/'course-audit.json').read_text());gate=json.loads((E/'paper-gate.json').read_text())
def figure(name,caption):return f'<figure class="b09-figure"><img src="../labs/figures/b09/{name}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
results=f'**Course status: {audit["status"]}; {audit["completed"]}/9 model–seed runs completed.**\n\n'
results+='| Model | Seed | RMSE ↓ | Load s | Fit s | First call s | Warm median s | Peak RSS MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for model in ['tabfm','exaone','nori']:
 for seed in range(3):
  rows=[r for r in audit['rows'] if r['model']==model and r['seed']==seed]
  if not rows:results+=f'| {model} | {seed} | {audit.get("missing_status",{}).get(f"{model}/{seed}","NOT_RUN")} | — | — | — | — | — |\n';continue
  x=rows[0];results+='| '+model+' | '+str(seed)+' | '+' | '.join(f'{x[k]:.3f}' for k in ['rmse','load_s','fit_s','first_predict_s','warm_median_s','peak_rss_mib'])+' |\n'
results+='\nTimes are seconds per **89 query rows**, not per 1,000. Warm = median of ten timed calls after first call and two warmups. Process peak RSS includes all stages. Downloads are excluded from the load column and charged to preparation. The first saved prediction array is scored; timed repetitions measure repeatability separately.\n\n'
for x in audit['summary']:results+=f'{x["model"]}: RMSE **{x["rmse_mean"]:.3f} ± {x["rmse_sd"]:.3f}**, mean ± sample SD over three splits. '
if audit['missing']:results+='\n\n**Incomplete arms:** '+', '.join(audit['missing'])+'. Consult the attempt log and budget ledger; missing results are not zero-cost or low-error points. TabFM stopped at a conservative resource preflight: its unmodified-loader memory allowance exceeds available RAM. No measured out-of-memory failure is claimed.'
if len([r for r in audit['rows'] if r['model'] in ['exaone','nori']])==6:
 ex={r['seed']:r for r in audit['rows'] if r['model']=='exaone'};no={r['seed']:r for r in audit['rows'] if r['model']=='nori'}
 diffs=[no[i]['rmse']-ex[i]['rmse'] for i in range(3)]
 results+='\n\nPaired **Nori minus EXAONE RMSE** by split: '+', '.join(f'{d:+.3f}' for d in diffs)+'. Negative favors Nori. These are paired dataset splits, not independent datasets. The missing TabFM arm prevents a complete three-model frontier.'
 results+=' Support-mean RMSE by split: '+', '.join(f'{ex[i]["baseline_rmse"]:.3f}' for i in range(3))+'. This baseline uses only support targets.'
widgets={'COST_WIDGET':'''<div class="b09-board" data-b09="cost"><h3>When does preparation pay off?</h3><p>Illustrative fixed quality and hardware. Predict the crossover.</p><label>Requests R: <strong data-value>1</strong><input type="range" min="1" max="100" value="1"></label><label><input type="checkbox" checked> Reuse B’s prepared support</label><button type="button">Reset cost example</button><output aria-live="polite">Baseline, one request: A=6.02 seconds; B=1.08 seconds. At 100 cached requests: A=8, B=9.</output></div>''','ATTENTION_WIDGET':'''<div class="b09-board" data-b09="attention"><h3>Change a forbidden value</h3><p>Equal-score primitive. Support values [2,6]; Q1=10.</p><label>Q2 value: <strong data-value>30</strong><input type="range" min="10" max="30" step="10" value="30"></label><label><input type="checkbox"> Incorrectly permit query keys</label><button type="button">Reset attention example</button><output aria-live="polite">Legal query readout=(2+6)/2=4. With all four keys it becomes (2+6+10+30)/4=12.</output></div>'''}
body=(R/'lessons/content'/f'{S}.md').read_text();replacements={**widgets,'ARCHITECTURE':'<div class="b09-desktop-architecture">'+figure('exaone-architecture.png','Current EXAONE release path; shapes omit only the variable number of encoded feature slots. Pretrained weights remain frozen.')+'</div><div class="b09-mobile-architecture"><ol><li><strong>Encode evidence.</strong> 353 support rows with targets;89 queries with hidden targets. Each cell becomes a 192-coordinate vector.</li><li><strong>Across columns.</strong> Three item-summary tokens gather features within each row.</li><li><strong>Across support rows.</strong> Thirty-two feature-summary tokens gather examples for each feature.</li><li><strong>Exchange across axes.</strong> Cells read opposite-axis summaries, then mix within an axis. Repeat 12 layers; feature attention twice per layer.</li><li><strong>Read queries.</strong> The head emits 999 quantiles, then the wrapper reverses target transforms and aggregates views. No gradient updates.</li></ol><p>Equal-score access trace: Q1 reads support values2and6 → weights ½, ½ → readout 4. Forbidden Q2=30 cannot change this readout.</p></div>','OTHER_ARCHITECTURES':figure('other-architectures.png','Current released inference paths, not a claim of historical checkpoint identity. TabFM collapses the cell grid into row vectors; Nori retains alternating feature/sample processing.'),'RESULTS':results,'RESULT_FIGURE':figure('results.png','Fresh CPU predictions and timings. Numbers beside points identify seeds. Missing arms have no points.') if audit['rows'] else '*No complete fresh run is available to plot.*','PAPER_GATE':'**INCOMPLETE_SOURCE_PROTOCOL.** Unresolved: '+ '; '.join(gate['gaps'])+'.'}
for key,value in replacements.items():body=body.replace('[['+key+']]',value)
assert '[[' not in body

def doc(title,text,scripts=False):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/cost-frontier.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+render(text)+'</article>'+(''.join('<script src="../assets/'+s+'.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','cost-frontier']) if scripts else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc('B09 · Current cost frontier',body,True))
reference='''# B09 · Cost-frontier field guide

An operating point records **checkpoint + data/split + support + preprocessing + ensemble + cache + precision + hardware + query workload**.

| Quantity | Boundary | Common failure |
|---|---|---|
| Cold load | Imports/construction/deserialization reported explicitly | Download hidden or included inconsistently |
| Support preparation | Fit preprocessing and caches on declared evidence | Lazy work attributed to another model stage |
| First prediction | First actual query call | Mistaken for steady-state cost |
| Warm prediction | Same request size, after declared warmups | Amortization and cache state omitted |
| Peak RSS | Absolute process high-water memory | Confused with model tensor bytes or GPU VRAM |
| Quality | RMSE on authenticated complete query IDs | Predictions silently reordered |

T=load+preparation+R×prediction when preparation is reused. A=(6,.02), B=(1,.08): crossover R≈83.3. Fewer parameters only reduce the weight-storage term; they do not fix preprocessing or attention work.

**Dominance:** A is no worse on every cost axis and strictly better on at least one. Ties stay on the frontier. Include uncertainty and incomplete arms before choosing a system. Three splits of one dataset cannot yield a cross-dataset ranking.

**Access:** item-summary tokens summarize features for each row; feature-summary tokens summarize support examples per column. Cross-axis exchange passes those summaries into the other attention direction. Changing a query target must not change predictions. Unlabeled-query preprocessing is a separate transductive dependency.

**Evidence:** B09-MATCHED-COST is current-release CPU inference on diabetes. EXAONE paper Figure3(b) remains INCOMPLETE_SOURCE_PROTOCOL. Nori-6M is not Nori-30M. TabFM base, + and Auto are different systems. Provider reports are not independent validation.

[Lesson](../lessons/b09-cost-frontier.html) · [Lab](../labs/b09-cost-frontier.ipynb) · [Protocol](../labs/b09-reproduction.md) · [EXAONE primary reading](https://arxiv.org/html/2608.25774v1#S2)
'''
(R/'reference'/f'{S}.html').write_text(doc('B09 · Cost-frontier field guide',reference))
# Portable packet: source-visible replay of saved evidence, not fresh checkpoint inference.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for path in sorted((P/'data/b09').glob('*.npz')):z.write(path,'data/'+path.name)
 for path in sorted(E.glob('*-predictions.npz')):z.write(path,'predictions/'+path.name)
 z.writestr('audit.json',json.dumps(audit));z.writestr('gate.json',json.dumps(gate))
packet=buf.getvalue();(E/'replay.zip').write_bytes(packet)
module=(P/'relkit/cost_b09.py').read_text();tree=ast.parse(module);functions=[ast.get_source_segment(module,n) for n in tree.body if isinstance(n,ast.FunctionDef)]
checks=['''assert abs(aligned_rmse([4,7],[2,6],[7,4],[8,2])-2**.5)<1e-12
for keys,values in [([4,4],[2,6]),([4,9],[2,6]),([4,7],[2,float('nan')])]:
    try: aligned_rmse([4,7],[2,6],keys,values)
    except ValueError: pass
    else: raise AssertionError('CHECK: duplicate, foreign or nonfinite rows must fail')
print('CHECK: reordered rows score correctly; corrupt packets fail')''','''assert pareto_front([[1,2,4],[2,3,5],[2,1,4],[1,2,4]]).tolist()==[True,False,True,True]
print('CHECK: strict dominance and exact ties handled')''','''q=np.array([[1.,0.]]);k=np.array([[0.,1.],[0.,-1.],[100.,0.]])
v=np.array([[2.],[6.],[999.]])
np.testing.assert_allclose(support_attention(q,k,v,2),[[4.]])
v[-1]=-1000
np.testing.assert_allclose(support_attention(q,k,v,2),[[4.]])
print('CHECK: forbidden query values cannot change the readout')''']
for solution in [False,True]:
 cells=[]
 def md(x):cells.append(nb.v4.new_markdown_cell(x))
 def code(x):cells.append(nb.v4.new_code_cell(x))
 md('# B09 · Defend a prediction-cost comparison\n\nSkill: authenticate rows, measure the entire prediction system, and interpret a Pareto frontier. PROVIDED cells supply data and plumbing; TODO cells are live implementations; CHECK cells give immediate feedback; EXIT requires your written defense.\n\n[Lesson](https://avist.github.io/relational/lessons/b09-cost-frontier.html). This portable notebook replays saved author evidence. It does **not** download or rerun large checkpoints by default. Fresh inference is a separate operator below.')
 code('# @colab-bootstrap\nimport sys, subprocess, importlib.util\nfor package in ["numpy", "matplotlib"]:\n    if importlib.util.find_spec(package) is None:\n        subprocess.check_call([sys.executable,"-m","pip","install",package])\nimport numpy as np\nimport base64, hashlib, io, json, zipfile\nfrom IPython.display import display, Markdown')
 md('## Concept recap\n\nA support set contains labeled examples; a query set withholds targets. A frozen checkpoint performs inference without gradient updates, but its wrapper may prepare transforms or ensembles.\n\nAn operating point includes those transforms, context size, hardware, precision and cache. Warm prediction omits cold load and may omit support work. Absolute peak RSS includes Python/native runtime memory; it is not GPU memory.\n\n**Worked example:** preparation/prediction costs A=(6,.02), B=(1,.08) seconds give A=6.02 and B=1.08 for one request. At 100 requests with cached support: A=8, B=9.\n\nRMSE is the square root of mean squared errors on the same row identities. Errors [0,2] yield √2. A Pareto point has no competitor no worse on all cost axes and strictly better on one. Equal points do not dominate each other.\n\nPredict: is a smaller checkpoint always cheaper? Explain which operation could reverse your answer.')
 md('## Model architecture · current EXAONE release\n\n![EXAONE cross-axis path](data:image/png;base64,'+base64.b64encode((P/'figures/b09/architecture-refined.png').read_bytes()).decode()+')\n\nCells carry vectors; item summaries collect columns for each row; feature summaries collect support rows for each column. Cells read the opposite-axis summaries before axis-wise attention. Current release: 12 layers, width 192, two feature-attention repetitions, 3 item slots and 32 feature slots. The regression head produces 999 quantiles before point aggregation. The portable visibility task is scaled dot-product attention, not full SSMax. Full inference source is visible in the appendix.\n\nTabFM embeds numeric feature groups with Fourier features, alternates column/row attention, pools rows, then predicts through an ICL stack. Nori alternates feature/sample attention and decodes a quantile distribution. Their wrappers change work and evidence access; do not infer historical identity from names.')
 md('![TabFM and Nori current release paths](data:image/png;base64,'+base64.b64encode((P/'figures/b09/other-architectures.png').read_bytes()).decode()+')\n\nTabFM compresses rows before its ICL stack. Nori keeps alternating feature/sample axes. Source inspection is separate from executed checkpoint evidence.')
 goals=['Authenticate the complete query identity and calculate RMSE after alignment. Reject duplicate/missing/nonfinite entries.','Return a boolean for each nondominated point. All supplied axes are minimized; equal points must survive.','Compute scaled dot-product attention using only the first support_count keys and values. A forbidden query must have no influence.']
 for i,src in enumerate(functions):
  name=ast.parse(src).body[0].name
  md(f'## TODO {i+1} · {name}\n\n**Goal:** {goals[i]}\n\n**Why:** this contract prevents an apparently plausible but invalid benchmark conclusion. Implement the function; the CHECK uses an example different from the author dataset. Do not edit the CHECK.')
  if solution:code(src)
  else:code(src.split('\n')[0]+f'\n    raise NotImplementedError("Complete {name}")')
  code('# CHECK\n'+checks[i])
 md('## PROVIDED · authenticated author-reference packet\n\nThis is saved-evidence replay, not fresh inference. Raw diabetes features were split 80/20 using seeds0/1/2; support353, queries89. Each completed arm saved full query identities. No checkpoint is trained in this notebook. Source and weight hashes accompany the downloadable reproduction ledger.')
 code('payload=base64.b64decode('+repr(base64.b64encode(packet).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(packet).hexdigest())+'\narchive=zipfile.ZipFile(io.BytesIO(payload))\naudit=json.loads(archive.read("audit.json"))\nprint(audit["status"], audit["completed"], "/", audit["expected"])')
 code('''# CHECK: your alignment function scores every complete author run.
records=[]
for row in audit['rows']:
    seed=row['seed'];model=row['model']
    d=np.load(io.BytesIO(archive.read(f'data/split-{seed}.npz')))
    p=np.load(io.BytesIO(archive.read(f'predictions/{model}-{seed}-predictions.npz')))
    score=aligned_rmse(d['test'],d['y'][d['test']],p['ids'],p['predictions'])
    assert abs(score-row['rmse'])<1e-9
    records.append((model,seed,score,row['warm_median_s'],row['peak_rss_mib']))
print('Authenticated completed runs:',len(records),'Missing:',audit['missing'])
if len(records)==9:
    points=np.array([[r[3],r[4],r[2]] for r in records])
    print('Run-level nondominated flags:',pareto_front(points).tolist())
else:
    print('No complete three-model frontier claim: missing arms remain visible.')''')
 md('## Author-reference results\n\n'+results)
 if (P/'figures/b09/results.png').exists():md('![Measured CPU results](data:image/png;base64,'+base64.b64encode((P/'figures/b09/results.png').read_bytes()).decode()+')\n\nLabels are split seeds; each point is one completed model/split. Axes have different units. Missing points are unexecuted arms, never inferred values.')
 md('## EXIT · written defense\n\n1. Explain why shuffled query IDs must be aligned before RMSE.\n2. Choose a one-request or repeated-request workload and name every charged stage.\n3. Defend what the completed matrix establishes and which missing arm blocks a stronger claim.\n4. Distinguish current-release diabetes inference from historical Figure3(b).\n\nPaste your answers to the tutor. Passing code leaves **PENDING_WRITTEN_DEFENSE**. Ask follow-up questions. Revisit the cost equation in1/7/30days after learner completion.')
 md('## Published-result gate and fresh inference\n\nThe exact target is EXAONE paper Figure3(b), regression accuracy–latency. Missing historical checkpoint/split/timing/Elo-pool identities block execution. No full pretraining or whole-paper claim is made.\n\nFrom the repository root, follow `labs/b09-reproduction.md`; run `labs/_run_b09.py` only after pinned sources, weights and dependencies are ready and budget remains. The displayed worker below is the complete measurement operator. The saved notebook packet is intentionally a separate lane.')
 code('RUN_HISTORICAL_FIGURE=False\ngate=json.loads(archive.read("gate.json"))\nprint(gate["status"],gate["gaps"])\nif RUN_HISTORICAL_FIGURE:\n    raise RuntimeError("Historical source protocol unresolved; no substitute run is allowed")')
 md('### Visible measurement operator\n\n```python\n'+(P/'_worker_b09.py').read_text()+'\n```')
 md('## Source-visible inference appendix\n\nThese are unmodified pinned release modules, provided for inspection with their licenses. They are not redefined or executed by the small visibility task. No unpublished pretraining trainer is invented. Full dependency modules are in the source archives. Follow the current wrapper configuration in each measured record; do not assume paper/release identity.')
 files=[('EXAONE CAST layer','exaone/src/exaonetabular/model/layer.py'),('EXAONE stack and readout','exaone/src/exaonetabular/model/transformer.py'),('EXAONE attention normalization','exaone/src/exaonetabular/model/attention.py'),('TabFM full PyTorch forward path','tabfm/tabfm/src/pytorch/model.py')]
 nori_files=list(Path('/tmp/b09-src/nori/src/synthefy_nori/model').glob('*.py'))
 files += [('Nori '+p.stem,str(p.relative_to('/tmp/b09-src'))) for p in nori_files if p.name in ['model.py','transformer.py','layer.py','attention.py','encoders.py','quantile_dist.py']]
 for label,path in files:
  f=Path('/tmp/b09-src')/path
  if f.exists():md('### '+label+'\n\n```python\n'+f.read_text()+'\n```')
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 target=P/('solutions' if solution else '')/(S+'.ipynb');target.parent.mkdir(exist_ok=True);nb.write(notebook,target)
print('Built lesson, reference, student and solution; author packet',audit['completed'],'/9')
