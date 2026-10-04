"""Build the lesson and standalone notebooks from canonical code and sealed evidence."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b20';F=P/'figures/b20';S='b20-curriculum-order'
r=json.loads((E/'diagnostic.json').read_text());paper=json.loads((E/'reproduction.json').read_text())
plt.rcParams.update({'font.size':11,'figure.facecolor':'#fafcf9','axes.facecolor':'#fafcf9','axes.spines.top':False,'axes.spines.right':False})
# Figure1: explicitly expose the attention information boundary.
fig=plt.figure(figsize=(8,10));gs=fig.add_gridspec(3,1,height_ratios=[1.5,2.1,1.2]);ax=fig.add_subplot(gs[0]);ax.axis('off');ax.set_title('Course PFN · order changes the next training task',loc='left',fontsize=15,weight='bold')
rows=[['support0','2','4','9','…','1'],['support1','3','2','5','…','0'],['…','…','…','…','…','…'],['query16','1','3','6','…','mean(y support)'],['query31','4','1','2','…','mean(y support)']]
t=ax.table(cellText=rows,colLabels=['row','x₁','x₂','x₃','x₄…₆','target slot'],loc='center',cellLoc='center',colWidths=[.18,.1,.1,.1,.13,.29],bbox=[0,.05,1,.75]);t.auto_set_font_size(False);t.set_fontsize(10)
for (i,j),cell in t.get_celld().items():cell.set_facecolor('#e1eee5' if i<4 else '#fff0da');cell.set_edgecolor('#bed0c5')
ax.text(0,-.08,'Normalize features using support only → scalar projections → [32 rows, 7 slots, 16 channels]',fontsize=10)
ax=fig.add_subplot(gs[1]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
def box(y,title,detail,color='#e1eee5'):
 ax.add_patch(FancyBboxPatch((.03,y),.94,.19,boxstyle='round,pad=.015',facecolor=color,edgecolor='#60816c'));ax.text(.06,y+.125,title,weight='bold');ax.text(.06,y+.055,detail,fontsize=10)
box(.76,'1 · Feature attention: seven slots read each other','Within each row; two heads. The target slot reads that row’s features.')
box(.48,'2 · Row attention: receivers read support rows','Keys/values = rows0–15 only. Query rows cannot send information.')
box(.20,'3 · Feed-forward + residual/normalization paths','16 → 32 → 16 channels. Repeat this entire block twice.')
for top,bottom in [(.76,.67),(.48,.39)]:ax.add_patch(FancyArrowPatch((.5,top),(.5,bottom),arrowstyle='-|>',mutation_scale=15,color='#306f59'))
ax.text(.06,.05,'Binary head on query target slots: [16,16] → [16,2] logits → probabilities',weight='bold',fontsize=11)
ax=fig.add_subplot(gs[2]);ax.axis('off')
ax.text(.02,.85,'Pretraining path',weight='bold',color='#306f59');ax.text(.02,.60,'Query cross-entropy → one AdamW update → next scheduled task\n192 updates; one persistent optimizer; final checkpoint only.',linespacing=1.6)
ax.text(.02,.27,'Evaluation path',weight='bold',color='#a16a29');ax.text(.02,.04,'Freeze weights → unseen support/query task → score hidden answers.\nNo optimizer updates; same24 evaluation tasks in every arm.',linespacing=1.6)
fig.tight_layout(h_pad=2);fig.savefig(F/'architecture.png',dpi=150);plt.close(fig)
# Figure2: four-example scalar SGD mechanism, not a measured PFN learning curve.
fig,axes=plt.subplots(1,2,figsize=(8,3.7));ex={'A':(1,1),'B':(1,1),'C':(2,-1),'D':(2,-1)}
for label,order,c in [('Staged','ABCD','#306f59'),('Mixed','ACBD','#ae722d')]:
 w=0;trace=[w]
 for id in order:x,y=ex[id];w-=.1*x*(w*x-y);trace.append(w)
 axes[0].plot(range(5),trace,'o-',label=label,color=c);axes[1].plot(range(5),[np.mean([.5*(v*x-y)**2 for x,y in ex.values()]) for v in trace],'o-',label=label,color=c)
axes[0].set_ylabel('Weight w');axes[1].set_ylabel('Loss on all four examples')
for ax in axes:ax.set_xticks(range(5));ax.set_xlabel('Updates');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Illustration · same examples, different update path',x=.03,ha='left',weight='bold');fig.tight_layout();fig.savefig(F/'order.png',dpi=150);plt.close(fig)
# Figure3: all paired seed scores, honest chance baseline.
fig,axes=plt.subplots(1,2,figsize=(8,4.2),sharey=True)
for ax,family in zip(axes,['single','relational']):
 for seed,c in zip([0,1,2],['#306f59','#a16a29','#54759a']):
  pair=[next(x for x in r['rows'] if x['family']==family and x['seed']==seed and x['mode']==mode)['auc'] for mode in ['staged','shuffled']]
  ax.plot([0,1],pair,'o-',label='seed'+str(seed),color=c)
 ax.axhline(.5,color='#666',ls='--',label='chance');ax.set_xticks([0,1],['Staged','Shuffled']);ax.set_title(family.capitalize()+' course prior',loc='left');ax.set_xlim(-.2,1.2);ax.set_ylim(.35,.65);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Mean of24 task AUROCs');axes[1].legend(fontsize=9,loc='lower right');fig.suptitle('Course result · paired seeds; no consistent staged advantage',x=.03,ha='left',weight='bold',fontsize=12);fig.tight_layout();fig.savefig(F/'scores.png',dpi=150);plt.close(fig)
# Figure4: denominator intervention, identical AUROCs.
fig,ax=plt.subplots(figsize=(8,3.4));v=[.638/.725,(.638-.5)/(.725-.5)]
ax.barh([1,0],v,color=['#54759a','#306f59'],height=.5);ax.set_yticks([1,0],['Raw AUROC\n0.638 /0.725','Above chance\n0.138 /0.225']);ax.set_xlim(0,1);ax.set_xlabel('Retained fraction of reference');ax.set_title('Same scores · two different denominators',loc='left',weight='bold')
for y,x in zip([1,0],v):ax.text(x+.015,y,f'{100*x:.1f}%',va='center',fontsize=13,weight='bold')
fig.tight_layout();fig.savefig(F/'retention.png',dpi=150);plt.close(fig)

def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b20/'+name+'.png'
 return f'<figure class="curriculum-figure" tabindex="0" role="region" aria-label="{caption}" style="max-width:100%;overflow-x:auto"><img src="{src}" alt="{caption}" style="display:block;width:100%;min-width:670px;height:auto"><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>'
results='| Prior | Order | AUROC: seed0 /1 /2 | Mean AUROC | Mean cross-entropy ↓ |\n|---|---|---|---:|---:|\n'
for family in ['single','relational']:
 for mode in ['staged','shuffled']:
  rows=[x for x in r['rows'] if x['family']==family and x['mode']==mode]
  results+=f'| {family} | {mode} | '+ ' / '.join(f"{x['auc']:.4f}" for x in rows)+f" | {np.mean([x['auc'] for x in rows]):.4f} | {np.mean([x['logloss'] for x in rows]):.4f} |\n"
pt='| Printed row | Mean of23 listed scores | Displayed average | Matches at3 decimals? |\n|---|---:|---:|---|\n'
for x in paper['rows']:pt+=f"| {x['label']} | {x['mean']:.6f} | {x['printed_average']:.3f} | {'Yes' if x['matches_3_decimals'] else 'No'} |\n"
widget='''<section class="curriculum-board" data-curriculum-order><h3>Change the order; keep all four examples</h3><div class="curriculum-controls"><label>Completed updates<select name="step"><option>0</option><option>1</option><option>2</option><option>3</option><option selected>4</option></select></label><label>Learning rate<select name="rate"><option>.025</option><option selected>.1</option><option>.2</option></select></label><button type="button">Reset</button></div><p>Staged: A B C D</p><div class="curriculum-track" data-track="Staged"></div><p>Mixed: A C B D</p><div class="curriculum-track" data-track="Mixed"></div><output aria-live="polite">After4 updates at rate0.1: staged w=−0.2516; mixed w=−0.2156. Both saw A,B,C,D once.</output><noscript><p>A and B have x=1,y=1. C and D have x=2,y=−1. Start w=0 and subtract rate×x×(wx−y) at each step. Use the static curve below to trace both orders.</p></noscript></section>'''
body=(R/'lessons/content'/f'{S}.md').read_text()
def fill(text,portable=False):
 items={'ORDER_WIDGET':'**Manual intervention:** compare orders A B C D and A C B D using the update above. Each example appears once.' if portable else widget,'ORDER_FIG':figure('order','Four-example SGD trace; illustrative training loss, not PFN evaluation.',portable),'ARCHITECTURE':figure('architecture','Source-shaped compact course PFN; query labels never enter the predictor.',portable),'RESULTS_FIG':figure('scores','All12 fits and three paired seeds; fixed synthetic evaluation pool.',portable),'RETENTION_FIG':figure('retention','Raw AUROC retention88% versus above-chance retention61.3%.',portable),'RESULTS':results,'PAPER_TABLE':pt}
 for k,v in items.items():text=text.replace('{{'+k+'}}',v)
 return text

def document(title,text,interactive=False):
 html=render(text).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','curriculum-order'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','benchmark-evidence','curriculum-order'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B20 · Curriculum Matters',fill(body),True))
ref='''# B20 · Ordering versus scale reference

[Lesson](../lessons/b20-curriculum-order.html) · [Notebook](../labs/b20-curriculum-order.ipynb) · [Reproduction contract](../labs/b20-reproduction.md)

| To isolate | Hold fixed | Measure separately |
|---|---|---|
| Order | Task bytes, repetitions, init, optimizer/state, updates, selection | Wall time and paired task scores |
| Generator | Backbone, evaluation, declared resource budget | Changed schema/content distribution |
| Data efficiency | Definition of a dataset/task/cell | Reuse and synthetic generation cost |
| Compute efficiency | Declared work or time budget | Width, attention work, padding, preprocessing |

**Worked update:** w=0,x=1,y=1,rate.1 → w=.1. Repeat → .19. Then x=2,y=−1 → −.086 → −.2516. Mixing the same examples gives a different path.

**Metric:** AUROC is positive-negative pair ordering with half credit for ties. Average within task before averaging tasks. Raw retention=s/b; above-chance retention=(s−.5)/(b−.5), defined here for b>.5. With s=.638,b=.725:88% versus61.3%.

**Course evidence:**12 fits,4608 predictions,96 tasks twice per fit;2-block width16 PFN. Near-chance scores and inconsistent order gains. No real-database generalization established.

'''+results+'''
**Paper evidence:** all322 printed scores audited; selected final A/B means0.702652/0.540783. A wins21/23 tasks. Best-ours row mean0.729652 differs from printed0.715. Exact historical training remains source-gated; printed arithmetic is not fresh inference.

**Architecture:** task generator → optional DFS → schedule → support-only normalization → feature attention → support-key row attention → query binary head. Training answers enter loss only. Keep optimizer state through stages; freeze weights at evaluation.

**Exit:** name the one changed factor, frozen identities, unit of cost, stopping rule and two falsification tests. Learner defense pending. Revisit after1/7/30days.
'''
(R/'reference'/f'{S}.html').write_text(document('B20 reference',ref))
# Standalone packet contains frozen source and evidence, not another repository checkout.
paths=[p for p in E.iterdir() if p.is_file() and p.name not in ['local-budget.json','seal.json','artifact-manifest.json','portable-packet.zip']]+[p for p in (P/'sources/b20').iterdir() if p.is_file()]
manifest={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(paths):
  info=zipfile.ZipInfo(str(p.relative_to(P)),date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
raw=buf.getvalue();(E/'portable-packet.zip').write_bytes(raw)
source=(P/'relkit/curriculum_b20.py').read_text();tree=ast.parse(source);nodes={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
# Keep compact code in coherent chunks, with three live learner definitions.
header=fill(body,True)
header=re.sub(r'<div id="[^"]+"></div>','',header)
header=re.sub(r'\]\((\.\./[^)]+|b\d[^)]+\.html)\)',lambda m:'](https://avistian.github.io/relational/'+(m.group(1)[3:] if m.group(1).startswith('../') else 'lessons/'+m.group(1))+')',header)
base=[nb.v4.new_markdown_cell(header),nb.v4.new_markdown_cell('## PROVIDED · environment and portable packet\nRun PROVIDED cells, implement TODOs, then run each CHECK. This notebook uses NumPy and PyTorch for the visible model. The default lane replays saved evidence, with no network or model downloads. Optional fresh course training comes after EXIT. Live Colab has not been tested.')]
bootstrap='''# @colab-bootstrap: standalone notebook; no repository checkout required.
import importlib.util, subprocess, sys
missing=[n for n in ['numpy','torch'] if importlib.util.find_spec(n) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import hashlib, io, json, zipfile
from pathlib import Path
from decimal import Decimal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
'''
base.append(nb.v4.new_code_cell(bootstrap))
encoded=base64.b64encode(raw).decode();bootstrap2=f'''import base64
packet=base64.b64decode({encoded!r})
assert hashlib.sha256(packet).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
root=Path('b20-packet');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(packet)) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
E=root/'evidence/b20'
print('Authenticated',len(manifest),'source/evidence files')'''
payload_cell=nb.v4.new_code_cell(bootstrap2);payload_cell.metadata.update({'tags':['hide-input'],'jupyter':{'source_hidden':True},'collapsed':True});base.append(payload_cell)
for name,title,check in [
 ('exposure_order','TODO1 · preserve the exposure multiset','''levels=np.array([2,0,1,0,2,1])
a=exposure_order(levels,'staged',7,2);b=exposure_order(levels,'shuffled',7,2)
assert len(a)==len(b)==12
assert np.all(np.diff(levels[a])>=0)
for order in [a,b]:np.testing.assert_array_equal(np.bincount(order,minlength=6),np.full(6,2))
np.testing.assert_array_equal(b,exposure_order(levels,'shuffled',7,2))
assert not np.array_equal(a,b)
print('CHECK: complete multiset, deterministic schedule, distinct order')'''),
 ('paired_contract','TODO2 · reject a confounded pair','''left=dict(pool='p',init='i',updates=12,optimizer={'lr':.001},exposures=[2,2,2],selection='final',evaluation='e',batch_size=1)
assert paired_contract(left,dict(left))
for key in left:
    right=dict(left);right[key]='changed'
    try:paired_contract(left,right)
    except ValueError:pass
    else:raise AssertionError('Accepted changed '+key)
print('CHECK: every changed control rejected')'''),
 ('auc_retention','TODO3 · make the denominator explicit','''r=auc_retention(.638,.725)
assert abs(r['raw']-.88)<1e-12
assert abs(r['above_chance']-.138/.225)<1e-12
try:auc_retention(.7,.5)
except ValueError:pass
else:raise AssertionError('Chance baseline accepted')
print('CHECK:',r)''')]:
 base.append(nb.v4.new_markdown_cell('## '+title+'\nImplement the named contract from the lesson. Predict which error would invalidate the comparison before writing code. Preserve the supplied signature; invalid inputs should raise ValueError.'))
 cell=nb.v4.new_code_cell(nodes[name]);cell.metadata['learner_function']=name;base.append(cell);base.append(nb.v4.new_code_cell(check))
for title,names in [('PROVIDED · course priors and identities',['task_pool','array_hash']),('PROVIDED · normalize and embed support/query inputs',['normalize_support','FeatureEncoder','TargetEncoder']),('PROVIDED · attention blocks and prediction head',['BiAttention','Decoder','RDBPFN']),('PROVIDED · persistent optimizer and final-checkpoint evaluation',['fit_course','evaluate'])]:
 base.append(nb.v4.new_markdown_cell('## '+title+'\nTrace this code against the architecture above. The course generator and model sizes are declared deviations from the paper.'))
 base.append(nb.v4.new_code_cell('\n\n'.join(nodes[x] for x in names)))
verify=(P/'_verify_b20.py').read_text();audit=next(n for n in ast.parse(verify).body if isinstance(n,ast.FunctionDef) and n.name=='audit')
base.append(nb.v4.new_markdown_cell('## CHECK · rescore all4608 predictions independently\nThe original run used sklearn AUROC. This audit counts every positive-negative pair directly, authenticates inputs and reconstructs predictions with the saved compact model weights. No new training is performed.'))
base.append(nb.v4.new_code_cell('import itertools\n'+ast.get_source_segment(verify,audit)))
base.append(nb.v4.new_code_cell('''report=json.loads((E/'diagnostic.json').read_text())
independent=audit(E,report)
Path('b20-replay.json').write_text(json.dumps(independent,indent=2)+'\\n')
print('Full grid verified:',len(independent),'fits and4608 predictions')
for family in ['single','relational']:
    for seed in [0,1,2]:
        rows=[r for r in report['rows'] if r['family']==family and r['seed']==seed]
        paired_contract(*rows)
        print(family,seed,'staged-minus-shuffled AUROC',round(rows[0]['auc']-rows[1]['auc'],5))'''))
repro=(P/'_reproduce_b20.py').read_text();func=next(n for n in ast.parse(repro).body if isinstance(n,ast.FunctionDef) and n.name=='replay')
base.append(nb.v4.new_markdown_cell('## CHECK · reconstruct all322 printed scores\nRead the archived primary HTML for provenance. The packet contains every task score from Tables5–6; the following Decimal calculation avoids binary rounding in the arithmetic audit.'))
base.append(nb.v4.new_code_cell(ast.get_source_segment(repro,func)))
base.append(nb.v4.new_code_cell('''paper_packet=json.loads((E/'paper-packet.json').read_text())
paper_report=replay(paper_packet)
assert paper_report==json.loads((E/'reproduction.json').read_text())
Path('b20-paper-replay.json').write_text(json.dumps(paper_report,indent=2)+'\\n')
print(paper_report['selected'])
for row in paper_report['rows']:
    if not row['matches_3_decimals']:print('SUMMARY MISMATCH:',row)
print('Printed arithmetic COMPLETE; historical training NOT_RUN')'''))
base.append(nb.v4.new_markdown_cell('''## EXIT · make the causal claim defensible
Explain why the small failed training experiment cannot refute the paper. Distinguish per-task best scores from a validation-selected checkpoint. Submit a frozen order-versus-scale proposal with two falsification tests. Learner status remains PENDING_WRITTEN_DEFENSE until assessed.

## Optional fresh course run · complete12-fit grid
This executes the visible trainer on the frozen pools; it is not paper pretraining. The saved replay above is the default. Do not select a favorable subset of seeds. No paid resources are used. The author's PyTorch/NumPy versions are recorded in the packet; other versions may not give byte-identical floating-point results.'''))
base.append(nb.v4.new_code_cell('''RUN_FRESH_COURSE=False
if RUN_FRESH_COURSE:
    import time
    start=time.monotonic();fresh=[]
    for seed in [0,1,2]:
        for family in ['single','relational']:
            pool=dict(np.load(E/f'pool-{family}-{seed}.npz'))
            for mode in ['staged','shuffled']:
                if time.monotonic()-start>600:raise RuntimeError('INCOMPLETE_LOCAL_BUDGET_GATE')
                model,trace=fit_course(pool,mode,seed)
                probability=evaluate(model,dict(np.load(E/'evaluation.npz')))
                fresh.append(dict(seed=seed,family=family,mode=mode,p=probability.tolist(),init=trace['init']))
    Path('b20-fresh-course.json').write_text(json.dumps(fresh))
    print('Complete course rerun:',len(fresh),'fits; paper NOT_RUN')'''))
base.append(nb.v4.new_markdown_cell('''## Paper-results operator · source gate
Full named target: Table1 A/B, all23 tasks at context1024. Exact historical corpus, schedule, architecture, seeds/splits, checkpoint selection and evaluation support identities remain unresolved. This notebook authenticates the archived source and reconstructs the printed matrix. It cannot manufacture the missing historical trainer. In the repository run `python labs/_reproduce_b20.py`; `--fresh` intentionally refuses execution. No fresh paper pretraining, raw paper prediction scoring or whole-paper reproduction has run.'''))
base.append(nb.v4.new_code_cell("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    raise RuntimeError('NOT_RUN: INCOMPLETE_SOURCE_PROTOCOL_GATE. Resolve the historical identities in b20-reproduction.md before training.')"))
for solution in [True,False]:
 import copy
 cells=copy.deepcopy(base)
 if not solution:
  for cell in cells:
   if name:=cell.metadata.get('learner_function'):
    signature=nodes[name].split('\n',1)[0];cell.source=signature+'\n    raise NotImplementedError("Implement '+name+'")'
 for index,cell in enumerate(cells):cell.id=f'b20-{index:03d}'
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 path=P/'solutions'/f'{S}.ipynb' if solution else P/f'{S}.ipynb';nb.write(book,path)
print('Built lesson/reference,4 figures and standalone student/solution notebooks')
