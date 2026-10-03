"""Build model-specific figures, readable HTML and offline portable teaching labs."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;S='b12-adaptation-mechanisms';E=P/'evidence/b12';F=P/'figures/b12';F.mkdir(parents=True,exist_ok=True)
INK='#183c50';TEAL='#16877c';PURPLE='#6557a5'
def canvas(title,sub,size=(11,6)):
 f,a=plt.subplots(figsize=size);f.patch.set_facecolor('#fbfcfd');a.set(xlim=(0,1),ylim=(0,1));a.axis('off');a.text(.03,.94,title,fontsize=21,weight='bold',color=INK);a.text(.03,.88,sub,fontsize=11,color='#52616b');return f,a
def box(a,x,y,w,h,title,body,color='#eaf6f2'):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.013,rounding_size=.018',facecolor=color,edgecolor='#bacfd5'))
 a.text(x+.02,y+h-.035,title,fontsize=11,weight='bold',color=INK,va='top');a.text(x+.02,y+h-.10,body,fontsize=10.5,color=INK,va='top',linespacing=1.4)
def arrow(a,x1,y1,x2,y2,color=INK):a.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=color,lw=1.7))
def save(f,name):f.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(f)
f,a=canvas('Griffin · task loss changes the model','Shared interfaces → task-conditioned cell reads → downstream fine-tuning',size=(11,7))
box(a,.03,.61,.27,.19,'Database cells','Typed values + metadata\n→ cell vectors, width D')
box(a,.37,.61,.27,.19,'Repeated cell reads','Evolving task query\n→ cell attention', '#eef0fb')
box(a,.71,.61,.25,.19,'Relational layers','Within-relation means\nAcross-relation pooling')
arrow(a,.31,.7,.35,.7);arrow(a,.65,.7,.69,.7)
box(a,.71,.29,.25,.18,'Decoder','Task vector → prediction\nCompare with task label','#fff1dd');arrow(a,.835,.59,.835,.49)
box(a,.37,.29,.27,.18,'Optimizer','Task loss → gradients\nUpdate downstream weights','#fff1dd');arrow(a,.69,.38,.66,.38);arrow(a,.5,.49,.5,.59,TEAL)
box(a,.03,.29,.27,.18,'Initialization','Others-2 checkpoint\nor random initialization');arrow(a,.31,.38,.35,.38)
a.text(.03,.12,'Selected release: D=512 · four message layers · validation selects checkpoint',fontsize=11,color=INK)
a.text(.03,.05,'Inference reuses trained weights. B12 preserves this source path; it performs no Griffin training.',fontsize=10,color='#52616b');save(f,'griffin')
f,a=canvas('OpenRFM · detached two-stage variant','Relational representations → tabular ICL; interleaved alternatives are explained in the text',size=(11,7))
box(a,.03,.61,.27,.19,'Per-example database','Sample eligible cells\nL cells × D channels')
box(a,.37,.61,.27,.19,'Relational blocks','Cell / row / FK context\n→ example representation','#eef0fb')
box(a,.71,.61,.25,.19,'Tabular ICL stage','Support representations\n+ labels + query', '#fff1dd')
arrow(a,.31,.70,.35,.70);arrow(a,.65,.70,.69,.70)
box(a,.03,.29,.27,.19,'Distant support','May be outside query walk\nStill part of support pool','#fff1dd');arrow(a,.31,.38,.81,.58,TEAL)
box(a,.71,.29,.25,.19,'Query readout','Across-example evidence\n→ target prediction','#fff1dd');arrow(a,.835,.59,.835,.50)
a.text(.03,.19,'Pretraining learns weights; frozen ICL changes context and activations.',fontsize=11,color=INK)
a.text(.03,.10,'Missing author checkpoint identity blocks paper inference in B12.',fontsize=11,color=INK)
a.text(.03,.04,'The course kernel uses a simpler two-output average; it is not the paper model.',fontsize=10,color='#52616b');save(f,'openrfm')
f,a=canvas('KumoRFM-2 · labels enter early','Task-conditioned tables → foreign-key interaction → cross-sample prediction',size=(11,7))
box(a,.03,.61,.27,.19,'Historical task context','Known support targets\nQuery target hidden')
box(a,.37,.61,.27,.19,'Table network','Column ↔ row attention\nTask-conditioned vectors','#eef0fb')
box(a,.71,.61,.25,.19,'Relational network','Foreign-key attention\nConnect table rows','#eef0fb')
arrow(a,.31,.70,.35,.70);arrow(a,.65,.70,.69,.70)
box(a,.71,.29,.25,.19,'Cross-sample attention','Other labeled examples\n→ query representation','#fff1dd');arrow(a,.835,.59,.835,.50)
box(a,.37,.29,.27,.19,'Prediction head','Query representation\n→ target prediction','#fff1dd');arrow(a,.69,.38,.66,.38)
box(a,.03,.29,.27,.19,'Optional fine-tuning','Task labels → loss\n→ parameter updates');arrow(a,.35,.38,.31,.38);arrow(a,.30,.49,.40,.59,TEAL)
a.text(.03,.16,'Main report evaluation: frozen ICL. Optional fine-tuning is a separate mode.',fontsize=11,color=INK)
a.text(.03,.07,'Report-level architecture: exact proprietary dimensions and historical weights are not recovered.',fontsize=10,color='#52616b');save(f,'kumo')
f,a=canvas('A path changes what can matter','Two reads of the same support · weights stay fixed',size=(11,5))
box(a,.03,.52,.28,.24,'Support labels','[0, 1] → shuffle → [1, 0]\nAttention weights [¼, ¾]')
box(a,.37,.52,.27,.24,'No relational support','Relational reader = 0.5\nBefore and after shuffle','#eef0fb')
box(a,.71,.52,.25,.24,'Batch label reader','Clean 0.75\nShuffled 0.25','#fff1dd')
a.text(.04,.34,'Dual course readout: clean (0.5 + 0.75)/2 = 0.625',fontsize=15,color=TEAL)
a.text(.04,.23,'After shuffle:                    (0.5 + 0.25)/2 = 0.375',fontsize=15,color=PURPLE)
a.text(.04,.07,'This fixture demonstrates possible influence. No training or performance ranking is implied.',fontsize=11,color=INK);save(f,'trace')
f,a=canvas('36 paired conditions · 432 keyed predictions','Three fixture seeds; each condition predicts the same twelve queries',size=(11,5))
for x,title,body in [(.03,'2 reachability levels','High: 4 labels per walk\nLow: 0 labels per walk'),(.37,'3 label interventions','Intact / shuffled / hidden\nSame support identities'),(.71,'2 channels','Relational only / dual\n8 eligible supports total')]:box(a,x,.48,.25,.28,title,body)
a.text(.04,.28,'3 seeds × 2 × 3 × 2 = 36 conditions; no fitted model',fontsize=18,color=TEAL)
a.text(.04,.12,'Fixed: features, parent rows, projection weights, cutoffs and query keys.',fontsize=12,color=INK);save(f,'experiment')
report=json.loads((E/'diagnostic.json').read_text());conditions=report['conditions']
rows=[]
for reach in ['low','high']:
 for channel in ['relational','dual']:
  for mode in ['intact','shuffled','hidden']:
   rr=[x for x in conditions if (x['reachability'],x['channel'],x['mode'])==(reach,channel,mode)]
   rows.append(dict(reach=reach,channel=channel,mode=mode,brier=np.mean([x['brier'] for x in rr]),sd=np.std([x['brier'] for x in rr],ddof=1),change=np.mean([x['mean_abs_change'] for x in rr])))
table='| Reachability | Channel | Labels | Brier mean ± SD | Mean absolute change |\n|---|---|---|---:|---:|\n'
for r in rows:table+=f'| {r["reach"]} | {r["channel"]} | {r["mode"]} | {r["brier"]:.4f} ± {r["sd"]:.4f} | {r["change"]:.4f} |\n'
table+='\nSD is sample standard deviation over three fixture seeds, not a confidence interval. Change is relative to the same seed/reachability/channel with intact labels.'
f,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True);f.patch.set_facecolor('#fbfcfd')
for ax,reach in zip(axes,['low','high']):
 for offset,channel,color in [(-.13,'relational',TEAL),(.13,'dual',PURPLE)]:
  for i,mode in enumerate(['intact','shuffled','hidden']):
   values=[r['mean_abs_change'] for r in conditions if (r['reachability'],r['channel'],r['mode'])==(reach,channel,mode)]
   ax.scatter([i+offset+delta for delta in [-.025,0,.025]],values,color=color,s=35,alpha=.8,label=channel if i==0 else None)
 ax.set(title=reach.capitalize()+' relational reachability',xticks=range(3),xticklabels=['Intact','Shuffled','Hidden'],ylim=(-.015,.4));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend()
axes[0].set_ylabel('Mean absolute prediction change');f.suptitle('Each point is one fixture seed · no parameter updates',color=INK);f.tight_layout();save(f,'results')

def figure(name,caption):
    paths={'griffin':['Cells + metadata → shared cell encoders.','Task-conditioned cell reads → relational aggregation.','Decoder → task loss → optimizer updates weights.','Validation selects checkpoint; inference uses updated weights.'],
      'openrfm':['Eligible cells sampled for each example.','Relational blocks → example representations.','Tabular ICL stage reads support representations and labels.','Distant support can enter outside the query walk; query readout produces prediction.'],
      'kumo':['Known context targets enter table inputs early.','Column/row attention builds task-conditioned row vectors.','Foreign-key and cross-sample attention mix evidence.','Prediction head reads the query; optional fine-tuning separately updates weights.']}
    cls='adapt-figure adapt-wide' if name in paths else 'adapt-figure'
    html=f'<figure class="{cls}"><img src="../labs/figures/b12/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
    if name in paths:html+='<div class="adapt-mobile"><strong>Architecture path</strong><ol>'+''.join('<li>'+x+'</li>' for x in paths[name])+'</ol><p>'+caption+'</p></div>'
    return html
widget='''<div class="adapt-board" data-adaptation-paths><h3>Change the route, then change the labels</h3><label>Relational reachability <select name="reach"><option value="low">Low: no label path</option><option value="high">High: both labels</option></select></label><label>Support labels <select name="mode"><option value="intact">Intact: 0, 1</option><option value="shuffled">Shuffled: 1, 0</option><option value="hidden">Hidden: neither visible</option></select></label><button type="button">Reset example</button><output aria-live="polite">Relational reader = 0.500. Batch reader = 0.750. Dual average = 0.625. Weights stay fixed.</output><noscript><p>Controls require JavaScript; the worked arithmetic remains visible.</p></noscript></div>'''
source=(R/'lessons/content'/f'{S}.md').read_text()
replacements={'GRIFFIN':figure('griffin','Griffin selected release: labels train parameters; source model and trainer preserved.'),'OPENRFM':figure('openrfm','OpenRFM conceptual path: relational processing plus cross-example ICL; no B12 checkpoint execution.'),'KUMO':figure('kumo','KumoRFM-2 report architecture: context labels enter table encoding; optional fine-tuning separate.'),'TRACE':figure('trace','Two-label worked example; course kernel, not a published model.'),'EXPERIMENT_FIGURE':figure('experiment','Eight-support, twelve-query diagnostic; used label counts differ by channel.'),'RESULTS':table,'RESULT_FIGURE':figure('results','Measured prediction changes for all 36 conditions, three points per cell.'),'WIDGET':widget,'WARMUP':'<div id="b12-warmup"></div>','PREDICT':'<div id="b12-predict"></div>','TEACHBACK':'<div id="b12-teachback"></div>'}
body=source
for name,value in replacements.items():body=body.replace('{{'+name+'}}',value)
def document(title,body,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','adaptation-paths'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/adaptation-paths.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+render(body)+'</article>'+''.join('<script src="../assets/'+s+'.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B12 · Adaptation mechanisms',body,True))
reference='''# B12 · Adaptation and support access

**Fine-tuning:** downstream labels change parameters through a loss and optimizer. **Frozen ICL:** support inputs change activations and predictions without updating parameters. **Reachability:** whether eligible support labels enter the sampled query context.

| Mode | New-task parameter updates | Label entry |
|---|---|---|
| Griffin selected SFT | Yes | Training loss |
| OpenRFM frozen ICL | No | Relational and cross-example context |
| KumoRFM-2 main evaluation | No | Early task-conditioned table encoding and context |
| KumoRFM-2 fine-tuning | Yes | Both task training and possible context |

**Visibility:** event≤cutoff AND arrival≤cutoff AND event+horizon≤cutoff. Negative timestamps may be valid; unknown availability fails this fixture's policy. Real deployments need observed availability evidence.

**Worked read:** support logits[0,log3] → weights[¼,¾]. Labels[0,1] yield.75; shuffled[1,0] yield.25. With empty relational support, local=.5; equal-average dual=.625 before/.375 after. Equal averaging is a course rule.

**Interpretation:** no label path implies no direct label effect. A path allows an effect but need not produce one. Sensitivity is not accuracy, proof of feature learning, or historical reproduction. Match available support budget and report used labels separately.

**Author evidence:** all 36courseconditions/432 predictions verified; zero fits and zero paid calls. Griffin20fit target remains budget-gated. OpenRFM author artifacts and Kumo historical identity unresolved. Whole-paper NOT_RUN; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/b12-adaptation-mechanisms.html) · [Lab](../labs/b12-adaptation-mechanisms.ipynb) · [Contract](../labs/b12-reproduction.md) · [Griffin](https://proceedings.mlr.press/v267/wang25da.html) · [OpenRFM](https://arxiv.org/html/2606.04320v1) · [KumoRFM-2](https://arxiv.org/html/2604.12596v1)
'''
(R/'reference'/f'{S}.html').write_text(document('B12 · Adaptation reference',reference))
# Portable archive: primary sources, authenticated model/trainer and protocol; no weights.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((P/'sources/b12').rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(P/'sources/b12')))
 z.write(P/'b12-reproduction.md','b12-reproduction.md')
packet=buf.getvalue();(E/'portable-sources.zip').write_bytes(packet)
module=(P/'relkit/adaptation_b12.py').read_text();funcs={n.name:ast.get_source_segment(module,n) for n in ast.parse(module).body if isinstance(n,ast.FunctionDef)}
tests=(P/'_test_b12.py').read_text();checks={n.name:ast.get_source_segment(tests,n) for n in ast.parse(tests).body if isinstance(n,ast.FunctionDef)}
def functions(path):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef))
def png(name):return '!['+name+'](data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode()+')'
for solution in [False,True]:
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s):cells.append(nb.v4.new_code_cell(s))
 md('# B12 · Where does the new task enter?\n\n**PROVIDED:** visible fixed-kernel model, source archives and independent oracle. **TODO:** three live operations. **CHECK:** numerical and intervention feedback. **EXIT:** written defense.\n\nTier C synthetic mechanism isolation;36 conditions,432 predictions. No fitting or paid execution. This is not OpenRFM or Kumo inference. Paper targets and complete Griffin source are in the appendix.')
 code('''# @colab-bootstrap — standalone NumPy lab; no repository imports.
import importlib.util,subprocess,sys
if importlib.util.find_spec('numpy') is None:subprocess.check_call([sys.executable,'-m','pip','install','numpy'])
import numpy as np
import copy,hashlib,itertools,json,math,tempfile,base64,io,zipfile
from pathlib import Path
workspace=Path(tempfile.mkdtemp(prefix='b12-lab-'))
print('NumPy:',np.__version__,'; source and evidence workspace:',workspace)
''')
 # Carry the complete narrative, stripped of web-only controls and local links.
 narrative=source[source.index('B11 compared'):source.index('## 6 · Lab and defense')]
 for name,value in replacements.items():
  if name in ['GRIFFIN','OPENRFM','KUMO','TRACE','EXPERIMENT_FIGURE','RESULT_FIGURE']:
   filename={'GRIFFIN':'griffin','OPENRFM':'openrfm','KUMO':'kumo','TRACE':'trace','EXPERIMENT_FIGURE':'experiment','RESULT_FIGURE':'results'}[name];value=png(filename)
  elif name in ['WIDGET','WARMUP','PREDICT','TEACHBACK']:value='**Predict before continuing:** which labels can affect this query, and where do they enter?'
  narrative=narrative.replace('{{'+name+'}}',value)
 # Local file links are replaced with portable archive descriptions below.
 import re
 narrative=re.sub(r'\[([^\]]+)\]\((?:\.\./|b11-|0164-|0191-)[^)]+\)',r'\1 (in course or extracted source archive)',narrative)
 for section in narrative.split('\n## '):md(section if section.startswith('B11') else '## '+section)
 md('## TODO 1 · Temporal eligibility\n\nReturn one boolean per support: known finite event, arrival and horizon; nonnegative horizon; event and arrival no later than cutoff; completed outcome window. Reject a nonfinite cutoff or negative horizon. Equality is admitted in this fixture.')
 for i,(name,check) in enumerate([('eligible_support','check_eligibility'),('label_attention','check_attention'),('intervene','check_intervention')]):
  if i==1:md('## TODO 2 · Masked label reader\n\nInputs: Q×D queries, S×D keys, S labels, Q×S boolean permissions. Return Q probabilities from scaled dot-product softmax. Empty support returns.5. Reject visible nonfinite/out-of-range labels; forbidden unknown labels must have no effect. Keep the function safe for unequal query/support counts.')
  if i==2:md('## TODO 3 · Controlled label intervention\n\nReturn a new array for intact, shuffled or hidden labels. Require a full bijective integer permutation. Hidden labels become NaN and will be masked by the predictor. Do not mutate the original labels; reject unknown modes.')
  code(funcs[name] if solution else funcs[name].split('\n')[0]+f'\n    raise NotImplementedError("Complete {name}")')
  code(checks[check]+'\n'+check+'('+name+')\nprint("CHECK passed: '+name+'")')
 md('## PROVIDED · Database fixture\n\nRead the generator: parent features affect representations, membership edges set label reachability, and query truth is kept separate from predictor inputs. No task fitting occurs.')
 code(funcs['make_fixture'])
 md('## PROVIDED · Forward pass\n\nYour three functions below are the live global functions called by this predictor. Replacing one changes the actual experiment, not just an isolated checker. There is deliberately no optimizer: this experiment tests inference access with fixed weights.')
 code(funcs['predict'])
 md('## CHECK · Execute every approved condition\n\nPredict the low-reachability shuffle effect before running. Retain every complete query identity and probability; no selected subset is reported.')
 code(functions(P/'_run_b12.py')+'\nreport=run()\nprint(report["status"], report["condition_count"], report["prediction_count"])')
 md('## CHECK · Independent scalar oracle and forbidden information\n\nThis recomputes attention using scalar sums/exponentials and checks all 432 predictions, labels and Brier scores. It also rejects deliberately wrong learner functions.')
 code(functions(P/'_verify_b12.py')+'\nverification=verify(report)\nprint(verification)')
 code('assert report["weights_unchanged"] and report["fitted_models"]==0\n(workspace/"b12-results.json").write_text(json.dumps(report,indent=2))\nfor c in report["conditions"]:\n    print(c["seed"],c["reachability"],c["channel"],c["mode"],"Brier",round(c["brier"],4),"change",round(c["mean_abs_change"],4))')
 md('## EXIT · Defend the result\n\nExplain weights versus activations; trace labels through three architectures; distinguish eligible and used support counts. Give two explanations for corruption invariance when a path exists. Explain why a sensitive untrained kernel cannot prove feature learning. Specify an actual-checkpoint experiment with complete keys, temporal availability and validation-only selection. Ask the teaching agent for feedback. Revisit in1/7/30 days.\n\n**Learner: PENDING_WRITTEN_DEFENSE.** Author output is not a learner score.')
 md('## NEXT STEP · Inspect the complete source packet\n\nThe archive contains pinned papers, source ledger, full Griffin source/model/trainer/operators and full reproduction contract. Original upstream license is included. No paid run is launched. Archive wrappers retain their original repository paths; the contract identifies commands to use in a full checkout. The archive digest checks transport, not historical model identity.')
 code('packet=base64.b64decode('+repr(base64.b64encode(packet).decode())+')\nassert hashlib.sha256(packet).hexdigest()=='+repr(hashlib.sha256(packet).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(packet)) as z:\n    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())\n    z.extractall(workspace/"sources")\nledger=json.loads((workspace/"sources/source-ledger.json").read_text())\nfor name,digest in ledger["files"].items():\n    assert hashlib.sha256((workspace/"sources"/name).read_bytes()).hexdigest()==digest\nprint("Source packet authenticated; full benchmarks remain NOT_RUN")')
 md('## PROVIDED · Griffin full release model and trainer\n\nThese archival listings are separate from the course kernel above. Read the class/method boundaries: cell attention, relation aggregation, task decoding, training and validation selection. They are not imported into the default NumPy experiment. To reproduce the release, use the complete archived dependencies and pinned data/checkpoint contract, including the preserved budget STOP. Apache2.0 license: `sources/griffin/upstream/LICENSE` in the extracted packet.')
 for filename in ['hmodel.py','hmaintask_downsample_absolute_eval_sample.py']:
  text=(P/'sources/b12/griffin/upstream'/filename).read_text();nodes=ast.parse(text).body
  md('### Archival file · '+filename)
  # Split at source definitions; preserve imports and remaining top-level statements.
  for node in nodes:
   segment=ast.get_source_segment(text,node)
   md('**PROVIDED · '+getattr(node,'name','module setup')+'**\n\n```python\n'+segment+'\n```')
 md('## Full-run gate\n\nGriffin Table12:20 fits, unchanged 512/4096 subsets and seeds 42–46, INCOMPLETE_BUDGET_GATE. OpenRFM Table5: INCOMPLETE_SOURCE_PROTOCOL. KumoRFM-2: INCOMPLETE_MODEL_IDENTITY. No guessed substitutes or unpinned service calls. Whole-paper NOT_RUN; live Colab NOT_CHECKED. See the extracted `b12-reproduction.md` for the full named protocols and source gaps.')
 code('RUN_FULL_REPRO=False\nif RUN_FULL_REPRO:\n    raise RuntimeError("BLOCKED: USD0 scope and unresolved budget/source/model gates; read b12-reproduction.md")')
 n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python')))
 dest=P/('solutions' if solution else '')/(S+'.ipynb');dest.parent.mkdir(exist_ok=True);nb.write(n,dest)
(E/'environment.json').write_text(json.dumps(dict(python=__import__('sys').version,numpy=np.__version__,matplotlib=matplotlib.__version__,nbformat=nb.__version__),indent=2)+'\n')
print('Built B12 lesson, reference, six figures and both portable notebooks')
