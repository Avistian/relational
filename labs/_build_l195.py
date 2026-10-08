"""Deterministic lesson/reference/portable-notebook builder for the frozen replay."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l195';F=P/'figures/l195';S='0195-thesis-stress-test'
F.mkdir(exist_ok=True,parents=True);r=json.loads((E/'report.json').read_text())
source=(P/'relkit/stress_l195.py').read_text();functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
s=(P/'_replay_l195.py').read_text();operators={n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l195','font.size':10})
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(F/(name+'.'+ext),dpi=170,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {'Software':'L195'})
 plt.close(fig)
fig,ax=plt.subplots(figsize=(3.8,7.4));fig.patch.set_facecolor('#f5f8f3');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.02,.98,'Four claims, four evidence needs',fontsize=11,weight='bold',va='top')
boxes=[(.76,'C1 · Useful information','Target-only → add legal records','Needs: matched information contrast'),(.53,'C2 · Learned recovery','Relational summaries ↔ learned model','Needs: fair processing comparison'),(.30,'C3 · Robust advantage','One task → new tasks / times','Needs: independent held-out tests'),(.07,'C4 · Undervaluation','Predictive gain → cost and adoption','Needs: economic / attention evidence')]
for y,title,change,need in boxes:
 ax.add_patch(FancyBboxPatch((.02,y),.96,.17,boxstyle='round,pad=.012',fc='white',ec='#9bb9aa'))
 ax.text(.055,y+.13,title,weight='bold',fontsize=11);ax.text(.055,y+.083,change,fontsize=9.4);ax.text(.055,y+.035,need,fontsize=9.2,color='#436355')
 if y>.1:ax.annotate('',xy=(.5,y-.045),xytext=(.5,y-.005),arrowprops={'arrowstyle':'->','color':'#627b6d'});ax.text(.54,y-.035,'needs new evidence',fontsize=8.5)
save(fig,'claims')
g=r['regression']['test'];ci=g['conditional_interval'];fig,(a,b)=plt.subplots(2,1,figsize=(3.8,6.2),gridspec_kw={'height_ratios':[3,1.6]},layout='constrained');fig.patch.set_facecolor('#f5f8f3')
for x in g['runs']:
 s=x['seed'];a.plot([x['fe_mae'],x['gnn_mae']],[s,s],color='#c4c9c0',lw=2);a.scatter(x['fe_mae'],s,color='#236652',marker='s',s=42,zorder=3);a.scatter(x['gnn_mae'],s,color='#a14c35',s=42,zorder=3)
a.set_yticks(range(5),[f'Run {s}' for s in range(5)]);a.invert_yaxis();a.set_xlim(3.7,4.5);a.set_xlabel('Test MAE · smaller is better');a.set_title('L149 · all five saved runs',loc='left',weight='bold');a.scatter([],[],color='#236652',marker='s',label='Engineered features');a.scatter([],[],color='#a14c35',label='Basic GNN');a.legend(loc='lower left',bbox_to_anchor=(0,1.03),fontsize=9,frameon=False);a.spines[['top','right']].set_visible(False)
b.axvline(0,color='#637167',ls='--');b.plot([ci['low'],ci['high']],[0,0],color='#a14c35',lw=4);b.scatter(g['gnn_advantage'],0,color='#a14c35',s=55,zorder=4);b.set_xlim(-.55,.2);b.set_ylim(-.6,.7);b.set_yticks([]);b.set_xlabel('GNN advantage · position units');b.set_title('Conditional 95% driver interval',loc='left',fontsize=10);b.text(g['gnn_advantage'],.25,f"Mean {g['gnn_advantage']:+.4f}",ha='center',fontsize=10);b.text(-.53,-.4,'← favors features',fontsize=9);b.text(.19,-.4,'favors GNN →',fontsize=9,ha='right');b.spines[['top','left','right']].set_visible(False);save(fig,'regression')
i=r['icl']['advantage'];fig,ax=plt.subplots(figsize=(3.8,5.2),layout='constrained');fig.patch.set_facecolor('#f5f8f3');ax.axvline(0,color='#627b6d',ls='--');ax.axvline(i['mean'],color='#236652',lw=1.5,label=f"Mean {i['mean']:+.4f}")
for s,v in enumerate(i['per_seed']):ax.plot([0,v],[s,s],color='#b7cbbb');ax.scatter(v,s,color='#236652' if v>0 else '#a14c35',s=45,zorder=4)
ax.set_yticks(range(10),[f'Draw {s}' for s in range(10)]);ax.invert_yaxis();ax.set_xlabel('RDB-PFN − TabICL · AUROC');ax.set_title(f'L182 · ten support draws\nMean {i["mean"]:+.4f} · positive on 6/10',loc='left',fontsize=11,weight='bold');ax.spines[['top','right']].set_visible(False);save(fig,'icl')
captions={'claims':'Each arrow needs additional evidence. A predictive comparison cannot fill a missing information intervention, robustness test, or economic measurement.','regression':'Stored test predictions, rescored in L195. Each line joins the same run index, not shared training randomness. The interval resamples 56 drivers after averaging query losses across runs; it does not describe new databases.','icl':'Stored L182 predictions, rescored in L195. Each point contrasts the same support draw and test queries. Ten draws reuse one F1 task; the mean line is not a confidence interval.'}
status='<div class="repro-status"><strong>Complete selected replay:</strong> 33,650 saved predictions rescored. No new training or inference. RDBLearn full model reproduction remains INCOMPLETE_SOURCE_PREPROCESSING_GATE. Learner defense pending.</div>'
claim_table='| Subclaim | Current defensible position | Missing decisive evidence |\n|---|---|---|\n| C1: useful structure can be lost | Possible; not inevitable for every task/map | Matched information intervention |\n| C2: learned recovery helps | Task-dependent hypothesis; strong FE comparator | Matched pipeline and prior/architecture controls |\n| C3: fair robust gain | Scoped results; broad advantage unestablished | New databases, time shifts, full uncertainty |\n| C4: undervalued opportunity | Not measured by these scores | Cost, utility and adoption evidence |'
def fill(text,portable=False):
 text=text.replace('[[STATUS]]',status).replace('[[CLAIM_TABLE]]',claim_table)
 text=text.replace('<div id="margin-explorer"></div>',f'<div id="margin-explorer" data-low="{ci["low"]}" data-high="{ci["high"]}"></div>')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l195/'+name+'.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="repro-figure"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 if portable:
  text=re.sub(r'<div id="[^\"]+"[^>]*></div>','',text).replace('<noscript>','<p>').replace('</noscript>','</p>')
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,scripts=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="repro-table" tabindex="0" role="region" aria-label="Scrollable evidence table"><table\1>',body).replace('</table>','</table></div>')
 tail=''.join('<script src="../assets/'+n+'.js"></script>' for n in ['retrieval-pool','retrieval-bank','predict','teachback','thesis-stress']) if scripts else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/multitask-reproduction.css"><link rel="stylesheet" href="../assets/thesis-stress.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+tail+'</body></html>'
prose=(R/'lessons/content'/(S+'.md')).read_text();(R/'lessons'/(S+'.html')).write_text(doc('Lesson 195 — Thesis stress-test',render(fill(prose)),True))
ref='''# Thesis stress-test · field guide

**Ask first:** which proposition could this observation contradict?

'''+claim_table+'''

**Sign convention:** positive candidate advantage is better. MAE: baseline minus candidate; AUROC: candidate minus baseline. Match complete entity/cutoff keys before comparing. Do not pool different metric units.

**Interval geometry:** with margin δ≥0, lower endpoint >δ is ABOVE_MARGIN; upper endpoint <−δ is BELOW_NEGATIVE_MARGIN; an interval contained in [−δ,+δ] with δ>0 is WITHIN_MARGIN; otherwise UNRESOLVED. Incomplete required runs yield INCOMPLETE. These labels describe geometry. A retrospectively chosen band and a conditional bootstrap do not establish formal equivalence.

**Worked measured example:** basic GNN advantage −0.174155 position units; conditional driver interval [−0.449621,+0.085850]. It crosses zero. At δ=0, UNRESOLVED. At δ=0.50, WITHIN_MARGIN, but that post hoc choice is not a preregistered result.

**Evidence boundaries:** a source failure is not a measured score loss; a replay is not an independent replication; repeated support draws are not databases; a tabular predictor may consume relational features; benchmark success does not measure undervaluation.

**Brief template:** exact claim and population → strongest counter-evidence → strongest reply → new observation that would reverse the position. For a future test, specify varied factor, fixed alternatives, legal feature availability, comparator/search budget, metric and practical margin, uncertainty unit and stop rule before test access.

**Current packets:** L149 all five runs per pipeline; L182 all three arms × ten draws; L194 all21task rows with zero fresh scores. Full RDBLearn model reproduction remains INCOMPLETE_SOURCE_PREPROCESSING_GATE. New training and whole-paper reproduction NOT_RUN.

[Lesson](../lessons/0195-thesis-stress-test.html) · [Student notebook](../labs/0195-thesis-stress-test.ipynb) · [Full brief](../labs/evidence/l195/falsification-brief.md) · [Reproduction protocol](../labs/l195-reproduction.md) · [RDBLearn primary reading](https://arxiv.org/html/2602.18495v1).
'''
(R/'reference/thesis-stress-test.html').write_text(doc('Thesis stress-test field guide',render(ref)))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((E/'packet').rglob('*'))+[E/'input-manifest.json']:
  if not p.is_file():continue
  info=zipfile.ZipInfo(str(p.relative_to(E)),date_time=(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
payload=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
contracts=[('paired_mae','truth, candidate, baseline','Each input contains (entity, cutoff, value) triples. Reject duplicate, missing, extra or invalid numeric rows. Align predictions to truth order. Return keys, candidate_mae, baseline_mae and the per-query advantage list (baseline absolute error minus candidate absolute error). Use float64 Python arithmetic. Reordering predictions must not change the result.',"t=[(1,10,2.),(1,20,4.)]\na=[(1,20,5.),(1,10,2.)];b=[(1,10,4.),(1,20,4.)]\nx=paired_mae(t,a,b)\nassert x['advantage']==[2.,-1.] and x['candidate_mae']==.5\ntry: paired_mae(t,a[:-1],b)\nexcept ValueError: pass\nelse: raise AssertionError('Missing query accepted')"),('interval_verdict','low, high, margin, complete=True','Validate a finite nonnegative numeric margin (not a boolean) and boolean completeness. Incomplete evidence returns INCOMPLETE. Otherwise require finite ordered endpoints. Apply the field-guide geometry: ABOVE_MARGIN, BELOW_NEGATIVE_MARGIN, WITHIN_MARGIN or UNRESOLVED. Keep strict inequality for directional decisions; the zero-width band never establishes equivalence.',"assert interval_verdict(-.45,.09,0)=='UNRESOLVED'\nassert interval_verdict(-.45,.09,.5)=='WITHIN_MARGIN'\nassert interval_verdict(None,None,.1,False)=='INCOMPLETE'\nassert interval_verdict(.03,.08,.02)=='ABOVE_MARGIN'\ntry: interval_verdict(1,0,0)\nexcept ValueError: pass\nelse: raise AssertionError('Reversed interval accepted')"),('claim_scope','claim, evidence','Implement this packet-specific policy. A pipeline claim requires authenticated, complete, measured and comparable to each be exactly True, giving SCOPED_COMPARISON; otherwise INSUFFICIENT_EVIDENCE. relational_signal, architecture_cause, general_superiority and undervaluation remain NOT_ESTABLISHED; fresh_training is NOT_RUN. Reject unknown claims. Explain what missing measurement would be needed for each stronger claim.',"e=dict(authenticated=True,complete=True,measured=True,comparable=True)\nassert claim_scope('pipeline',e)=='SCOPED_COMPARISON'\nassert claim_scope('pipeline',dict(e,measured=False))=='INSUFFICIENT_EVIDENCE'\nassert claim_scope('undervaluation',e)=='NOT_ESTABLISHED'\ntry: claim_scope('models always win',e)\nexcept ValueError: pass\nelse: raise AssertionError('Unknown claim accepted')")]
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 195 · Thesis stress-test\n\nTier B: real saved RelBench prediction evidence. This offline NumPy notebook replays all approved runs. It trains no model. Author-reference figures below precede your kernel outputs; successful execution is not your written defense.')
 code('# @colab-bootstrap\n# No download, checkout, model load, or paid API. Requires NumPy.\nimport base64, hashlib, io, json, math, tempfile, zipfile\nfrom pathlib import Path\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="l195-stress-"))')
 md(fill(prose,True))
 md('## PROVIDED · Frozen evidence\nThe embedded archive preserves prediction arrays, receipts, original manifest chains, the L194 paper/task packet and source contracts. A matching hash proves identity to the snapshot; it does not prove historical feature availability. No earlier lesson folder is required.')
 code('payload='+repr(payload)+'\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():\n        if not (workspace/name).resolve().is_relative_to(workspace.resolve()): raise ValueError("Unsafe archive path")\n    z.extractall(workspace)\npacket=workspace/"packet"\nmanifest=json.loads((workspace/"input-manifest.json").read_text())',['data-payload'])
 for name,args,contract,check in contracts:
  md('## TODO · '+name+'\n\n'+contract+'\n\nPredict a failure case before running your CHECK. This function remains a live dependency of the full replay.')
  code(functions[name] if solution else 'def '+name+'('+args+'):\n    raise NotImplementedError('+repr(name)+')')
  md('### CHECK · Test the contract');code(check+'\nprint("Contract PASS")')
 md('## PROVIDED · Rank AUROC\nAverage tied ranks give half-credit for tied positive-negative scores. Full keys align repeated queries. This inherited visible scorer is independently checked against all positive-negative pairs by the repository verifier.');code(functions['keyed_auc'])
 md('## PROVIDED · Driver-cluster interval\nGroup query differences by driver, sample whole drivers with replacement, and retain query weighting in each draw. Models and split are held fixed. Read the group totals and counts: replacing these with unweighted driver means would change the estimand.');code(functions['cluster_interval'])
 md('## PROVIDED · Full replay operator\nFirst authenticate bytes and original receipt chains. Then rescore every regression run; check validation selection; align common float64 targets. Next check all ICL runs against released keys, labels and support draws. Finally authenticate all21RDBLearn task rows and preserve missing scores. Your three functions control the result; none is replaced.');code(operators['replay195'])
 md('## PROVIDED · Falsification brief renderer\nRead how a scoped numerical comparison becomes a qualified claim, counterargument and future revision condition.');code(operators['brief_markdown'])
 md('## CHECK · Complete replay\nCompare your report with the authenticated author-reference analysis. This certifies replay consistency, not model training or mastery.')
 code('report=replay195(packet,manifest,paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval)\nassert report==json.loads('+repr(json.dumps(r,sort_keys=True))+')\nPath("l195-report.json").write_text(json.dumps(report,indent=2))\nPath("l195-falsification-brief.md").write_text(brief_markdown(report))\nprint(brief_markdown(report))')
 md('## EXIT · Your own defense\nChoose C1–C4 and write the strongest objection and reply. Specify a new observation that would change your position: population, varied factor, held-fixed factors, metric/units, practical margin, uncertainty unit and invalidation conditions. Do not select a convenient margin from the existing results and call it preregistered. Send your defense to the teacher. The fields below stay empty in the author solution because they are your work.')
 code('submission=dict(learner="PENDING_WRITTEN_DEFENSE",claim="",counter_evidence="",strongest_reply="",population="",vary="",hold_fixed="",metric_units="",practical_margin="",uncertainty_unit="",revision_condition="",invalidation_conditions="")\nPath("l195-submission.json").write_text(json.dumps(submission,indent=2))')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l195-{i:03}'
 return book
for solution in [False,True]:
 path=P/('solutions' if solution else '')/(S+'.ipynb');book=notebook(solution)
 if solution and path.exists():
     old=nb.read(path,4);book.metadata=old.metadata
     previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
     if [c.source for c in previous]==[c.source for c in current]:
         for prior,c in zip(previous,current):
             c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('Built lesson/reference, three figures, and portable student/solution notebooks')
