"""Build B21 from canonical lesson, visible code and authenticated full evidence."""
import ast,base64,copy,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b21';F=P/'figures/b21';S='b21-structural-robustness'
r=json.loads((E/'diagnostic.json').read_text());paper=json.loads((E/'reproduction.json').read_text())
plt.rcParams.update({'font.size':11,'figure.facecolor':'#fafcf9','axes.facecolor':'#fafcf9','axes.spines.top':False,'axes.spines.right':False})
# Model-specific architecture: show matrix routing and one actual aggregation coordinate.
fig=plt.figure(figsize=(9,10));gs=fig.add_gridspec(3,1,height_ratios=[1.5,2,1.5]);ax=fig.add_subplot(gs[0]);ax.axis('off')
ax.set_title('Course model · a foreign key selects the message route',loc='left',weight='bold',fontsize=15)
A=np.zeros((6,4),dtype=int);A[np.arange(6),r['data']['original']]=1
rows=[[f'event{i}',r['data']['child_x'][i][0],*A[i]] for i in range(6)]
t=ax.table(cellText=rows,colLabels=['row','value','account0','account1','account2','future3'],cellLoc='center',bbox=[0,.08,1,.72]);t.auto_set_font_size(False);t.set_fontsize(10)
for (i,j),cell in t.get_celld().items():cell.set_edgecolor('#bdccc1');cell.set_facecolor('#e1eee5' if i<3 else '#fff9ee')
ax.text(0,-.06,'A: [6 events × 4 accounts]. Reverse graph R=Aᵀ. Future column stays zero.',fontsize=11)
ax=fig.add_subplot(gs[1]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
def box(x,y,w,h,title,detail,color='#e1eee5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#668874'));ax.text(x+.02,y+h-.035,title,weight='bold',fontsize=11,va='top');ax.text(x+.02,y+h-.125,detail,fontsize=10,va='top',linespacing=1.5)
box(.02,.72,.45,.23,'Events: 6 × 2 → 6 × 4','Numeric features × event projection')
box(.53,.72,.45,.23,'Accounts: 4 × 2 → 4 × 4','Numeric features × account projection')
box(.02,.32,.45,.25,'Event update','Mean of linked account vectors\n+ own path + bias → ReLU')
box(.53,.32,.45,.25,'Account update','Mean of linked event vectors\n+ own path + bias → ReLU')
for x1,x2 in [(.24,.76),(.76,.24)]:ax.add_patch(FancyArrowPatch((x1,.71),(x2,.58),arrowstyle='-|>',mutation_scale=16,color='#aa6729',connectionstyle='arc3,rad=.1'))
ax.text(.5,.63,'A and R route the messages',ha='center',fontsize=10,bbox=dict(facecolor='#fafcf9',edgecolor='none'))
ax.text(.04,.19,'Both types update simultaneously. Repeat twice; all weights remain fixed.',weight='bold',fontsize=11)
ax.text(.04,.055,'Account head: 4 channels → 1 scalar; score only accounts0,1,2.',fontsize=11)
ax=fig.add_subplot(gs[2]);ax.axis('off');ax.text(0,.91,'One coordinate through the operation',weight='bold',fontsize=12)
ax.text(0,.64,'Clean: account0 reads [1,3] → mean2; account1 reads [2,4] → mean3.\nMove session0: account0 reads [] → zero; account1 reads [1,3,2,4] → mean2.5.',linespacing=1.7)
ax.text(0,.31,'Paper / public tutorial pipeline',weight='bold',fontsize=12,color='#8b5928')
ax.text(0,.04,'Table + time encoders → trained 2-layer GraphSAGE → task head.\nTutorial: width128, sum aggregation, normalization. Course: width4, mean,\nno normalization, random weights, MSE. This is not the paper model.',fontsize=10,linespacing=1.5)
fig.tight_layout(h_pad=2.8);fig.savefig(F/'architecture.png',dpi=150);plt.close(fig)
# FK trace: not merely boxes; include paired before/after route and exact cell cost.
fig,axes=plt.subplots(1,2,figsize=(9,4.8));values=[1,3,2,5,4,6]
for ax,owners,title in zip(axes,[[0,0,1,2,1,2],[1,1,1,2,1,2]],['Clean owners','Move session0 · two FK cells']):
 ax.set_xlim(0,1);ax.set_ylim(-.6,6);ax.axis('off');ax.set_title(title,loc='left',weight='bold',fontsize=12)
 for i,p in enumerate(owners):
  y=5-i;ax.text(.02,y,f'e{i}  value{values[i]}',va='center',fontsize=11)
  color='#ad6428' if i<2 else '#547b67'
  ax.annotate('',xy=(.73,4.5-2*p),xytext=(.35,y),arrowprops=dict(arrowstyle='->',color=color,alpha=.8,connectionstyle='arc3,rad=.04'))
 for p in range(3):
  group=[values[i] for i,v in enumerate(owners) if v==p];mean=np.mean(group) if group else 0
  ax.text(.76,4.5-2*p,f'account{p}\nmean {mean:g}',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.3',fc='#e1eee5',ec='#73927f'))
 ax.text(.02,-.55,'e0 and e1 share session0; both must have one owner.',fontsize=9)
fig.suptitle('Schema consistency and prediction stability are different checks',x=.03,ha='left',weight='bold',fontsize=13);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(F/'fk-trace.png',dpi=150);plt.close(fig)
# Removal/addition arithmetic: signed contributions rather than add-only heuristic.
fig,axes=plt.subplots(1,2,figsize=(9,3.8),gridspec_kw={'width_ratios':[1.7,1]});vals=[-2,5,-7,11]
axes[0].bar(range(4),vals,color=['#aa632b','#397358','#aa632b','#397358']);axes[0].axhline(0,color='#4b5d52',lw=.8);axes[0].set_xticks(range(4),['Remove\nforward','Add\nforward','Remove\nreverse','Add\nreverse']);axes[0].set_ylabel('Contribution to local loss change');axes[0].set_ylim(-9,14)
for i,v in enumerate(vals):axes[0].text(i,v+( .6 if v>0 else -.7),f'{v:+}',ha='center',va='bottom' if v>0 else 'top',weight='bold')
axes[1].axis('off');axes[1].text(.02,.75,'Full replacement',weight='bold',fontsize=13);axes[1].text(.02,.55,'−2 + 5 − 7 + 11 = 7',fontsize=13);axes[1].text(.02,.29,'New edges alone give16.\nThat omits what was removed.\nA slope is not an exact finite jump.',fontsize=10,linespacing=1.6)
fig.suptitle('Worked derivative · include the entire legal edit',x=.03,ha='left',weight='bold',fontsize=13);fig.tight_layout();fig.savefig(F/'direction.png',dpi=150);plt.close(fig)
# Plot regret: common axis and all individual seeds, no fake confidence intervals.
fig,axes=plt.subplots(1,3,figsize=(9,4),sharey=True)
for seed,ax in enumerate(axes):
 for method,color,marker in [('random','#a57231','s'),('gradient','#42785e','o'),('exhaustive','#466d9c','^')]:
  rows=[x for x in r['conditions'] if x['seed']==seed and x['method']==method]
  ax.plot([x['budget'] for x in rows],[x['regret'] for x in rows],marker+'-',color=color,label=method)
 ax.set_title('Frozen random model · seed'+str(seed),fontsize=10,loc='left');ax.set_xticks([0,1,2]);ax.set_xlabel('Allowed FK edits');ax.grid(alpha=.2);ax.set_ylim(-.015,.42)
axes[0].set_ylabel('Regret in MSE (lower is better)');axes[2].legend(fontsize=9);fig.suptitle('Finite exhaustive reference · linearization can miss the worst case',x=.03,ha='left',fontsize=13,weight='bold');fig.tight_layout();fig.savefig(F/'results.png',dpi=150);plt.close(fig)

def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b21/'+name+'.png'
 return f'<figure class="fk-figure" tabindex="0" role="region" aria-label="{caption}" style="max-width:100%;overflow-x:auto"><img src="{src}" alt="{caption}" style="display:block;width:100%;min-width:680px;height:auto"><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>'
results='| Seed | Budget | Random MSE | Gradient MSE | Exhaustive MSE | Exhaustive edits used |\n|---|---:|---:|---:|---:|---:|\n'
for seed in range(3):
 for b in [0,1,2]:
  a={x['method']:x for x in r['conditions'] if x['seed']==seed and x['budget']==b}
  results+=f"| {seed} | {b} | {a['random']['loss']:.6f} | {a['gradient']['loss']:.6f} | {a['exhaustive']['loss']:.6f} | {a['exhaustive']['cost']} |\n"
widget='''<section class="fk-board" data-fk-board><h3>Predict which rule will reject this edit</h3><div class="fk-controls"><label>Rows to change<select name="unit"><option value="session">Session0: events0 and1</option><option value="single">Event0 only</option><option value="other">Event2 only</option></select></label><label>New owner<select name="dest"><option value="0">Account0 · time0</option><option value="1" selected>Account1 · time1</option><option value="2">Account2 · time2</option><option value="3">Account3 · future time12</option><option value="4">Account4 · missing</option></select></label><label>Allowed FK cells<select name="budget"><option>0</option><option>1</option><option selected>2</option></select></label><button type="button">Reset</button></div><div class="fk-pair"><section class="fk-state"><strong>Clean database</strong><ol data-before><li>owners:0,0,1,2,1,2</li></ol></section><section class="fk-state"><strong>Proposed database</strong><ol data-after><li>owners:1,1,1,2,1,2</li></ol></section></div><output aria-live="polite">ADMISSIBLE · 2 FK cells changed; allowance2. Both session0 rows move together to account1.</output><p data-preview>Clean account means:2,3,5.5. Edited means:0,2.5,5.5. These are aggregation values, not final predictions.</p><noscript><p>Manual check: a one-row session0 edit breaks its shared-owner rule. The two-row move needs budget2. Account3 arrives after all events; account4 does not exist.</p></noscript></section>'''
body=(R/'lessons/content'/f'{S}.md').read_text()
def fill(text,portable=False):
 items={'FK_WIDGET':'**Manual intervention:** compare owners[0,0,1,2,1,2] with[1,1,1,2,1,2]. Count changed cells, check session0 and timestamps, then recompute means.' if portable else widget,'ARCHITECTURE':figure('architecture','Course bidirectional GraphSAGE-style forward pass; paper/tutorial differences labeled.',portable),'FK_FIG':figure('fk-trace','The same feature values follow different valid owner links; two cells change together.',portable),'GRADIENT_FIG':figure('direction','Removal and insertion in both directions yield total slope7, not16.',portable),'RESULTS_FIG':figure('results','All three seeds and budgets; exhaustive regret is zero by definition.',portable),'RESULTS':results}
 for k,v in items.items():text=text.replace('{{'+k+'}}',v)
 return text

def document(title,text,interactive=False):
 html=render(text).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','structural-robustness'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','benchmark-evidence','structural-robustness'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B21 · Valid links, vulnerable predictions',fill(body),True))
ref='''# B21 · Structural robustness reference

[Lesson](../lessons/b21-structural-robustness.html) · [Notebook](../labs/b21-structural-robustness.ipynb) · [Contract](../labs/b21-reproduction.md)

| Check | Ask | Counterexample |
|---|---|---|
| FK existence | Does this parent exist? | A missing ID |
| Functional dependency | Does session still determine owner? | Only one of two coupled rows moved |
| Temporal eligibility | Was the parent available at event time? | Existing parent created later |
| Database budget | How many final FK cells differ? | One operation changes two rows |
| Graph consistency | Are both directions rebuilt from the database? | Stale reverse edge |
| Clean equivalence | Does the attackable model preserve clean predictions? | New bias parameters after permissive loading |

**Direction:** score = Σ G_forward×(A_new−A_old) + Σ G_reverse×(A_new−A_old)ᵀ. Remove old edges as well as adding new ones. Example−2+5−7+11=7. A correct derivative is not a guarantee about a finite edit.

**Bound:** exhaustive worst-case loss over at-mostB edits cannot decrease whenB grows. A particular heuristic's achieved loss may decrease. Regret = exact maximum−achieved loss. Include the clean state. The bound applies only to the enumerated model/input/threat set.

**Frozen course:**3random-network seeds ×3budgets ×3methods;105seed/states,27conditions. Mean squared error on three parent queries. No training/generalization claim.

'''+results+'''
**Historical target:** qualifying-position Table4:7methods ×5budgets ×5seeds. Public tutorial settings and saved outputs differ; current-runtime clean conversion fails.35printed cells and35saved rows audited, but historical run NOT_RUN / INCOMPLETE_SOURCE_PROTOCOL_GATE. Source audit and a local repair do not authenticate original checkpoints.

**Exit:** defend threat model, changed-cell counting, constraints, clean equivalence and evidence boundary. Learner status PENDING_WRITTEN_DEFENSE. Revisit1/7/30days.
'''
(R/'reference'/f'{S}.html').write_text(document('B21 reference',ref))
# Complete portable packet, no repository/network needed after dependencies are present.
paths=[p for p in E.iterdir() if p.is_file() and p.name not in ['local-budget.json','artifact-manifest.json','portable-packet.zip']]+[p for p in (P/'sources/b21').iterdir() if p.is_file()]
manifest={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(paths):
  info=zipfile.ZipInfo(str(p.relative_to(P)),date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
raw=buf.getvalue();(E/'portable-packet.zip').write_bytes(raw)
source=(P/'relkit/structural_b21.py').read_text();nodes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
header=re.sub(r'<div id="[^"]+"></div>','',fill(body,True))
header=re.sub(r'\]\((\.\./[^)]+|b\d[^)]+\.html)\)',lambda m:'](https://avistian.github.io/relational/'+(m.group(1)[3:] if m.group(1).startswith('../') else 'lessons/'+m.group(1))+')',header)
base=[nb.v4.new_markdown_cell(header),nb.v4.new_markdown_cell('## PROVIDED · standalone environment\nDefault lane: fresh complete finite experiment plus independent saved replay and source/table audit. There is no paper training or model download. Complete the three TODOs before running the later experiment. Tested author versions are recorded in the packet; live Colab is NOT_CHECKED.')]
base.append(nb.v4.new_code_cell('''# @colab-bootstrap: no repository checkout needed.
import importlib.util, subprocess, sys
missing=[package for module,package in [('numpy','numpy'),('torch','torch'),('bs4','beautifulsoup4')] if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import base64,copy,hashlib,io,itertools,json,re,zipfile
from pathlib import Path
import numpy as np
import torch
from bs4 import BeautifulSoup
torch.set_num_threads(1)
'''))
cell=nb.v4.new_code_cell(f'''packet=base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(packet).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
root=Path('b21-packet');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(packet)) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
E=root/'evidence/b21'
print('Authenticated',len(manifest),'source/evidence files')''');cell.metadata.update({'tags':['hide-input'],'jupyter':{'source_hidden':True}});base.append(cell)
base.append(nb.v4.new_code_cell(nodes['fixture']))
checks={
'changed_cells':'''assert changed_cells([0,0,1],[2,2,1])==2
assert changed_cells([0,0],[0,0])==0
try:changed_cells([0],[0,1])
except ValueError:pass
else:raise AssertionError('Mismatched shapes accepted')
print('CHECK: count cells, not coupled operations or directed edges')''',
'validate_assignment':'''d=fixture()
assert validate_assignment(d,d['original'],0)
assert validate_assignment(d,[2,2,1,2,1,2],2)
for a,b in [([2,0,1,2,1,2],2),([2,2,1,2,1,2],1),([3,3,1,2,1,2],2),([4,4,1,2,1,2],2),([-1,-1,1,2,1,2],2),([0.,0.,1.5,2.,1.,2.],2)]:
    try:validate_assignment(d,a,b)
    except ValueError:pass
    else:raise AssertionError('Invalid assignment accepted')
print('CHECK: coupled rule, timestamps, FK existence, type and budget')''',
'direction_score':'''gf=np.array([[2.,5.]]);gr=np.array([[7.],[11.]])
a=np.array([[1.,0.]]);b=np.array([[0.,1.]])
assert direction_score(gf,gr,a,b)==7
assert direction_score(gf,gr,a,a)==0
try:direction_score(gf,gr,np.zeros((2,2)),np.zeros((2,2)))
except ValueError:pass
else:raise AssertionError('Bad shapes accepted')
print('CHECK: remove old edges, add new edges, include reverse')'''}
for i,name in enumerate(checks,1):
 base.append(nb.v4.new_markdown_cell(f'## TODO{i} · `{name}`\nImplement the rule from the lesson. Preserve the signature. Invalid shapes, assignments or budgets should raise ValueError. This function is used by the complete experiment below.'))
 cell=nb.v4.new_code_cell(nodes[name]);cell.metadata['learner_function']=name;base.append(cell);base.append(nb.v4.new_code_cell(checks[name]))
for title,names in [('PROVIDED · graph identities and all legal states',['adjacency','legal_states','validate_graph']),('PROVIDED · fixed random weights and complete two-layer forward',['parameters','forward']),('PROVIDED · complete paired selection experiment',['experiment'])]:
 base.append(nb.v4.new_markdown_cell('## '+title+'\nTrace the code against the architecture and frozen contract. There is no training loop: the experiment freezes random weights and changes legal FK assignments.'))
 base.append(nb.v4.new_code_cell('\n\n'.join(nodes[n] for n in names)))
base.append(nb.v4.new_code_cell('''fresh=experiment()
saved=json.loads((E/'diagnostic.json').read_text())
assert len(fresh['conditions'])==27 and len(fresh['states'])==105
for actual,reference in zip(fresh['states'],saved['states']):
    assert actual['assignment']==reference['assignment'] and actual['seed']==reference['seed']
    np.testing.assert_allclose(actual['prediction'],reference['prediction'],rtol=1e-10,atol=1e-10)
for actual,reference in zip(fresh['conditions'],saved['conditions']):
    assert actual['assignment']==reference['assignment'] and actual['method']==reference['method']
Path('b21-fresh.json').write_text(json.dumps(fresh,indent=2)+'\\n')
print('Fresh finite experiment COMPLETE:105 model/states,27 comparisons')
for row in fresh['conditions']:
    if row['budget']==2:print(row['seed'],row['method'],'MSE',round(row['loss'],6),'edits',row['cost'],'regret',round(row['regret'],6))'''))
verify=(P/'_verify_b21.py').read_text();v={n.name:ast.get_source_segment(verify,n) for n in ast.parse(verify).body if isinstance(n,ast.FunctionDef)}
base.append(nb.v4.new_markdown_cell('## CHECK · independent full NumPy replay\nEnumerate six independent FK cells and reject illegal states. Recompute all105 model/state predictions without PyTorch or the course forward function.'))
base.append(nb.v4.new_code_cell(v['numpy_forward']+'\n\n'+v['audit']))
base.append(nb.v4.new_code_cell("verified=audit(saved)\nassert audit(fresh)['status']=='PASS'\nPath('b21-replay.json').write_text(json.dumps(verified,indent=2)+'\\n')\nprint({k:v for k,v in verified.items() if k!='scores'})"))
repro=(P/'_reproduce_b21.py').read_text();func=next(n for n in ast.parse(repro).body if isinstance(n,ast.FunctionDef) and n.name=='replay')
base.append(nb.v4.new_markdown_cell('## CHECK · primary source and saved tutorial output\nAuthenticate the archived files, extract all35 printed target cells and35 tutorial result rows. Their training/budget/seed settings differ. This is a source/output audit, not fresh paper inference.'))
base.append(nb.v4.new_code_cell(ast.get_source_segment(repro,func)))
base.append(nb.v4.new_code_cell("paper_report=replay(root/'sources/b21')\nassert paper_report==json.loads((E/'reproduction.json').read_text())\nPath('b21-paper-replay.json').write_text(json.dumps(paper_report,indent=2)+'\\n')\nprint(paper_report['status'],paper_report['historical_status'])\nfor row in paper_report['common_budget_comparison']:print(row)"))
base.append(nb.v4.new_markdown_cell('''## EXIT · defend the boundary
Explain why existing IDs are insufficient, why two coupled rows cost two, and why the differentiable model must match the clean one. Identify which methods require query answers. A finite exact search establishes only the worst case in its declared set. Submit your explanation for feedback; learner status remains PENDING_WRITTEN_DEFENSE.

## Historical execution gate
The named Table4 target requires authenticated original identities and verified source behavior. `--fresh` on the repository reproduction operator refuses execution while these are unresolved. The complete author code follows for inspection, not automatic execution. A repaired current release would be a separate reconstruction. Paid computeUSD0; no paper training has run.'''))
base.append(nb.v4.new_code_cell("RUN_HISTORICAL_PAPER=False\nif RUN_HISTORICAL_PAPER:raise RuntimeError('NOT_RUN: INCOMPLETE_SOURCE_PROTOCOL_GATE. Resolve b21-reproduction.md before training.')"))
base.append(nb.v4.new_markdown_cell('## Source-probe evidence · current runtime\nThe packet contains executed source probes. These are not rerun by the default notebook because PyG/RelBench are optional dependencies for that separate operator.'))
base.append(nb.v4.new_code_cell("probes=json.loads((E/'source-probes.json').read_text())\nprint(probes['versions'])\nfor row in probes['clean_conversion']:print('seed',row['seed'],'clean error',row['max_clean_embedding_error'],'zero-new-bias error',row['zero_new_bias_error'])\nprint('Saved current-runtime probes; original-paper identity remains unresolved')"))
base.append(nb.v4.new_markdown_cell('## Appendix · complete original model, trainer and attacks\nUnmodified author `code/utils.py`, pinned commit1bc60c7eee605fddbb0975a1c28c64ecc8ce515a. This readable source is archived inside the authenticated packet. It is not the course implementation and is not executed by this appendix.\n\n<details><summary>Read the complete original Python source</summary>\n\n```python\n'+(P/'sources/b21/utils.py').read_text()+'\n```\n\n</details>'))
for solution in [True,False]:
 cells=copy.deepcopy(base)
 if not solution:
  for cell in cells:
   if name:=cell.metadata.get('learner_function'):cell.source=nodes[name].split('\n',1)[0]+f'\n    raise NotImplementedError("Implement {name}")'
 for i,cell in enumerate(cells):cell.id=f'b21-{i:03d}'
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 nb.write(book,P/'solutions'/f'{S}.ipynb' if solution else P/f'{S}.ipynb')
print('Built B21 lesson, reference,4 figures and standalone student/solution notebooks')
