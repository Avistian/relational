"""Build B23 lesson, computation-specific figures and portable evidence lab."""
import ast,base64,copy,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b23';F=P/'figures/b23';S='b23-declared-comparison';r=json.loads((E/'report.json').read_text())
plt.rcParams.update({'font.size':11,'figure.facecolor':'#fafcfb','axes.facecolor':'#fafcfb','axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
def box(ax,x,y,w,h,title,body,color='#e3eee8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor='#80998d'))
 ax.text(x+.015,y+h-.025,title,weight='bold',va='top',fontsize=11)
 ax.text(x+.015,y+h-.070,body,va='top',fontsize=10,linespacing=1.5)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#367563',linewidth=1.5))
fig,ax=plt.subplots(figsize=(9,10));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(.015,.99,'RDB-PFN · from flat relational features to a query score',va='top',weight='bold',fontsize=15)
box(ax,.02,.80,.96,.125,'One context · 512 supports + 702 queries · 72 numeric features','Released DFS table; support medians fill missing values.\nSupport labels are known; query answers are withheld.')
arrow(ax,(.5,.795),(.5,.76))
box(ax,.02,.59,.96,.16,'Feature and target tokens → 1 × 1214 × 73 × 96','Feature scalar → support normalization → clamp ±100 → linear 1→96\nTarget slot: support label; query slot: support-label mean.\n73 slots per row = 72 features + one target token.')
arrow(ax,(.5,.585),(.5,.55))
box(ax,.02,.31,.96,.23,'Repeat six times · width 96 · four heads','① Feature attention: each row mixes its 73 tokens.\n② Row attention: each slot reads ONLY 512 support keys/values.\n    A query can read support labels through their evolving tokens.\n③ Feed-forward 96→192→96 with GELU.\nResidual addition + layer normalization follow each operation.')
arrow(ax,(.5,.305),(.5,.27))
box(ax,.02,.15,.96,.11,'Read query target tokens → 96→192→2 → class probabilities','Output: 702 × 2. Score column 1 against released labels.')
ax.text(.025,.10,'Training boundary: weights learned previously; no B23 optimizer update.',fontsize=11,weight='bold')
ax.text(.025,.052,'Single-table-prior checkpoint uses this same shape and relational test features.\nTabICL is a different released model; logistic is a separately fitted linear baseline.',fontsize=10,linespacing=1.5)
fig.subplots_adjust(left=.035,right=.975,top=.98,bottom=.015);fig.savefig(F/'architecture.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,4.6));axes[0].axis('off');axes[0].set_title('Join by both identity columns',loc='left',weight='bold')
t=axes[0].table(cellText=[['7','Jan 1','0.8'],['7','Feb 1','0.2'],['9','Jan 1','0.5'],['9','Feb 1','0.5']],colLabels=['Driver','Date','Score'],cellLoc='center',bbox=[0,.35,.95,.5]);t.auto_set_font_size(False);t.set_fontsize(11)
axes[0].text(0,.20,'Driver 7 occurs twice.\nDriver-only joins are ambiguous.',fontsize=11,linespacing=1.6)
M=np.array([[1,1],[.5,1]]);axes[1].imshow(M,cmap='Greens',vmin=0,vmax=1)
for i in range(2):
 for j in range(2):axes[1].text(j,i,str(M[i,j]),ha='center',va='center',color='white' if M[i,j]==1 else '#243f34',weight='bold',fontsize=16)
axes[1].set_xticks([0,1],['negative 0.5','negative 0.2']);axes[1].set_yticks([0,1],['positive 0.8','positive 0.5']);axes[1].set_title('Four pair credits → AUROC 0.875',loc='left',weight='bold',fontsize=12)
fig.suptitle('Identity first; ranking second · illustrative four-row arithmetic',x=.025,ha='left',fontsize=14,weight='bold');fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(F/'identity.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(9,8.3));colors=['#236c58','#b47935','#5c72a7','#945867']
for i,(name,m) in enumerate(r['models'].items()):
 axes[0].scatter(np.array(m['per_seed']),np.full(10,i)+np.linspace(-.12,.12,10),s=26,color=colors[i],alpha=.85)
 axes[0].plot([m['mean']]*2,[i-.22,i+.22],color='#182c22',linewidth=3)
axes[0].set_yticks(range(4),['RDB-PFN','Single-table prior','TabICL v1.1','Logistic (course)']);axes[0].invert_yaxis();axes[0].set_xlim(.4,.8);axes[0].set_xlabel('AUROC · full shown range 0.4–0.8');axes[0].set_title('Ten support draws per arm; black marks = means',loc='left',weight='bold');axes[0].grid(axis='x',alpha=.2)
for i,c in enumerate(r['comparisons']):
 axes[1].plot(range(10),c['per_seed'],marker='o',label='vs '+c['reference'],color=colors[i+1],linewidth=1)
axes[1].axhline(0,color='#555',linewidth=1);axes[1].set_xticks(range(10));axes[1].set_xlabel('Shared support draw');axes[1].set_ylabel('RDB-PFN − reference AUROC');axes[1].set_title('Pair before averaging; positive favors RDB-PFN',loc='left',weight='bold');axes[1].legend(fontsize=10);axes[1].grid(alpha=.15)
fig.text(.03,.012,'Same 702 test queries throughout. These are support-draw differences, not confidence intervals or independent datasets.',fontsize=9)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(F/'results.png',dpi=150);plt.close(fig)

def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b23/'+name+'.png'
 return f'<figure class="comparison-figure" tabindex="0" role="region" aria-label="{caption}" style="max-width:100%;overflow-x:auto"><img src="{src}" alt="{caption}" style="display:block;width:100%;min-width:680px;height:auto"><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>'
table='| Model | Fresh mean AUROC | Sample SD across supports | Published target |\n|---|---:|---:|---:|\n'
for a,m in r['models'].items():table+=f"| {a} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target'] if m['target'] is not None else 'Course baseline'} |\n"
widget='''<section class="comparison-board" data-comparison-board><h3>One shared support draw; one signed difference</h3><div class="comparison-controls"><label>Reference<select name="reference"><option>RDBPFN_single</option><option selected>TabICLv1.1</option><option>Logistic</option></select></label><label>Support draw: <span data-seed>0</span><input name="seed" type="range" min="0" max="9" step="1" value="0"></label><label><span>Remove one declared run</span><input name="missing" type="checkbox"></label><button type="button">Reset</button></div><output aria-live="polite">Draw0: RDB-PFN0.738398 − TabICL0.715698 = +0.022700 AUROC. Complete40-run grid; historical availability NOT_ESTABLISHED.</output><noscript><p>Manual check: draw2 gives0.703860−0.721466=−0.017606. A positive average need not win every draw. Removing one run leaves the full experiment incomplete.</p></noscript></section>'''
body=(R/'lessons/content'/f'{S}.md').read_text()
def fill(text,portable=False):
 for k,v in dict(ARCHITECTURE=figure('architecture','Checkpoint-compatible forward path; target answers never enter query inputs.',portable),IDENTITY=figure('identity','Complete keys and tie-aware pair credits answer different validity questions.',portable),RESULTS_FIG=figure('results','Fresh B23 predictions; course baseline is separately labeled.',portable),RESULTS=table,WIDGET='**Manual pairing:** use draw0 and draw2 from the per-draw evidence. Explain the opposite signs, then decide whether a missing run permits the full comparison.' if portable else widget).items():text=text.replace('{{'+k+'}}',v)
 return text

def document(title,text,interactive=False):
 html=render(text).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','b23-results','comparison-evidence'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','benchmark-evidence','comparison-evidence'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'assets/b23-results.js').write_text('window.B23Results='+json.dumps(r)+';\n')
(R/'lessons'/f'{S}.html').write_text(document('B23 · Reproduce one declared comparison',fill(body),True))
ref='''# B23 · Comparison audit reference

[Lesson](../lessons/b23-declared-comparison.html) · [Notebook](../labs/b23-declared-comparison.ipynb) · [Contract](../labs/b23-reproduction.md)

**Named scope:** RDB-PFN v5 Table9, rel-f1/driver-dnf,512supports,10paired draws,702queries.30published-model evaluations plus10course logistic fits. No checkpoint search or test tuning.

**Admit:** hashes → complete(entity,date)keys → shared support identities → withheld query labels → finite probabilities → independent metric → complete40-run grid.

**AUROC:** positive-negative wins plus half ties, divided by number of pairs. Example positives[.8,.5],negatives[.5,.2] →3.5/4=.875.

**Pair:** challenger minus reference on the same support draw. Mean and sampleSD over10draws; one shared test population, not10databases. Positive favors challenger. Standard deviation is not a confidence interval.

'''+table+'''
**Interpret:** RDBPFN−TabICL+.004366,positive6/10; versus single-prior+.057948 and logistic+.078643,positive10/10each. Logistic is fixed,untuned and trained on the same supports; no tuned-tree superiority claim.

**Separate:** complete selected release execution, numerical CLOSE, historical identity NOT_ESTABLISHED, raw feature availability NOT_ESTABLISHED, pretraining/fullDFS/wholepaper NOT_RUN, learner PENDING_WRITTEN_DEFENSE.

**Deviations:** checkpoint width96 versus appendix128; released labels complement current rawDNF reconstruction; rawDFS not regenerated; baseline NumPy2.5.0/sklearn1.9.0 versus pinned cloud1.26.4/1.6.1. Repeating this task is not independent-task evidence.

**Budget:** USD10aggregate,stop8/reserve2;5960reservedworker seconds plus2USDoverhead allowance→3.6909712USD envelope,not invoice;3600local seconds including retries.

**EXIT:** named claim, simpler baseline, missing provenance, untouched-task falsification. Ask the agent for feedback; revisit1/7/30days. [Primary Table9](https://arxiv.org/html/2603.03805v5#A6).
'''
(R/'reference'/f'{S}.html').write_text(document('B23 reference',ref))
# Deterministic standalone audit archive. Notebook report explicitly receives learner functions.
files={}
for folder in ['packet','pilot-1','full-1','baseline-1']:
 for p in (E/folder).rglob('*'):
  if p.is_file():files['evidence/b23/'+str(p.relative_to(E))]=p.read_bytes()
for name in ['packet-manifest.json','input-manifest.json','preflight.json','admission.json','remote-pilot.json','remote-full.json','report.json']:files['evidence/b23/'+name]=(E/name).read_bytes()
for name in ['_audit_b23.py','_test_b23.py','_verify_b23.py','relkit/exit_l200.py']:files[name]=(P/name).read_bytes()
for p in (P/'sources/b23').glob('*'):
 if p.is_file():files['sources/b23/'+p.name]=p.read_bytes()
source=(P/'relkit/comparison_b23.py').read_text();nodes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
stubs={name:code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+name+'")' for name,code in nodes.items()}
files['relkit/comparison_b23.py']=source.encode();files['relkit/__init__.py']=b''
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in sorted(files.items()):
  info=zipfile.ZipInfo(name,date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
raw=buf.getvalue();(E/'portable-packet.zip').write_bytes(raw);manifest={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
header=re.sub(r'<div id="[^"]+"></div>','',fill(body,True));header=re.sub(r'\]\((\.\./[^)]+|b\d[^)]+\.html)\)',lambda m:'](https://avistian.github.io/relational/'+(m.group(1)[3:] if m.group(1).startswith('../') else 'lessons/'+m.group(1))+')',header)
cells=[nb.v4.new_markdown_cell(header),nb.v4.new_markdown_cell('## PROVIDED · portable audit environment\nOnly NumPy is required for this offline rescore. Install it with `%pip install numpy` if missing. The embedded archive is authenticated before use. Fresh training/inference requires the separately pinned environment and commands; it is not launched below.'),nb.v4.new_code_cell('import base64, hashlib, io, json, sys, zipfile\nfrom pathlib import Path\nimport numpy as np\n# @colab-bootstrap: embedded evidence, no repository checkout needed.')]
cells.append(nb.v4.new_code_cell('packet=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\nassert hashlib.sha256(packet).hexdigest()=='+repr(hashlib.sha256(raw).hexdigest())+'\nmanifest='+repr(manifest)+'\nroot=Path("b23-portable");root.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(packet)) as archive:\n    for name,digest in manifest.items():\n        payload=archive.read(name);assert hashlib.sha256(payload).hexdigest()==digest\n        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)\nsys.path.insert(0,str(root.resolve()))\nevidence=root/"evidence/b23"',metadata={'tags':['hide-input']}))
# Visible scoring code and helpers; live TODOs are injected separately.
metric=(P/'relkit/exit_l200.py').read_text();tree=ast.parse(metric);auc=next(ast.get_source_segment(metric,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='keyed_auc')
cells += [nb.v4.new_markdown_cell('## PROVIDED · identity-aware metric\nTable9 reports AUROC. Inspect the join and half-credit tie rule. Neither line assumes the prediction rows arrive in label order.'),nb.v4.new_code_cell(auc),nb.v4.new_code_cell('assert keyed_auc([[1,0],[2,0],[3,0],[4,0]],[1,0,1,0],[[1,0],[2,0],[3,0],[4,0]],[.8,.5,.5,.2])==.875\nrecords=[]\nfor phase in ["pilot-1","full-1","baseline-1"]:\n    records.extend(json.loads((evidence/phase/"receipt.json").read_text())["records"])\nassert len(records)==40')]
tasks=[('admit_grid','Require every declared evaluation','A survivor-only average can hide failed runs. Require the exact model/draw identities and complete row counts; reject malformed records.',"assert admit_grid(records)==dict(runs=40,predictions=28080,paper_runs=30,course_runs=10)\ntry: admit_grid(records[:-1])\nexcept ValueError: pass\nelse: raise AssertionError('Missing run accepted')"),('paired_comparison','Join before subtraction','Order in a file is not experiment identity. Match by support draw and report challenger-minus-reference effects.',"paired=paired_comparison(list(reversed(records)), 'TabICLv1.1', 'RDBPFN')\nassert paired['positive']==6\nassert abs(paired['mean']-.004366369004050163)<1e-12"),('claim_status','Separate what was established','A numerical match cannot repair missing historical availability, and an author report cannot pass the learner.',"assert claim_status(True,True,False,False)['historical']=='NOT_ESTABLISHED'\nassert claim_status(False,True,False,False)['numerical']=='INCOMPLETE'\nassert claim_status(True,True,False,False)['learner']=='PENDING_WRITTEN_DEFENSE'")]
for name,title,why,check in tasks:
 cells += [nb.v4.new_markdown_cell('## TODO · '+title+'\n**Goal:** implement `'+name+'`.\n\n**Why:** '+why+'\n\n**Hint boundary:** the task specifies invariants, not implementation. Write your function, then use CHECK.'),nb.v4.new_code_cell(stubs[name],metadata={'learner_function':name}),nb.v4.new_code_cell('# CHECK\n'+check)]
cells += [nb.v4.new_markdown_cell('## CHECK · the report calls your functions\nRaw bytes, full identities and all predictions are authenticated before your evidence and comparison functions construct the result. Exact agreement below is replay of the fresh author packet, not another model inference.'),nb.v4.new_code_cell('from _test_b23 import checks\nassert checks(admit_grid,paired_comparison,claim_status)=="PASS"\nfrom _audit_b23 import audit\nresult=audit(evidence,admit_grid,paired_comparison,claim_status)\nassert result==json.loads((evidence/"report.json").read_text())\nPath("b23-replay.json").write_text(json.dumps(result,indent=2))\nfor arm,value in result["models"].items():\n    print(f"{arm:18s} AUROC {value[\'mean\']:.6f} ± {value[\'sample_sd\']:.6f} support SD")\nprint(result["claims"])'),nb.v4.new_markdown_cell('## EXIT · written defense\nWrite your narrow claim, baseline limitation, missing historical evidence and untouched-task falsification test. Explain the6/10paired result. Author outputs do not fill this answer; paste it to the agent for assessment.')]
# Visible model, runner and baseline appendices do not execute heavy model imports.
model=(P/'sources/b23/rdbpfn_l166.py').read_text();mt=ast.parse(model)
cells.append(nb.v4.new_markdown_cell('## Readable model appendix · released-shape RDB-PFN\nThis inherited source-shaped implementation is displayed for inspection; the fresh run uses the authenticated original source. Previous checkpoint parity is explicitly inherited. The course prior generator is omitted here because it is not the source of B23 weights. All inference operations are shown below.'))
for node in mt.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name not in ['relational_prior','dfs_summary']:
  cells.append(nb.v4.new_markdown_cell('### '+node.name+'\n```python\n'+ast.get_source_segment(model,node)+'\n```'))
for title,name in [('Unchanged released-checkpoint evaluation loop','_run_l166.py'),('Course logistic fit and convergence gate','_baseline_b23.py'),('Authenticated report wiring','_audit_b23.py')]:
 cells.append(nb.v4.new_markdown_cell('## '+title+'\n```python\n'+(P/name).read_text()+'\n```'))
for solved,dest in [(False,P/f'{S}.ipynb'),(True,P/'solutions'/f'{S}.ipynb')]:
 n=nb.v4.new_notebook(cells=copy.deepcopy(cells),metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 for i,c in enumerate(n.cells):
  c.id=f'b23-{i:03d}'
  if solved and c.metadata.get('learner_function'):c.source=nodes[c.metadata['learner_function']]
 nb.write(n,dest)
print('Built B23 lesson, reference,3figures and student/solution notebooks')
