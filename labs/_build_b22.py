"""Build canonical lesson, four portable figures and standalone B22 notebooks."""
import ast,base64,copy,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b22';F=P/'figures/b22';S='b22-support-state-refinement'
r=json.loads((E/'diagnostic.json').read_text());verified=json.loads((P/'_verify_b22_results.json').read_text())
plt.rcParams.update({'font.size':11,'figure.facecolor':'#fafcfb','axes.facecolor':'#fafcfb','axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
def box(ax,x,y,w,h,title,lines,color='#e1eee8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor='#77948a',linewidth=1))
 ax.text(x+.018,y+h-.025,title,weight='bold',va='top',fontsize=11)
 ax.text(x+.018,y+h-.072,lines,va='top',fontsize=10,linespacing=1.45)
def arrow(ax,start,end):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,color='#407966',linewidth=1.5))
fig,ax=plt.subplots(figsize=(9,11));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(.01,.99,'RefineICL L24 · paper architecture',va='top',fontsize=16,weight='bold')
ax.text(.01,.95,'Trace: a labeled support row becomes memory for a later query.',va='top',fontsize=11)
box(ax,.02,.795,.96,.12,'Typed table input → feature tokens, dimension 256','Numeric / categorical / missing-value encoding; support labels known.\nQuery answers withheld. Feature interactions select up to 16 partners.')
arrow(ax,(.48,.79),(.48,.752))
box(ax,.02,.625,.59,.12,'Two feature-interaction rounds → RowCLS','Pair message: project → multiply → project\nThree row-encoder blocks compress features to a row state.')
box(ax,.66,.625,.32,.12,'Retain typed memory','Feature states kept\nbefore compression.','#fff0db')
arrow(ax,(.61,.685),(.65,.685));arrow(ax,(.31,.62),(.31,.57))
box(ax,.02,.37,.96,.19,'24 contextual blocks · row width 1024','LayerNorm(row states) → attention reads support states\nNormalized read ⊙ SiLU(state projection) → output projection → add\nBoth support and query states evolve; no standalone expanded FFN.\nMemory re-entry at 1/3 and 2/3 depth: gated reads, RMS bound 0.25.')
arrow(ax,(.83,.62),(.83,.565));arrow(ax,(.48,.365),(.48,.32))
box(ax,.02,.225,.96,.085,'Query head → class scores','Pretraining learns weights across tasks; inference keeps weights fixed.')
ax.text(.02,.182,'Course implementation · deliberately smaller and untrained',weight='bold',fontsize=12,color='#9a5c20')
ax.text(.02,.145,'12 support + 16 query rows × 2 numeric features → width 8\nSupport-only label embedding → 3 attention-gated blocks → 16 × 2 logits\nIntervene after block 2; read changed support memory in block 3.\nNo feature interactions, RowCLS, typed memory, learned normalization or training.',fontsize=11,va='top',linespacing=1.7)
fig.subplots_adjust(left=.045,right=.975,top=.97,bottom=.015);fig.savefig(F/'architecture.png',dpi=150);plt.close(fig)
# Scalar arithmetic ladder: geometric area displays weighting before gating.
fig,axes=plt.subplots(1,2,figsize=(9,4.3),gridspec_kw={'width_ratios':[1.05,1]});ax=axes[0]
ax.barh([1,0],[1.5,1.25],color=['#397c68','#bd793c'],height=.5)
ax.set_yticks([1,0],['Support 1: 0.75 × 2','Support 2: 0.25 × 5']);ax.set_xlim(0,2);ax.set_xlabel('Contribution to attention read');ax.set_ylim(-.6,1.6)
for y,v in [(1,1.5),(0,1.25)]:ax.text(v+.04,y,f'{v:.2f}',va='center',weight='bold')
ax=axes[1];ax.axis('off');ax.text(.03,.88,'Read  1.50 + 1.25 = 2.75',weight='bold',fontsize=12)
ax.text(.03,.64,'Scale  0.50 × 2.75 = 1.375',fontsize=12)
ax.text(.03,.40,'Add  4 + 1.375 = 5.375',weight='bold',fontsize=12)
ax.text(.03,.10,'Illustrative scalar gate, not a learned SiLU gate.\nChanging a state is not an accuracy measurement.',fontsize=10,linespacing=1.6)
fig.suptitle('Read → scale → residual update',x=.035,ha='left',weight='bold',fontsize=15);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(F/'operator.png',dpi=150);plt.close(fig)
# Matched intervention table with exact scalar continuation.
fig,ax=plt.subplots(figsize=(9,5.3));ax.axis('off');ax.set_title('Branch after one block; preserve the current query',loc='left',weight='bold',fontsize=15,pad=20)
rows=[['normal','[2, 5]','4','2.75','5.375'],['identity','[2, 5]','4','2.75','5.375'],['skip write','[1, 3]','4','1.50','4.750'],['permute updates','[3, 4]','4','3.25','5.625']]
t=ax.table(cellText=rows,colLabels=['Arm','Support state','Query state','Later read','Final state'],cellLoc='center',bbox=[0,.31,1,.50],colWidths=[.24,.20,.18,.18,.20]);t.auto_set_font_size(False);t.set_fontsize(11)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('#bacbc4');cell.set_facecolor('#e1eee8' if i==0 or j==2 else '#fff5e9' if i in [3,4] else '#fafcfb')
ax.text(0,.93,'Incoming support [1, 3] → actual output [2, 5]; updates are [1, 2].',fontsize=11)
ax.text(0,.18,'Only support memory changes. Later read weights stay [0.75, 0.25].',fontsize=11)
ax.text(0,.07,'Final scalar = preserved query 4 + gate 0.5 × later read.\nPermutation swaps updates before adding them to the original input states.',fontsize=11,linespacing=1.6)
fig.tight_layout();fig.savefig(F/'intervention.png',dpi=150);plt.close(fig)
# All paired cases; common CE units, with explicitly labeled magnified permute axis.
fig,axes=plt.subplots(1,2,figsize=(9,4.7));colors={'linear':'#2b7664','xor':'#b5752d','radial':'#5c70a1'}
for ax,mode,title in zip(axes,['skip','permute'],['Skip support write','Permute support updates']):
 for row in r['paired']:
  if row['mode']!=mode:continue
  episode=row['episode'];offset=((episode%8)-3.5)*.014+(episode//8-1)*.15
  ax.scatter(row['seed']+offset,row['delta_ce'],s=20,alpha=.8,color=colors[r['data'][episode]['family']])
 for seed in [0,1,2]:
  vals=[x['delta_ce'] for x in r['paired'] if x['mode']==mode and x['seed']==seed]
  ax.plot([seed-.29,seed+.29],[np.mean(vals)]*2,color='#202e28',linewidth=2)
 ax.axhline(0,color='#789388',lw=1);ax.set_xticks([0,1,2]);ax.set_xlabel('Fixed random-weight seed');ax.set_title(title,loc='left',fontsize=12);ax.set_ylim(-.18,.08);ax.grid(axis='y',alpha=.18)
axes[0].set_ylabel('Changed − normal query CE');
for family,color in colors.items():axes[1].scatter([],[],color=color,label=family,s=20)
axes[1].legend(loc='lower right',fontsize=9)
fig.suptitle('All 72 paired cases per intervention · shared vertical scale',x=.035,ha='left',fontsize=14,weight='bold');fig.text(.03,.01,'Above zero: worse CE. Below zero: better CE. Black bars: 24-episode means. No training or confidence interval.',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(F/'results.png',dpi=150);plt.close(fig)

def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b22/'+name+'.png'
 return f'<figure class="support-figure" tabindex="0" role="region" aria-label="{caption}" style="max-width:100%;overflow-x:auto"><img src="{src}" alt="{caption}" style="display:block;width:100%;min-width:680px;height:auto"><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>'
results='| Arm | Random seed | Mean ΔCE | Episode ΔCE range | Worse CE | Mean Δaccuracy (pp) |\n|---|---:|---:|---:|---:|---:|\n'
for x in verified['summary']:
 results+=f"| {x['mode']} | {x['seed']} | {x['mean_delta_ce']:+.6f} | [{x['min_delta_ce']:+.6f}, {x['max_delta_ce']:+.6f}] | {x['positive']}/24 | {x['mean_delta_accuracy_pp']:+.3f} |\n"
widget='''<section class="support-board" data-support-board><h3>Predict the later read before changing the memory</h3><div class="support-controls"><label>Support intervention<select name="mode"><option value="normal">Normal</option><option value="identity">Identity replacement</option><option value="skip" selected>Skip support write</option><option value="permute">Permute support updates</option></select></label><label>First support attention: <span data-attention>0.75</span><input name="attention" type="range" min="0" max="1" step=".25" value=".75"></label><label>Illustrative scalar gate: <span data-gate>0.50</span><input name="gate" type="range" min="0" max="1" step=".25" value=".5"></label><button type="button">Reset</button></div><div class="support-pair"><section class="support-state"><strong>Normal baseline</strong><div data-normal>Support[2,5]; query4; later read2.75; final5.375.</div></section><section class="support-state"><strong>Changed memory</strong><div data-changed>Support[1,3]; query4; later read1.50; final4.750.</div></section></div><output aria-live="polite">Normal5.375 · skip4.750 · difference−0.625. Current query4 is preserved.</output><noscript><p>Static default: skipping the write changes only support memory. Manually set gate0: both final states become4. Set identity: both become5.375.</p></noscript></section>'''
body=(R/'lessons/content'/f'{S}.md').read_text()
def fill(text,portable=False):
 items={'ARCHITECTURE':figure('architecture','Full paper topology and explicitly separate course implementation.',portable),'OPERATOR':figure('operator','Weighted read, gate and residual arithmetic in one scalar example.',portable),'INTERVENTION':figure('intervention','Same current query; different support memory changes the later read.',portable),'RESULTS_FIG':figure('results','Random-model course results, not the paper checkpoint experiment.',portable),'RESULTS':results,'WIDGET':'**Manual widget:** reproduce5.375,4.750 and5.625 from the scalar worked trace. Set the gate to zero and explain why all become4.' if portable else widget}
 for key,value in items.items():text=text.replace('{{'+key+'}}',value)
 return text

def document(title,text,interactive=False):
 html=render(text).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','support-state'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','benchmark-evidence','support-state'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B22 · The support set is working memory',fill(body),True))
ref='''# B22 · Support-state intervention reference

[Lesson](../lessons/b22-support-state-refinement.html) · [Notebook](../labs/b22-support-state-refinement.ipynb) · [Reproduction contract](../labs/b22-reproduction.md)

| Term | Operational meaning |
|---|---|
| Support | Labeled examples available to the predictor |
| Query | Rows whose answers are withheld during prediction |
| State | Per-row intermediate vector; can change with fixed model weights |
| Attention | Softmax similarity weights multiplied by support value vectors |
| Gate | State-dependent multiplier; SiLU is not a probability |
| Support-write intervention | Restore support inputs after a block, preserving query outputs |
| Identity control | Copy actual support outputs back; require identical predictions |
| Paired effect | Changed minus normal metric on the same query rows |

**Scalar trace:** normal support[2,5], skip[1,3], permuted updates[3,4]. Preserve current query4. Weights[.75,.25], gate.5 → final states5.375/4.750/5.625. This is arithmetic, not accuracy.

**Metric:** CE = mean(logsumexp(logits) − true-class logit). PositiveΔCE is worse; positiveΔaccuracy is better. Multiply fractional accuracy differences by100 for percentage points.

**Frozen course:**3random models ×24shared episodes ×4arms =288trajectories; no training. Never treat these as288independent trained models. Student functions: attention_read, replace_support, paired_effect.

'''+results+'''
**Historical target:** RefineICL v1 Appendix E.2, L24 15K, block12,72RBF episodes,1024supports/256queries. Paper-reportedΔCE+0.0510; our historical inference NOT_RUN / INCOMPLETE_SOURCE_PROTOCOL_GATE. Checkpoint and executable probe/episode identities are missing. [Primary source](https://arxiv.org/html/2609.27679v1#A5.SS2).

**Exit:** defend preserved routes, negative control, metric sign and missing historical identity. Internal sensitivity alone does not establish learned utility or generalization. Revisit1/7/30days; ask the agent for feedback.
'''
(R/'reference'/f'{S}.html').write_text(document('B22 reference',ref))
# Zip only immutable source and numerical evidence; no mutable run ledger.
paths=[p for p in (P/'sources/b22').iterdir() if p.is_file()]+[E/name for name in ['diagnostic.json','runtime.json','reproduction.json']]
manifest={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(paths):
  info=zipfile.ZipInfo(str(p.relative_to(P)),date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
raw=buf.getvalue();(E/'portable-packet.zip').write_bytes(raw)
source=(P/'relkit/refinement_b22.py').read_text();nodes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
header=re.sub(r'<div id="[^"]+"></div>','',fill(body,True));header=re.sub(r'\]\((\.\./[^)]+|b\d[^)]+\.html)\)',lambda m:'](https://avistian.github.io/relational/'+(m.group(1)[3:] if m.group(1).startswith('../') else 'lessons/'+m.group(1))+')',header)
base=[nb.v4.new_markdown_cell(header),nb.v4.new_markdown_cell('## PROVIDED · standalone environment\nNo repository checkout or model download. Install NumPy and BeautifulSoup if missing; all source/evidence and four figures are embedded. Course float64 results are checked with tolerance2e-12; exact environment versions are in the packet. Live Colab NOT_CHECKED.')]
base.append(nb.v4.new_code_cell('''import importlib.util,subprocess,sys
missing=[package for module,package in [('numpy','numpy'),('bs4','beautifulsoup4')] if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import base64,copy,hashlib,io,json,math,tarfile,zipfile
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
'''))
cell=nb.v4.new_code_cell(f'''packet=base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(packet).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
root=Path('b22-packet');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(packet)) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
E=root/'evidence/b22'
print('Authenticated',len(manifest),'files')''');cell.metadata.update({'tags':['hide-input'],'jupyter':{'source_hidden':True}});base.append(cell)
checks={
'attention_read':'''q=np.zeros((2,2));k=np.eye(2);v=np.array([[2.,4.],[6.,8.]])
np.testing.assert_allclose(attention_read(q,k,v),[[4,6],[4,6]],atol=1e-14)
np.testing.assert_allclose(attention_read(np.array([[1000.,0.]]),k,v),[[2,4]])
try:attention_read(q,k,np.ones((3,2)))
except ValueError:pass
else:raise AssertionError('Misaligned values accepted')
print('CHECK: normalized read, stability and shape contract')''',
'replace_support':'''before=np.arange(12.).reshape(4,3);after=before+np.arange(1,5)[:,None]
for mode in ['normal','identity','skip','permute']:
    got=replace_support(before,after,2,mode)
    np.testing.assert_array_equal(got[2:],after[2:])
    expected=after[:2] if mode in ['normal','identity'] else before[:2] if mode=='skip' else before[:2]+[[2],[1]]
    np.testing.assert_array_equal(got[:2],expected)
np.testing.assert_array_equal(before,np.arange(12.).reshape(4,3))
np.testing.assert_array_equal(after,before+np.arange(1,5)[:,None])
try:replace_support(before,after,2,'bad')
except ValueError:pass
else:raise AssertionError('Invalid mode accepted')
print('CHECK: query preservation, identity, skip, permuted updates and no mutation')''',
'paired_effect':'''labels=np.array([0,1]);normal=np.array([[2.,0.],[0.,2.]])
got=paired_effect(normal,normal[:,::-1],labels)
assert abs(got['delta_ce']-2)<1e-12 and got['delta_accuracy_pp']==-100
assert paired_effect(normal,normal,labels)['delta_ce']==0
try:paired_effect(normal,normal,np.array([0,2]))
except ValueError:pass
else:raise AssertionError('Invalid class accepted')
print('CHECK: paired metric signs, units and alignment')'''}
for i,(name,check) in enumerate(checks.items(),1):
 base.append(nb.v4.new_markdown_cell(f'## TODO{i} · `{name}`\nImplement this function from the worked trace. It drives all experiment trajectories or their paired scores below. Preserve the signature and reject invalid inputs with ValueError.'))
 cell=nb.v4.new_code_cell(nodes[name]);cell.metadata['learner_function']=name;base.append(cell);base.append(nb.v4.new_code_cell(check))
for title,names in [('PROVIDED · fixed weights and complete episode generator',['parameters','episodes']),('PROVIDED · visible forward pass and full experiment',['forward','experiment'])]:
 base.append(nb.v4.new_markdown_cell('## '+title+'\nNo training occurs. Query answers appear only in scoring. Read the state trace against the architecture.'))
 base.append(nb.v4.new_code_cell('\n\n'.join(nodes[n] for n in names)))
base.append(nb.v4.new_code_cell('''fresh=experiment();saved=json.loads((E/'diagnostic.json').read_text())
assert len(fresh['conditions'])==288 and len(fresh['paired'])==216
for a,b in zip(fresh['conditions'],saved['conditions']):
    assert (a['seed'],a['episode'],a['mode'])==(b['seed'],b['episode'],b['mode'])
    np.testing.assert_allclose(a['logits'],b['logits'],rtol=0,atol=2e-12)
Path('b22-fresh.json').write_text(json.dumps(fresh))
print('Fresh complete course experiment:288 trajectories; no paper inference')'''))
base.append(nb.v4.new_markdown_cell('## CHECK · query labels belong only to scoring\nChange labels after inference. Predictions stay fixed; a metric can change. This tests the software boundary, not every possible leakage path in a real data pipeline.'))
base.append(nb.v4.new_code_cell('''d=episodes()[0];w=parameters(0)
a=forward(w,d['support_x'],d['support_y'],d['query_x'])
flipped=1-np.asarray(d['query_y'])
b=forward(w,d['support_x'],d['support_y'],d['query_x'])
assert a==b
original_ce=paired_effect(a['logits'],a['logits'],np.asarray(d['query_y']))['baseline_ce']
flipped_ce=paired_effect(b['logits'],b['logits'],flipped)['baseline_ce']
assert abs(original_ce-flipped_ce)>1e-8
print('Same predictions; scoring changed:',original_ce,flipped_ce)'''))
verify=(P/'_verify_b22.py').read_text();vn={n.name:ast.get_source_segment(verify,n) for n in ast.parse(verify).body if isinstance(n,ast.FunctionDef)}
base.append(nb.v4.new_markdown_cell('## CHECK · independent scalar-loop oracle\nReconstruct every state, logit and score without calling the course forward pass. This also checks full coverage and the intervention boundary.'))
base.append(nb.v4.new_code_cell(vn['oracle']+'\n\n'+vn['audit']))
base.append(nb.v4.new_code_cell("verified=audit(saved)\nassert audit(fresh)['status']=='PASS'\nPath('b22-replay.json').write_text(json.dumps(verified,indent=2))\nprint(verified)"))
repro=(P/'_reproduce_b22.py').read_text();fn=next(n for n in ast.parse(repro).body if isinstance(n,ast.FunctionDef))
base.append(nb.v4.new_markdown_cell('## CHECK · source audit, separate from historical inference\nAuthenticate the arXiv bytes and inventory the archive. The supplement described by the paper was not recovered in these responses.'))
base.append(nb.v4.new_code_cell(ast.get_source_segment(repro,fn)))
base.append(nb.v4.new_code_cell("paper=source_audit(root/'sources/b22')\nassert paper==json.loads((E/'reproduction.json').read_text())\nprint(paper)\nRUN_HISTORICAL=False\nif RUN_HISTORICAL:raise RuntimeError('NOT_RUN: INCOMPLETE_SOURCE_PROTOCOL_GATE; recover authenticated checkpoint and probe first')"))
base.append(nb.v4.new_markdown_cell('## EXIT · defend the result\nExplain why only support states are restored, why the identity arm must match exactly, why a random update can hurt, and why the paper lane remains unrun. Submit your functions and defense to the agent. PENDING_WRITTEN_DEFENSE; no learner mastery inferred.'))
for solution in [True,False]:
 cells=copy.deepcopy(base)
 if not solution:
  for cell in cells:
   if name:=cell.metadata.get('learner_function'):cell.source=nodes[name].split('\n',1)[0]+f'\n    raise NotImplementedError("Implement {name}")'
 for i,cell in enumerate(cells):cell.id=f'b22-{i:03d}'
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 nb.write(book,P/'solutions'/f'{S}.ipynb' if solution else P/f'{S}.ipynb')
print('Built B22 HTML, reference,4 figures and portable student/solution notebooks')
