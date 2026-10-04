"""Build the defense lesson, explanatory figures and standalone learner notebooks."""
import ast,base64,copy,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b24';F=P/'figures/b24';S='b24-architecture-thesis-defense';r=json.loads((E/'report.json').read_text())
plt.rcParams.update({'font.size':11,'figure.facecolor':'#fafcfb','axes.facecolor':'#fafcfb','axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
# Figure 1: expose a lost distinction and a stronger flat repair, not graph superiority.
fig=plt.figure(figsize=(10,8.2));grid=fig.add_gridspec(3,2,height_ratios=[1,1,1.15],hspace=.65,wspace=.36)
for i,(name,values) in enumerate([('History A',[0,1]),('History B',[.5,.5])]):
 ax=fig.add_subplot(grid[0,i]);ax.bar([0,1],values,color=['#397962','#397962']);ax.axhline(.5,color='#9e6230',linestyle='--');ax.set_ylim(0,1.16);ax.set_xticks([0,1],['event 1','event 2']);ax.set_ylabel('Observed value');ax.set_title(name+' · legal values',loc='left',weight='bold');ax.text(.97,.90,'mean = 0.5',transform=ax.transAxes,ha='right',color='#865320')
 ax=fig.add_subplot(grid[1,i]);res=(np.array(values)-.5)**2;ax.bar([0,1],res,color='#627daa');ax.set_ylim(0,.30);ax.set_xticks([0,1],['event 1','event 2']);ax.set_ylabel('(value − 0.5)²');ax.set_title('Square, then mean → variance '+str(float(res.mean())),loc='left',fontsize=11,weight='bold')
ax=fig.add_subplot(grid[2,:]);ax.axis('off')
for y,head,body in [(.90,'Mean-only flat path','records → [mean 0.5] → predictor: A and B indistinguishable'),(.57,'Stronger flat path','records → [mean, variance] → trees / TFM: A and B distinguishable'),(.24,'Learned graph path','records + edges → learned messages → task head: distinction possible, not guaranteed')]:
 ax.text(.0,y,head,weight='bold',color='#245541');ax.text(.0,y-.14,body,fontsize=11)
fig.suptitle('What information reaches the predictor?',x=.07,ha='left',fontsize=16,weight='bold');fig.subplots_adjust(left=.09,right=.97,top=.91,bottom=.035);fig.savefig(F/'architecture.png',dpi=150);plt.close(fig)
# Figure 2: every measured paired effect, no independent-dataset uncertainty claim.
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
for ax,c,color in zip(axes,r['comparisons'],['#b07b38','#34785f','#7b639d']):
 v=np.array(c['per_seed']);ax.axhline(0,color='#58645f',lw=1);ax.plot(range(10),v,'o-',color=color);ax.axhline(c['mean'],color=color,linestyle='--',alpha=.8);ax.set_ylim(-.04,max(max(c['per_seed']) for c in r['comparisons'])+.02);ax.set_ylabel('Δ AUROC');ax.set_title(f"RDB-PFN − {c['reference']}  ·  mean {c['mean']:+.6f}  ·  positive {c['positive']}/10",loc='left',fontsize=11,weight='bold');ax.grid(alpha=.15)
axes[-1].set_xticks(range(10));axes[-1].set_xlabel('Paired support draw · same 702 query rows throughout')
fig.suptitle('B23 evidence, independently replayed in B24',x=.10,ha='left',fontsize=15,weight='bold');fig.text(.1,.015,'Dashed line = mean; connected dots = individual draws. These are not confidence intervals.',fontsize=10);fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(F/'effects.png',dpi=150);plt.close(fig)
# Figure 3: demonstrate non-compensating gates beside scores.
fig,ax=plt.subplots(figsize=(10,5.5));ax.axis('off');rows=[['10 / 10','No','Clear','NOT ELIGIBLE'],['10 / 10','Yes','Unresolved','NOT ELIGIBLE'],['8 / 10, one zero','Yes','Clear','NOT ELIGIBLE'],['8 / 10, no zero','Yes','Clear','ELIGIBLE*']]
t=ax.table(cellText=rows,colLabels=['Rubric','Required evidence','Leakage','Eligibility'],cellLoc='left',colLoc='left',bbox=[.01,.25,.98,.58]);t.auto_set_font_size(False);t.set_fontsize(11)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('#a7b8ad');cell.set_facecolor('#dcece3' if i==0 else '#f7faf8');cell.PAD=.07
 if i==0:cell.set_text_props(weight='bold')
ax.text(.01,.97,'Which missing fact can a high score compensate for?',va='top',weight='bold',fontsize=15)
ax.text(.01,.88,'None of the required evidence gates.',va='top',fontsize=12,color='#245541')
ax.text(.01,.16,'*Eligibility is hypothetical. Human assessment of the submitted work is still required.',fontsize=11)
ax.text(.01,.08,'Actual learner: PENDING_WRITTEN_DEFENSE. B23 historical availability remains unestablished.',fontsize=10)
fig.subplots_adjust(left=.035,right=.975,top=.96,bottom=.04);fig.savefig(F/'gates.png',dpi=150);plt.close(fig)

def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/f'{name}.png').read_bytes()).decode() if portable else f'../labs/figures/b24/{name}.png'
 return f'<figure class="defense-figure" tabindex="0" role="region" aria-label="{caption}" style="max-width:100%;overflow-x:auto"><img src="{src}" alt="{caption}" style="display:block;width:100%;min-width:700px;height:auto"><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>'
table='| Arm | Mean AUROC | Sample SD over supports | Evidence lane |\n|---|---:|---:|---|\n'
for arm,v in r['models'].items():table+=f"| {arm} | {v['mean']:.6f} | {v['sample_sd']:.6f} | {'Course logistic baseline' if arm=='Logistic' else 'Selected published configuration'} |\n"
axes=['Protocol','Baselines','Reproducibility','Interpretation','Falsifiability']
gatewidget='<section class="defense-board" data-defense-board><h3>Try to compensate for a missing gate</h3><p class="scope">Hypothetical scores and evidence flags. Changing these controls never changes B23 provenance or assesses your work.</p><div class="defense-controls">'+''.join(f'<label>{a}<select data-score name="{a.lower()}"><option value="0">0 · missing</option><option value="1">1 · partial</option><option value="2" selected>2 · justified</option></select></label>' for a in axes)+'''<label>Required reproduction complete<input name="reproduction" type="checkbox"></label><label>Unresolved leakage<input name="leakage" type="checkbox" checked></label><button type="button">Reset</button></div><output aria-live="polite">Hypothetical rubric 10/10: NOT ELIGIBLE. Reproduction pending; unresolved leakage. Actual learner: pending written defense.</output><noscript><p>Manually remove both evidence blockers, then try scores [2,2,2,2,0]. The total is8 but the zero still blocks eligibility.</p></noscript></section>'''
falsifywidget='''<section class="defense-board" data-falsification-board><h3>Predict when the reversal rule fires</h3><p class="scope">A hypothetical future comparison. Change the threshold only to explore a different preregistration, never after observing real test results.</p><div class="defense-controls"><label>Illustrative effect: <span data-effect>-0.005</span><input name="effect" type="range" min="-0.03" max="0.03" step="0.005" value="-0.005"></label><label>Frozen threshold: <span data-threshold>0.000</span><input name="threshold" type="range" min="0" max="0.02" step="0.005" value="0"></label><label>Information and selection budget matched<input name="matched" type="checkbox" checked></label><button type="button">Reset</button></div><output aria-live="polite">Illustrative effect −0.005 ≤ threshold0.000: REVISE the task-specific superiority claim. Proposed tests NOT_RUN.</output><noscript><p>A tie at threshold also triggers revision. If information or selection budgets differ without a declared design, the result cannot answer the matched-comparison question.</p></noscript></section>'''
body=(R/'lessons/content'/f'{S}.md').read_text();coverage=(P/'sources/b24/coverage.md').read_text()
def fill(text,portable=False):
 for k,v in dict(RESULTS=table,ARCHITECTURE=figure('architecture','Illustrative histories: variance repairs a distinction lost by mean-only flattening.',portable),EFFECTS=figure('effects','All ten paired effects against each comparator; one reused task, not ten databases.',portable),GATES=figure('gates','Illustrative eligibility cases; missing validity cannot be repaired with rubric points.',portable),COVERAGE=coverage,DEFENSE_WIDGET='**Manual gate exercise:** try10/10with missing reproduction, then8/10with one zero. Explain why neither is eligible; keep learner assessment separate.' if portable else gatewidget,FALSIFICATION_WIDGET='**Manual reversal exercise:** threshold0, effects−.005,0,+.005. Which trigger revision? What changes when information access is unmatched?' if portable else falsifywidget).items():text=text.replace('{{'+k+'}}',v)
 return text

def document(title,md,interactive=False,template=False):
 html=render(md).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable evidence table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','research-defense'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','benchmark-evidence','research-defense'])+'</head><body><article'+(' class="defense-template"' if template else '')+'><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B24 · Defend the architecture and the thesis',fill(body),True))
(R/'reference/b24-proposal-template.html').write_text(document('B24 · Five-page proposal template',(P/'b24-proposal-template.md').read_text(),template=True))
ref='''# B24 · Defense field guide

[Lesson](../lessons/b24-architecture-thesis-defense.html) · [Notebook](../labs/b24-architecture-thesis-defense.ipynb) · [Proposal template](b24-proposal-template.html) · [Frozen contract](../labs/b24-reproduction.md)

**Argument:** claim → evidence → warrant → revision condition. Name the population, operation and comparator. A numerical advantage does not identify its cause.

**Information-path check:** legal records → representation → learned operation → output. Histories[0,1]and[.5,.5]share mean.5; variances.25and0restore the distinction. A strong flat feature can defeat a weak architecture argument.

**B23/B24 scope:**40runs,28,080predictions;10paired512-support draws,702queries. B24 saved-evidence replay of B23 fresh selected-release evaluation. AUROC ranks positives above negatives with half credit for ties. SupportSD is not cross-database uncertainty.

'''+table+'''
**Observed:** RDBPFNminusTabICL+.004366,positive6/10draws; versus fixed logistic+.078643,positive10/10. Neither comparison establishes graph-native superiority or tuned-tree superiority.

**Baseline inventory:** tuned trees+time-safeFE; RealMLP/TabM/TabPack; retrieval; TabPFN/TabICL/TabDPT/Mitra; semantic models; hypernetworks; scaling alternative; RDBLearn/TabPFN-Rel; RelGNN/RelGT; RT/relationalFMs. Record LimiX/TabFM/EXAONE/Nori/RT-J and Seldon/NEXUS. SAP-RPT-1-OSS is the ConTextTab alias. Include, condition or exclude with a reason and reopening condition. See lesson for primary sources.

**Two reversal rules:** simpler-baseline win and untouched-task failure. Name task, models, metric, direction, numerical threshold, matched information/selection budget and action. Positive oriented effect favors the challenger; <=threshold triggers revision. Verify prior-use history; another F1seed is not untouched.

**Contracts:** B18a support/cache/update/rebuild/latency/cost; B19a metric/distribution/calibration; B19b forecast-time inputs only if relevant.

**Gates:** >=8/10; no zero; no unresolved leakage; reproduction appropriate to claim. Teacher reads the actual defense. Status PENDING_WRITTEN_DEFENSE until assessed.

**Open evidence:** historical identity/availability NOT_ESTABLISHED; fullDFS/pretraining/wholepaper/untouchedtask NOT_RUN. B24 fresh inference NOT_RUN. USD0paid;3600aggregate local seconds. [Primary Table9](https://arxiv.org/html/2603.03805v5#A6).

**Handoff:** L201hypothesis; L202sources; L203protocol; L204baselines; L219stress tests. Ask the agent for a skeptical review; retrieve after1/7/30days.
'''
(R/'reference'/f'{S}.html').write_text(document('B24 defense reference',ref))
# Compact deterministic bundle includes exact original B23 archive, no new source downloads.
files={'b23-packet.zip':(P/'evidence/b23/portable-packet.zip').read_bytes(),'source-identity.json':(E/'source-identity.json').read_bytes(),'report.json':(E/'report.json').read_bytes(),'_audit_b24.py':(P/'_audit_b24.py').read_bytes(),'_test_b24.py':(P/'_test_b24.py').read_bytes(),'relkit/defense_b24.py':(P/'relkit/defense_b24.py').read_bytes(),'relkit/__init__.py':b''}
for name in ['b24-reproduction.md','b24-proposal-template.md']:files[name]=(P/name).read_bytes()
for p in (P/'sources/b24').glob('*'):files['sources/'+p.name]=p.read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in sorted(files.items()):
  info=zipfile.ZipInfo(name,date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
raw=buf.getvalue();(E/'portable-packet.zip').write_bytes(raw);manifest={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
source=(P/'relkit/defense_b24.py').read_text();nodes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
audit=(P/'_audit_b24.py').read_text();audits={n.name:ast.get_source_segment(audit,n) for n in ast.parse(audit).body if isinstance(n,ast.FunctionDef)}
header=re.sub(r'<div id="[^"]+"></div>','',fill(body,True));header=re.sub(r'\]\((\.\./[^)]+|b\d[^)]+\.html)\)',lambda m:'](https://avistian.github.io/relational/'+(m.group(1)[3:] if m.group(1).startswith('../') else 'lessons/'+m.group(1))+')',header)
cells=[nb.v4.new_markdown_cell(x) for x in re.split(r'(?=^## )',header,flags=re.M) if x.strip()]
cells += [nb.v4.new_markdown_cell('## PROVIDED · offline evidence and environment\nTier B: real released relational task evidence; no new inference. Only Python and NumPy are needed. Install NumPy if absent. The base64 packet below is hidden in prepared HTML; all audit operations and learner functions are visible. The three TODOs must drive your report. This notebook is a standalone replay, not a model training lab.'),nb.v4.new_code_cell('import base64, hashlib, io, json, math, sys, tempfile, zipfile\nfrom pathlib import Path\nimport numpy as np\n# @colab-bootstrap: embedded authenticated evidence; no checkout or network required.'),nb.v4.new_code_cell('packet=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\nassert hashlib.sha256(packet).hexdigest()=='+repr(hashlib.sha256(raw).hexdigest())+'\nmanifest='+repr(manifest)+'\nroot=Path("b24-portable");root.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(packet)) as archive:\n    for name,digest in manifest.items():\n        content=archive.read(name);assert hashlib.sha256(content).hexdigest()==digest\n        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)\nsys.path.insert(0,str(root.resolve()))\nidentity=json.loads((root/"source-identity.json").read_text())\nARMS=["RDBPFN","RDBPFN_single","TabICLv1.1","Logistic"]',metadata={'tags':['hide-input']})]
cells += [nb.v4.new_markdown_cell('## PROVIDED · authenticate and independently score\n`unpack` requires the frozen digest before reading evidence. `rank_auc` uses average ranks: subtract the positive-rank sum under perfect bottom ranking, then divide by positive×negative pairs. Inspect the tie loop. `scan` checks every declared run and reconstructs the fixed baseline from its saved fit state.'),nb.v4.new_code_cell(audits['unpack']),nb.v4.new_code_cell(audits['rank_auc']),nb.v4.new_code_cell(audits['scan']),nb.v4.new_code_cell('assert rank_auc([1,0,1,0],[.8,.5,.5,.2])==.875\nwith tempfile.TemporaryDirectory() as td:\n    evidence=unpack(root/"b23-packet.zip",identity,td)\n    records,verification=scan(evidence)\nassert len(records)==40\nprint(verification)')]
tasks=[('paired_effect','Pair before interpreting','Return reference,challenger,seeds,per_seed,mean,sample_sd and positive. Reject missing/duplicate/nonfinite matched data. Pair by draw identity, never row order. Use sample SD; positive means higher challenger AUROC.',"paired=paired_effect(list(reversed(records)),'TabICLv1.1','RDBPFN')\nassert paired['positive']==6\nassert abs(paired['mean']-.004366369004050163)<1e-12"),('defense_gate','Keep rubric quality and evidence gates separate','Accept exactly five integer0–2scores (protocol,baselines,reproducibility,interpretation,falsifiability) and Boolean reproduction/leakage/assessed flags. Return total,eligible,blockers,learner. Blockers in order: SCORE_BELOW_8,ZERO_AXIS,REPRODUCTION_PENDING,UNRESOLVED_LEAKAGE. Unassessed learner remains PENDING_WRITTEN_DEFENSE; assessed work is PASS or REVISION_REQUIRED.',"scores=dict.fromkeys(['protocol','baselines','reproducibility','interpretation','falsifiability'],2)\nassert not defense_gate(scores,False,False,False)['eligible']\nassert defense_gate(scores,True,False,False)['learner']=='PENDING_WRITTEN_DEFENSE'"),('falsification_contract','Reject a test that cannot reverse the claim','Validate exactly one simpler_baseline and one untouched_task record. Require nonempty task,challenger,reference,metric,orientation,action; distinct models; orientation higher/lower; finite numeric threshold; both matching flags exactly True. Reject reused untouched tasks. Return ready=True,tests=2,execution=NOT_RUN. This structural check cannot prove prose truth or real prior-use history.',"from _test_b24 import fixtures\n_,tests=fixtures()\nassert falsification_contract(tests,['rel-f1/driver-dnf'])['execution']=='NOT_RUN'\ntry: falsification_contract(tests,['new-db/new-task'])\nexcept ValueError: pass\nelse: raise AssertionError('Reused task admitted')")]
for name,title,why,check in tasks:
 cells += [nb.v4.new_markdown_cell('## TODO · '+title+'\n**Goal and contract:** '+why+'\n\n**Why:** this decision affects whether your evidence supports the proposal. Implement the function before running CHECK. The examples are teaching fixtures, not a submitted research plan.'),nb.v4.new_code_cell(nodes[name].split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+name+'")',metadata={'learner_function':name}),nb.v4.new_code_cell('# CHECK\n'+check)]
cells += [nb.v4.new_markdown_cell('## PROVIDED · report wiring\nThe complete replay explicitly calls your three live functions. It also keeps historical validity and unexecuted experiments separate. Illustrative rubric inputs do not grade your writing.'),nb.v4.new_code_cell(audits['replay']),nb.v4.new_code_cell('from _test_b24 import checks\nassert checks(paired_effect,defense_gate,falsification_contract)=="PASS"\nresult=replay(root/"b23-packet.zip",identity,paired_effect,defense_gate,falsification_contract)\nassert result==json.loads((root/"report.json").read_text())\nPath("b24-replay.json").write_text(json.dumps(result,indent=2))\nfor c in result["comparisons"]:\n    print(f"RDB-PFN minus {c[\'reference\']}: {c[\'mean\']:+.6f}, positive {c[\'positive\']}/10")\nprint(result["execution"],result["learner"])'),nb.v4.new_markdown_cell('## EXIT · your written defense\nComplete the five-page template in the extracted packet. Supply a claim, architecture trace, coverage/baseline table, actual source/evidence links and two proposed reversal tests. Reserve and audit a genuinely untouched task; the new-db/new-task fixtures are artificial identifiers and do not satisfy that requirement. Explain why replay completion leaves historical availability open. Paste the proposal to the agent for assessment; automatic checks do not establish mastery.\n\n**Model computation:** B23’s original source, checkpoint-shaped model and runner are preserved inside `b23-packet.zip`, under `sources/b23/`, and explained in the [B23 portable lab](https://avistian.github.io/relational/labs/b23-declared-comparison.ipynb). This evaluation-only lesson adds visible defense/evidence mechanisms; it performs no fresh model computation.')]
for solved,dest in [(False,P/f'{S}.ipynb'),(True,P/'solutions'/f'{S}.ipynb')]:
 n=nb.v4.new_notebook(cells=copy.deepcopy(cells),metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 for i,c in enumerate(n.cells):
  c.id=f'b24-{i:03d}'
  if solved and c.metadata.get('learner_function'):c.source=nodes[c.metadata['learner_function']]
 nb.write(n,dest)
print('Built B24 lesson, reference, five-page template, 3figures and portable notebooks')
