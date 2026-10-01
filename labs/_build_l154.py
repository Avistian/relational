"""Deterministic lesson/reference, figures and standalone evidence-replay notebooks."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _replay_l154 import replay,render_report
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l154';S='0154-portfolio-synthesis';TITLE='Portfolio synthesis: make every claim auditable'
manifest=json.loads((E/'input-manifest.json').read_text());report=replay(P,manifest)
(E/'report.json').write_text(json.dumps(report,indent=2)+'\n');(E/'report.md').write_text(render_report(report))
F=P/'figures/l154';F.mkdir(parents=True,exist_ok=True)
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="390" viewBox="0 0 960 390" role="img" aria-label="Evidence moves through integrity, comparability and completion gates into a bounded claim">
<rect width="960" height="390" rx="14" fill="#f3f6f3"/>
<g font-family="sans-serif"><text x="34" y="38" fill="#315d55" font-size="13" letter-spacing="2">FROM SAVED PREDICTIONS TO A DEFENSIBLE SENTENCE</text>
<text x="34" y="76" fill="#152f2b" font-size="26">A number becomes evidence only with its contract.</text>
<g fill="white" stroke="#bbcfca"><rect x="34" y="108" width="270" height="164" rx="9"/><rect x="345" y="108" width="270" height="164" rx="9"/><rect x="656" y="108" width="270" height="164" rx="9"/></g>
<g fill="#315d55" font-size="14"><text x="54" y="138">01 · INTEGRITY</text><text x="365" y="138">02 · COMPARABILITY</text><text x="676" y="138">03 · COMPLETENESS</text></g>
<g fill="#162f2a" font-size="18"><text x="54" y="174">55 pinned input files</text><text x="365" y="174">One task, one metric</text><text x="676" y="174">2 of 3 test tasks</text></g>
<g fill="#50655f" font-size="14"><text x="54" y="208">61,148 rows rescored</text><text x="54" y="238">Keys: entity + cutoff</text><text x="365" y="208">Preserve units and origins</text><text x="365" y="238">Do not pool training lanes</text><text x="676" y="208">0 fresh matched comparators</text><text x="676" y="238">Pilot stays validation only</text></g>
<g fill="#315d55" font-size="29"><text x="315" y="199">→</text><text x="626" y="199">→</text></g>
<rect x="34" y="299" width="892" height="57" rx="8" fill="#173f37"/><text x="55" y="334" fill="white" font-size="18">Bounded conclusion: report replay PASS · local superiority NOT_ESTABLISHED</text>
</g></svg>'''
(F/'flow.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(F/'flow.png'))
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l154','font.size':10})
fig,axes=plt.subplots(1,3,figsize=(11.4,3.5),layout='constrained');fig.patch.set_facecolor('#f3f6f3')
for ax in axes:ax.set_facecolor('#f3f6f3');ax.spines[['top','right']].set_visible(False)
for ax,name,title,scale,label in [(axes[0],'classification_reference','Classification · AUROC',100,'AUROC (%) ↑'),(axes[1],'regression_reference','Regression · MAE',1,'Finishing-position MAE ↓')]:
 rows=report['runs'][name]['test'];ax.scatter([r['seed'] for r in rows],[r['score']*scale for r in rows],color='#176a5c',s=55,zorder=3)
 avg=report['summaries'][name]['test']['mean']*scale;ax.axhline(avg,color='#176a5c',ls='--',lw=1)
 ax.set(title=title,xlabel='Training seed',ylabel=label,xticks=list(range(5)));ax.grid(axis='y',alpha=.15)
axes[2].axis('off');axes[2].text(.04,.87,'Recommendation · MAP@10',fontsize=12,weight='bold',transform=axes[2].transAxes)
axes[2].text(.04,.64,'TEST NOT_RUN',fontsize=20,color='#945c29',weight='bold',transform=axes[2].transAxes)
axes[2].text(.04,.45,'32-batch validation pilot only\nNo completed five-seed test lane\nMissing does not mean zero',fontsize=11,linespacing=1.7,va='top',transform=axes[2].transAxes)
fig.suptitle('Different questions. Separate scales. One explicit coverage gap.',fontsize=15,color='#173f37')
fig.savefig(F/'scores.svg',metadata={'Date':None});fig.savefig(F/'scores.png',dpi=150,metadata={'Software':'Lesson154'});plt.close(fig)
full_report=render_report(report);table=full_report.split('| Task / metric',1)[1].split('\n\n',1)[0];table='| Task / metric'+table
captions={'flow':'The report checks integrity, comparability and completion separately. Passing one gate cannot substitute for the others.',
          'scores':'Actual reference test scores, five seeds per completed task. Dashed lines are seed means; the third panel preserves the unrun test.'}
replay_text=f"""**Replay result:** {report['total_prediction_rows']:,} rows independently rescored: 8,925 classification reference, 8,925 classification selected, 6,295 regression and 37,003 recommendation pilot rankings. All 15 completed-fit validation checkpoint selections passed. The primary test means are **69.2588% AUROC** and **3.9708 MAE**; both fall inside their previously frozen descriptive paper bands. That is not statistical equivalence or historical reproduction.

The separate classification course-selected lane is **68.4570 ± 1.3502% AUROC**. It does not become the reference result. Recommendation validation MAP@10 is **1.050815%** after 32 training batches; no test result exists. [Machine-readable replay](../labs/evidence/l154/report.json)."""

def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[TABLE]]',table).replace('[[REPLAY]]',replay_text)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l154/'+name+'.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 text=text.replace('[[WARMUP]]','Before reading on: why is cutoff part of identity? Which split selects a checkpoint? What does a missing seed change?' if portable else '<div id="warmup"></div>')
 text=text.replace('[[PREDICT]]','Predict before checking: does lower MAE than a published result establish a fresh matched win? Explain why the evidence origin matters.' if portable else '<div id="predict"></div>')
 text=text.replace('[[WIDGET]]','Synthetic intervention: MAE 3 vs4 gives +1 position benefit. With a published comparator, this is context only; with a matched local comparator, it is a descriptive local difference. With incomplete runs, no final comparison is admitted. Try changing these fields in your function tests.' if portable else '<div class="portfolio-evidence" id="l154-evidence"></div><noscript>Synthetic MAE3 vs4 gives +1 position benefit. Published baseline: context only. Matched local fixture: descriptive difference. Incomplete runs: no final comparison.</noscript>')
 text=text.replace('[[TEACHBACK]]','Write the defense before reviewing the teacher answer. Submit it to the teaching agent.' if portable else '<div id="teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 body=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','portfolio-evidence','l154-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 154 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+name+'.css">' for name in ['lesson','atomic-route','checkpoint','portfolio-evidence'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0153-recommendation-portfolio.html">Lesson 153</a></nav><header><p class="route-kicker">Year 4 · Quarter 4 · Lesson 154</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+name+'.js"></script>' for name in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Contract before comparison
Task + metric + unit + direction + split + query population + evaluation rule + training lane + complete seeds + origin. Align by `(entity, cutoff)`; reject duplicate/missing keys. Pin inputs before replay.

| Question | Rule |
|---|---|
| What is the average run? | Mean across complete declared seeds in one lane |
| How variable are seeds? | Sample SD, denominator n−1; not cross-database uncertainty |
| Is the difference favorable? | Higher: model−baseline; lower: baseline−model |
| How large relatively? | 100×oriented gap/baseline; undefined at zero |
| Can raw metrics be averaged? | No: AUROC, MAE and MAP have different meanings and units |
| Does a paper baseline count as local? | No: PUBLISHED_CONTEXT_ONLY |
| Does a pilot count as complete? | No: validation-only pilot preserves missing test |
| Can matching scores prove equivalence? | No: descriptive tolerance is not statistical equivalence |

## Current measured portfolio
'''+table+'''

Two of three completed tasks; zero fresh matched comparator tasks. Local superiority NOT_ESTABLISHED. Selected classification seeds10–14 remain separate from reference seeds0–4. L153 full reproduction INCOMPLETE; whole paper/fresh FE/fresh trees NOT_RUN. The five-task broad curriculum ambition and three-task minimum remain unmet.

## Rebuild and inspect
Run `labs/_replay_l154.py` from the pinned repository or execute the standalone solution notebook. It verifies55 input files and rescores61,148 prediction rows. The notebook passes your three functions into the same replay adapter; it contains no training gate or cloud dispatch.

[Lesson](../lessons/0154-portfolio-synthesis.html) · [Generated report](../labs/evidence/l154/report.md) · [Contract](../labs/l154-reproduction.md) · [Student notebook](../labs/0154-portfolio-synthesis.ipynb) · [RelBench Tables 6–8](https://arxiv.org/html/2407.20060v1#A2.SS1).

## Next experiment and defense
Fresh matched FE/tree comparisons and completion of the recommendation test lane are separate missing work. Prior human effort and unitemized cloud overhead cannot be reconstructed from training seed SD. Defend provenance, units, completeness, uncertainty and conclusion (0–2 each;8/10 with no zero). Learner PENDING_WRITTEN_DEFENSE. Ask the teaching agent to review your exported report and explanation.
'''
(R/'reference/portfolio-synthesis.html').write_text(doc('Portfolio synthesis reference',reference))

def definitions(path):
 source=path.read_text()
 return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
functions=definitions(P/'relkit/portfolio_l154.py');checks=definitions(P/'_check_l154.py');replayers=definitions(P/'_replay_l154.py')
packet={name:base64.b64encode((P/name).read_bytes()).decode() for name in manifest['files']}
packed=base64.b64encode(zlib.compress(json.dumps(packet,sort_keys=True).encode(),9)).decode()
bootstrap='''# PROVIDED: CPU-only replay. Requires Python3.10+ and numpy; no network at runtime.
# Tested dependencies are recorded in labs/_execution_l154_results.json.
import base64,copy,gzip,hashlib,json,math,statistics,tempfile,zlib
from pathlib import Path
import numpy as np
from IPython.display import display,Markdown
'''
transport='''# PROVIDED: author evidence transport. The visible replay below consumes these bytes.
# A fresh temporary directory prevents overwriting your workspace.
MANIFEST='''+repr(manifest)+'''
PACKET_SHA256='''+repr(hashlib.sha256(packed.encode()).hexdigest())+'''
PACKET='''+repr(packed)+'''
assert hashlib.sha256(PACKET.encode()).hexdigest()==PACKET_SHA256
packet=json.loads(zlib.decompress(base64.b64decode(PACKET)))
assert set(packet)==set(MANIFEST['files'])
workspace=tempfile.TemporaryDirectory(prefix='l154-author-replay-')
REPLAY_ROOT=Path(workspace.name)
for name,digest in MANIFEST['files'].items():
    dest=(REPLAY_ROOT/name).resolve()
    assert dest.is_relative_to(REPLAY_ROOT.resolve())
    raw=base64.b64decode(packet[name]);assert hashlib.sha256(raw).hexdigest()==digest
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
print('Verified portable author packet:',len(packet),'files; fresh training NOT_RUN')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson154 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. The default lane is a standalone CPU replay of real author evidence. Complete the three functions before running the final report. It does not train models, download datasets, or use cloud services. Source/target provenance remains bounded by the embedded ledgers. Offline execution needs numpy and IPython; standard Jupyter environments provide them.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(transport,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for number,name,check in [('2','summarize_runs','check_summary'),('4','compare_entries','check_comparison'),('5','portfolio_verdict','check_verdict')]:
   if section.startswith('## '+number):
    code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.append(nb.v4.new_code_cell('# TODO: live reporting function used by the actual replay.\n'+code))
    cells.append(nb.v4.new_code_cell('# CHECK: adversarial contracts.\n'+checks['rejects']+'\n'+checks['fixture']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 5'):
   cells.append(nb.v4.new_markdown_cell('### PROVIDED · Visible replay implementation\n\nAUC uses average ranks for tied scores; MAE aligns predictions by full query keys; MAP iterates ranked candidates and uses the full relevant set. Selection histories and protocol fields are checked before aggregation. The following code reads only the temporary packet.'))
   cells.append(nb.v4.new_code_cell('\n\n'.join(replayers[n] for n in ['aligned_score','verify_inputs','replay','render_report'])))
   cells.append(nb.v4.new_code_cell('''# CHECK: these exact learner functions construct the real measured report.
report=replay(REPLAY_ROOT,MANIFEST,summarize=summarize_runs,compare=compare_entries,verdict=portfolio_verdict)
assert report['total_prediction_rows']==61148
assert report['verdict']['complete_tasks']==2
assert report['verdict']['matched_baseline_tasks']==0
assert report['verdict']['local_superiority']=='NOT_ESTABLISHED'
display(Markdown(render_report(report)))
'''))
 cells.append(nb.v4.new_code_cell('''# EXIT: your evidence replay and your reasoning have separate statuses.
WRITTEN_DEFENSE = ""  # Write 250–400 words; submit for review, never auto-award mastery.
export=dict(report,written_defense=WRITTEN_DEFENSE,learner='PENDING_WRITTEN_DEFENSE')
Path('l154-portfolio-report.json').write_text(json.dumps(export,indent=2)+'\\n')
Path('l154-portfolio-report.md').write_text(render_report(report)+'\\n'+WRITTEN_DEFENSE)
Path('l154-report.json').write_text(json.dumps(dict(status='PASS',prediction_rows=report['total_prediction_rows'],verdict=report['verdict'],additional_cloud_spend_usd=0),indent=2))
print('Exported l154-portfolio-report.json and .md; defense PENDING_WRITTEN_DEFENSE')
workspace.cleanup()
'''))
 for i,c in enumerate(cells):c.id=f'l154-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}})
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  for new,prev in zip(notebook.cells,old.cells):
   if new.cell_type==prev.cell_type=='code' and new.source==prev.source:
    new.outputs=prev.outputs;new.execution_count=prev.execution_count;new.metadata=prev.metadata
 nb.write(notebook,path)
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
executed=nb.read(P/'solutions'/(S+'.ipynb'),4)
if all(c.execution_count is not None for c in executed.cells if c.cell_type=='code'):
 exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
 html,_=exporter.from_notebook_node(executed)
 # Saved report links are relative to evidence/l154, not the HTML gallery.
 for n in (151,152,153):html=html.replace(f'href="../../l{n}-reproduction.md"',f'href="../l{n}-reproduction.md"')
 (P/'html'/(S+'.html')).write_text(html)
print('Built Lesson154, reference, two figures and standalone student/solution notebooks')
