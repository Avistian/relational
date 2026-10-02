"""Build a connected lesson, five-page dossier and portable inline-code notebooks."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l190';F=P/'figures/l190';S='0190-research-gap-checkpoint'
r=json.loads((E/'report.json').read_text());cases=json.loads((E/'packet/cases.json').read_text())
source=(P/'relkit/checkpoint_l190.py').read_text();functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
replay_source=(P/'_replay_l190.py').read_text();replay_code=ast.get_source_segment(replay_source,next(n for n in ast.parse(replay_source).body if isinstance(n,ast.FunctionDef)))
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l190','font.size':10})
fig,ax=plt.subplots(figsize=(8.5,3.5),layout='constrained');fig.patch.set_facecolor('#f5f8f6');ax.set_facecolor('#f5f8f6')
y=r['paired_rdbpfn_minus_tabicl']['per_seed'];ax.bar(range(10),y,color=['#24796b' if v>0 else '#b77738' for v in y],width=.65);ax.axhline(0,color='#31423e',linewidth=.8);ax.axhline(r['paired_rdbpfn_minus_tabicl']['mean'],color='#244d72',linestyle='--',label='Mean +0.004369')
ax.set_xticks(range(10));ax.set_xlabel('Paired support draw');ax.set_ylabel('AUROC difference');ax.set_title('RDB-PFN minus TabICL · every saved support draw',loc='left',weight='bold',pad=14);ax.legend(frameon=False,loc='lower left',fontsize=9);ax.spines[['top','right']].set_visible(False)
for ext in ['png','svg']:fig.savefig(F/f'paired.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'L190'})
plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,5.0),layout='constrained');ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off');fig.patch.set_facecolor('#f5f8f6')
for x,y0,title,big,detail,col in [(0.25,5.6,'SAVED MODEL EVIDENCE','30 runs → 21,060 scores','Authenticate → align full keys → re-score','#22665d'),(.25,3.65,'LITERATURE EVIDENCE','2 / 4 searches complete','31 records → 30 unique papers; gaps retained','#9a612b'),(.25,1.7,'RESEARCH DECISION','Three candidate questions','Priority is a judgment; novelty is unestablished','#355477')]:
 ax.text(x,y0,title,fontsize=10,color=col,weight='bold');ax.text(x,y0-.5,big,fontsize=20,color=col);ax.text(x,y0-.94,detail,fontsize=11,color='#3c4c46')
ax.text(9.7,.12,'Replay completion ≠ research readiness',ha='right',fontsize=10,color='#3c4c46')
for ext in ['png','svg']:fig.savefig(F/f'flow.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'L190'})
plt.close(fig)
status='<div class="checkpoint-summary"><strong>Selected replay COMPLETE.</strong> Fresh training NOT_RUN; novelty NOT_ESTABLISHED. The research checkpoint remains INCOMPLETE and the learner PENDING_WRITTEN_DEFENSE.</div>'
results='| Saved model | Mean AUROC | Sample SD across support draws |\n|---|---:|---:|\n'+'\n'.join(f"| {a} | {v['mean']:.6f} | {v['sample_sd']:.6f} |" for a,v in r['models'].items())
ranking='| Candidate | Impact | Data / implementation / compute | Priority |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {c['title']} | {c['impact']} | {' / '.join(map(str,c['feasibility']))} | {r['ranking'][0]['scores'][c['id']]:.3f} |" for c in cases)
case_text='\n\n'.join('**'+c['title']+'.** '+c['hypothesis']+' '+c['reason'] for c in cases)
captions={'flow':'Measured replay boundaries: a complete selected model replay, incomplete literature coverage and an authored research decision remain separate.', 'paired':'Measured AUROC differences on the same frozen test population. All ten paired support draws shown; six are positive. No database-level confidence claim.'}
def fill(s,portable=False):
 s=s.replace('[[STATUS]]',status).replace('[[RESULTS]]',results).replace('[[RANKING]]',ranking).replace('[[CASES]]',case_text).replace('[[GATE_CODE]]','```python\n'+functions['claim_gate']+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l190/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="checkpoint-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 if portable:
  s=s.replace('<div id="checkpoint-quiz"></div>','**Predict first:** choose the supported saved-comparison claim; explain why it does not establish novelty.').replace('<div id="claim-explorer"></div>','**Try in Python:** remove one gate flag, then request a novelty claim.').replace('<div id="ranking-explorer"></div>','**Try in Python:** call rank_cases(cases), inspect all weights and alter an authored score.')
  import re
  s=re.sub(r'<noscript>(.*?)</noscript>',r'\1',s,flags=re.S)
  s=re.sub(r'<textarea.*?</textarea>','Write your defense in the EXIT cell below.',s,flags=re.S)
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False,dossier=False):
 scripts='<script id="checkpoint-data" type="application/json">'+json.dumps(dict(cases=cases))+'</script><script src="../assets/quiz.js"></script><script src="../assets/research-checkpoint.js"></script>' if interactive else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/research-checkpoint.css"></head><body'+(' class="dossier"' if dossier else '')+'><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a> · <a href="../lessons/'+S+'.html">Lesson 190</a></nav>'+body+'</article>'+scripts+'</body></html>'
prose=(R/'lessons/content'/(S+'.md')).read_text()
body=render(fill(prose)).replace('<table>','<div class="cp-table" tabindex="0"><table>').replace('</table>','</table></div>')
(R/'lessons'/(S+'.html')).write_text(doc('Research-gap checkpoint', '<h1>From evidence to a research question</h1>'+body,True))
parts=(R/'lessons/content/0190-research-gap-document.md').read_text().split('<!-- PAGE -->')
assert len(parts)==5
sheets=''.join('<section class="sheet"><div class="folio">RELATIONAL LEARNING · RESEARCH NOTE · '+str(i+1)+' / 5</div>'+render(fill(s))+'</section>' for i,s in enumerate(parts))
(R/'reference/research-gap-document.html').write_text(doc('Research-gap document · five-page worked example',sheets,dossier=True))
ref='''# Research-gap checkpoint · field guide

A defensible question follows **observation → narrow claim → alternative explanation → matched experiment → decision**.

| Question | Required evidence |
|---|---|
| Did the saved metric replay? | Exact bytes, complete keys/grid, independent score check |
| Is the claim comparable? | Same task, support, preprocessing and evaluation contract |
| Is the idea novel? | Narrow contribution and a disconfirming related-work review |
| Can the experiment run? | Data availability, healthy implementation, complete cost bound |
| Did the learner defend it? | Their own written argument and teacher review |

**Paired contrast:** compute each same-seed model difference before averaging. Positive in six support draws is not six independent datasets.

**Ranking:** impact × weighted feasibility; preserve ties and reveal score judgments. Stable weights do not validate scores.

**Five-page structure:** question; evidence; alternatives; falsifiable experiment; limitations and decision. Include sources at the claim, not only a bibliography.

**Stop rules:** missing query/run → BLOCKED; missing discovery → INCOMPLETE; untested novelty → NOT_ESTABLISHED; absent fresh run → NOT_RUN; unreviewed learner argument → PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0190-research-gap-checkpoint.html) · [Worked document](research-gap-document.html) · [Five-page PDF](research-gap-document.pdf) · [Replay protocol](../labs/l190-reproduction.md) · [Versioned RDB-PFN paper](https://arxiv.org/html/2603.03805v5).
'''
(R/'reference/research-gap-checkpoint.html').write_text(doc('Research-gap field guide',render(ref)))
# Frozen portable payload retains originals. Zip paths and individual files are authenticated.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((E/'packet').rglob('*'))+[E/'input-manifest.json']:
  if p.is_file():
   info=zipfile.ZipInfo(str(p.relative_to(E)),date_time=(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
payload=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
expected=json.dumps(r,sort_keys=True)
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 190 · Research-gap checkpoint\n\n'+fill(prose,True))
 md('## PROVIDED · Offline packet\nPython 3 + NumPy. Colab normally includes NumPy; another clean Python environment can install it with `python3 -m pip install numpy`. No model downloads, repository checkout, cloud account or network access is needed for this replay. The payload contains real original saved arrays and raw literature responses, not generated replacement data.')
 code('import base64, hashlib, io, itertools, json, math, tempfile, zipfile\nfrom fractions import Fraction\nfrom pathlib import Path\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="l190-replay-"))')
 code('payload='+repr(payload)+'\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as archive:\n    for name in archive.namelist():\n        if not (workspace/name).resolve().is_relative_to(workspace.resolve()):\n            raise ValueError("Unsafe archive path")\n    archive.extractall(workspace)\npacket=workspace/"packet"\nmanifest=json.loads((workspace/"input-manifest.json").read_text())\ncases=json.loads((packet/"cases.json").read_text())', ['data-payload'])
 tasks=[('keyed_auc','truth, predictions','Align complete (entity, cutoff) identities; reject duplicates/nonfinite values; give half credit for ties.',"truth=[(1,10,0),(1,20,1),(2,10,1),(2,20,0)]\npred=[(2,20,.5),(1,20,.9),(1,10,.1),(2,10,.5)]\nassert keyed_auc(truth,pred)==.875\ntry:\n    keyed_auc(truth,pred[:-1])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Incomplete predictions accepted')"),('claim_gate','kind, evidence','Implement the conservative four-check saved-metric gate. The other claim kinds stay unestablished or unrun.',"e=dict(authenticated=True,complete=True,metric_checked=True,comparable=True)\nassert claim_gate('saved_metric',e)=='SUPPORTED_REPLAY'\nassert claim_gate('saved_metric',dict(e,complete=False))=='BLOCKED'\nassert claim_gate('novelty',e)=='NOT_ESTABLISHED'"),('rank_cases','cases','Return all 27 weight triples, per-case priorities and all tied leaders. Validate the ordinal scores.',"ranking=rank_cases(cases)\nassert len(ranking)==27\nassert all(row['leaders']==['availability'] for row in ranking)\ntie=[cases[0],dict(cases[0],id='copy')]\nassert rank_cases(tie)[0]['leaders']==['availability','copy']")]
 for name,args,task,check in tasks:
  md('## TODO · '+name+'\n'+task+' Predict a failure case first.')
  code(functions[name] if solution else 'def '+name+'('+args+'):\n    # TODO: implement the stated contract.\n    raise NotImplementedError('+repr(name)+')')
  md('### CHECK · Synthetic examples');code(check+'\nprint("Contract examples PASS")')
 md('## PROVIDED · Full replay operator\nRead the identity and support checks before running. This calls your functions and refuses missing runs. Source label reconstruction and fresh model inference are outside its scope.')
 code(replay_code)
 md('## CHECK · All real saved evidence\nAuthenticate every input and process every run/page; compare the whole report.')
 code('report=replay190(packet,manifest,keyed_auc,claim_gate,rank_cases)\nassert report==json.loads('+repr(expected)+')\nPath("l190-report.json").write_text(json.dumps(report,indent=2))\nprint("Replay:",report["replay"],"| Predictions:",report["prediction_rows"])\nprint("Paired difference:",report["paired_rdbpfn_minus_tabicl"]["mean"])\nprint("Literature:",report["literature"]["collection_status"])\nprint("Checkpoint:",report["checkpoint"],"| Learner:",report["learner"])')
 md('## EXIT · Your five-page argument\nUse the supplied document as a worked example, not as your personal submission. Provide question, quantitative evidence, alternative explanations, a matched experiment/falsifier and limits. A replay pass cannot certify novelty or research judgment. Ask the teacher to review the argument.')
 code('submission=dict(learner="PENDING_WRITTEN_DEFENSE",question="",evidence="",alternatives="",matched_experiment="",falsifier="",aggregate_budget="",limitations="",what_changes_my_mind="")\nPath("l190-submission.json").write_text(json.dumps(submission,indent=2))')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l190-{i:03}'
 return book
for solution in [False,True]:
 path=P/('solutions' if solution else '')/(S+'.ipynb');book=notebook(solution)
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in book.cells]:book=old
 nb.write(book,path)
print('Built lesson, field guide, five-page HTML dossier, two figures and portable notebooks')
