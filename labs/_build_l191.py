"""Deterministic authoring from authenticated published-table evidence."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l191';Q=E/'packet';F=P/'figures/l191';S='0191-kumorfm2-sota-tracking'
report=json.loads((E/'report.json').read_text());tables=json.loads((Q/'tables.json').read_text());policy=json.loads((Q/'method-policy.json').read_text())
src=(P/'relkit/tracking_l191.py').read_text();functions={n.name:ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
rsrc=(P/'_replay_l191.py').read_text();replay_functions=[ast.get_source_segment(rsrc,n) for n in ast.parse(rsrc).body if isinstance(n,ast.FunctionDef)]
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join('---' for _ in headers)+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
summary=[]
for t in report['tables']:
 for pool in ['foundation','supervised']:
  p=t['pools'][pool];summary.append([t['number'],pool,', '.join(p.get('single_methods',[])) or 'No eligible row',f"{p['single_gap']:.6f}" if 'single_gap' in p else 'Missing', 'AUROC points' if t['metric']=='AUROC' else 'normalized MAE'])
summary=mdtable(['Table','Bounded open pool','Best single method','Kumo advantage','Unit'],summary)
differences=[]
for t in report['tables']:
 for row in t['audit']:
  if row['aggregate_status']=='OUTSIDE_ROUNDING_BOUND':differences.append([t['number'],row['method'],f"{row['published_aggregate']:.6f}",f"{row['recomputed_aggregate']:.6f}",f"{row['rounding_interval'][0]:.6f}–{row['rounding_interval'][1]:.6f}"])
discrepancies=mdtable(['Table','Method','Published','From cells','Cell-rounding bounds'],differences)
full=[]
for t,s in zip(tables,report['tables']):
 data=[]
 for row,audit in zip(t['rows'],s['audit']):data.append([row['method']+(' *' if row['author_evaluated'] else ''),*row['displayed'],row['published_aggregate'],f"{audit['recomputed_aggregate']:.6f}",row['published_rank'],f"{audit['displayed_mean_rank']:.6f}"])
 text=mdtable(['Method',*t['tasks'],'Published avg','Recomputed avg','Published rank','Displayed rank'],data)
 full.append((t['number'],t['suite']+' · '+t['metric'],text))
status='<div class="sota-status"><strong>Complete selected-table reconstruction.</strong> 401 task scores · 45 method rows · $0 cloud/API. Fresh inference: NOT_RUN. Historical model reproduction and current global SOTA: NOT_ESTABLISHED. Learner defense: pending.</div>'
figures={'architecture':'Conceptual KumoRFM-2 forward path from §3. Amber tracks the query; known context labels enter the table stage. Symbolic dimensions and illustrative rows, not recovered weights or an executed model.','gaps':'Computed from all 12 Table 3 task columns. The single-method gap is 4.053 AUROC points for the bounded foundation pool and 1.542 for the supervised pool. Oracle selection is descriptive and uses test outcomes.'}
prose=(R/'lessons/content'/(S+'.md')).read_text()
def fill(text,portable=False):
 text=text.replace('[[STATUS]]',status).replace('[[SUMMARY]]',summary).replace('[[DISCREPANCIES]]',discrepancies)
 text=text.replace('[[TABLES]]','\n\n'.join(('### ' if portable else '<details><summary>')+f'Table {n} · {title}'+('\n\n' if portable else '</summary>\n\n')+table+('\n' if portable else '\n</details>') for n,title,table in full)+'\n\n* marks a baseline evaluated by the Kumo authors. Other baseline values were taken from cited work. Source table ranks use an unestablished pool/precision convention; recomputed ranks use every displayed row.')
 for name,caption in figures.items():
  uri='data:image/png;base64,'+base64.b64encode((F/(name+'-mobile.png')).read_bytes()).decode() if portable else '../labs/figures/l191/'+name+'.svg'
  picture = f'<picture><source media="(max-width: 500px)" srcset="../labs/figures/l191/{name}-mobile.svg"><img src="{uri}" alt="{caption}"></picture>' if not portable else f'<img src="{uri}" alt="{caption}">'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="sota-figure'+(' sota-architecture' if name=='architecture' else '')+f'">{picture}<figcaption>{caption}</figcaption></figure>')
 text=text.replace('[[CODE]]','The TODO cells below implement direction, eligibility and full-coverage comparison.' if portable else '<details><summary>Visible Python · comparison direction and eligibility</summary>\n\n```python\n'+functions['signed_gap']+'\n\n'+functions['eligible']+'\n```\n</details>')
 if portable:
  for id in ['warmup','gap-predict','sota-explorer','gap-teachback']:text=text.replace('<div id="'+id+'"></div>','')
  text=re.sub(r'<noscript>(.*?)</noscript>',r'\1',text,flags=re.S)
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 else:
  text=re.sub(r'\$\$.*?\$\$',lambda m:'<p class="sota-equation">'+('AUROC gap = Kumo − comparator; MAE gap = comparator − Kumo.' if 'gap' in m.group() else 'Normalized MAE = mean over tasks of (method MAE ÷ LightGBM MAE).')+'</p>',text,flags=re.S)
 return text

def doc(title,body,interactive=False):
 scripts=''
 if interactive:
  scripts='<script id="sota-data" type="application/json">'+json.dumps(dict(tables=tables,policy=policy))+'</script>'+''.join('<script src="../assets/'+name+'.js"></script>' for name in ['retrieval-pool','retrieval-bank','predict','teachback','sota-tracking'])
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/sota-tracking.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a> · <a href="../lessons/'+S+'.html">Lesson 191</a></nav>'+body+'</article>'+scripts+'</body></html>'
def wrap(body):return body.replace('<table>','<div class="sota-table" tabindex="0"><table>').replace('</table>','</table></div>')
(R/'lessons'/(S+'.html')).write_text(doc('Lesson 191 · KumoRFM-2 SOTA tracking',wrap(render(fill(prose))),True))
ref='''# SOTA comparison · field guide

**Freeze:** source version, task IDs, score units, model pool, coverage and averaging rule.

**Direction:** AUROC advantage = Kumo − comparator (percentage points here). MAE advantage = comparator − Kumo (task-specific units).

**Regression across tasks:** average the per-task ratios MAE(method) / MAE(LightGBM). Subtract normalized averages to compare methods. Never average unrelated raw MAEs.

**Single method:** choose one complete-coverage row by aggregate. On test scores this is a retrospective summary, not a deployable selection policy.

**Taskwise oracle:** select the strongest eligible score separately for each task. This optimistic envelope is not a single fitted method.

**Open code:** inspected implementation. It does not certify checkpoint access, all backend licenses or a reproducible historical training run.

**Rounding:** a two-decimal score has half-unit bound0.005. Propagate bounds through ratios. These are not confidence intervals. Rank ties get average occupied rank; hidden precision can change them.

**As of2 October 2026:** the frozen April tables omit OpenRFM and later RDBLearn revisions. Current global SOTA remains NOT_ESTABLISHED.

## Reconstructed comparisons

'''+summary+'''
## Before claiming reproduction

Published-table arithmetic → COMPLETE_SELECTED_TABLES. Fresh predictions → NOT_RUN. Historical checkpoint/protocol identity → NOT_ESTABLISHED. Learner defense → PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0191-kumorfm2-sota-tracking.html) · [Student lab](../labs/0191-kumorfm2-sota-tracking.ipynb) · [Protocol](../labs/l191-reproduction.md) · [Full report](../labs/evidence/l191/report.json) · [Primary paper](https://arxiv.org/html/2604.12596v1).
'''
(R/'reference/kumorfm2-sota-tracking.html').write_text(doc('SOTA comparison field guide',wrap(render(ref))))
# Reproducible portable archive (no filesystem or network dependency after extraction).
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(Q.iterdir())+[E/'input-manifest.json']:
  info=zipfile.ZipInfo(str(p.relative_to(E)),date_time=(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
payload=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
tasks=[('signed_gap','target, comparator, metric','Return a positive value when Kumo is better. Accept only AUROC and MAE; reject nonfinite numbers. AUROC inputs are in published percentage units.',"assert abs(signed_gap(84.59,70.87,'AUROC')-13.72)<1e-10\nassert abs(signed_gap(7.298,7.374,'MAE')-.076)<1e-10\ntry: signed_gap(float('nan'),1,'MAE')\nexcept ValueError: pass\nelse: raise AssertionError('Reject nonfinite values')"),('eligible','metadata, pool','Require access=open_code and protocol=reported_same_table. Include only foundation/supervised families; pool is foundation, supervised or all. Missing evidence returns False. Unknown pool raises ValueError.',"m=dict(access='open_code',protocol='reported_same_table',family='foundation')\nassert eligible(m,'foundation')\nassert not eligible(dict(m,access='unverified'),'all')\nassert not eligible(m,'supervised')\nassert not eligible({},'all')"),('compare_pool','target, candidates, metric, baseline=None','Implement a complete-coverage comparison. Reject wrong lengths and nonfinite values; exclude rows with None. Return NO_ELIGIBLE_COMPARATOR when none remain. Otherwise normalize MAE by positive per-task baseline, compute each mean, retain tied best single methods, form taskwise oracle and its tied methods, and return direction-correct gaps. Use the exact output contract below. All ties use absolute tolerance 1e-12 for aggregate scores; task ties use equality.',"x=compare_pool([8,6],{'a':[9,1],'b':[4,4],'partial':[10,None]},'AUROC')\nassert x['single_methods']==['a'] and x['excluded']==['partial']\nassert x['single_gap']==2 and x['oracle_gap']==.5\ny=compare_pool([2,4],{'a':[4,4],'b':[2,8]},'MAE',[2,4])\nassert y['single_methods']==['a','b'] and y['oracle_gap']==0\nassert compare_pool([1],{},'AUROC')['status']=='NO_ELIGIBLE_COMPARATOR'")]
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 191 · Published-table reconstruction\n\n'+fill(prose,True))
 md('## PROVIDED · Offline bootstrap\nPython 3.10+ standard library only. No installs, API calls, GPU or model inference. The embedded packet contains original HTML, access receipts, the complete extracted tables and policy. Source archive integrity is checked before extraction. Online course links are optional reading, not execution dependencies.')
 code('import base64, hashlib, io, json, math, re, tempfile, zipfile\nfrom html.parser import HTMLParser\nfrom pathlib import Path\nworkspace=Path(tempfile.mkdtemp(prefix="l191-replay-"))', ['colab-bootstrap'])
 code('payload='+repr(payload)+'\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():\n        if not (workspace/name).resolve().is_relative_to(workspace.resolve()):\n            raise ValueError("Unsafe archive path")\n    z.extractall(workspace)\npacket=workspace/"packet"\nmanifest=json.loads((workspace/"input-manifest.json").read_text())', ['data-payload'])
 md('## PROVIDED · Midranks\nEqual displayed values share the mean of their occupied ranks. This helper does not recover unpublished precision.')
 code(functions['average_ranks'])
 for name,args,explanation,check in tasks:
  md('## TODO · '+name+'\n'+explanation+'\n\nBefore coding, predict one input that must be rejected or excluded.')
  if name=='compare_pool':md('Required successful fields: status=`COMPLETE_PUBLISHED_COMPARISON`; single_methods (sorted list), single_score, target_score, single_gap; oracle_values, oracle_methods (sorted list per task), oracle_score, oracle_gap; per_task_gaps (raw task units), excluded (sorted list). No-eligible output has only status and excluded. The single score is the strongest mean under the metric direction. Use your signed_gap function for every gap.')
  code(functions[name] if solution else 'def '+name+'('+args+'):\n    # TODO: implement the stated contract.\n    raise NotImplementedError('+repr(name)+')')
  md('### CHECK · Synthetic edge cases');code(check+'\nprint("Contract examples PASS")')
 md('## PROVIDED · Parse the original HTML\nThis parser visits only the four frozen table IDs, checks exact row counts and keeps printed decimals. It fails if the source shape changes. Database group headers are mapped explicitly to the paper task abbreviations.')
 code(replay_functions[0])
 md('## PROVIDED · Reconstruct every table\nRead the complete loop. It calls your eligibility and comparison functions, computes ranks over every displayed row and propagates rounding intervals. The intervals describe printed precision, not seed uncertainty.')
 code(replay_functions[1])
 md('## PROVIDED · Authenticate before arithmetic');code(replay_functions[2])
 md('## CHECK · Complete real-source reconstruction\nAll four tables, all 45 method rows and 401 task cells must reproduce the frozen author report. A passed check certifies reconstruction only.')
 code('report=replay191(packet,manifest)\nexpected=json.loads('+repr(json.dumps(report,sort_keys=True))+')\nassert report==expected\nPath("l191-report.json").write_text(json.dumps(report,indent=2))\nprint(report["reconstruction"],report["task_cells"],"task scores")\nfor table in report["tables"]:\n    for pool,row in table["pools"].items():\n        print("Table",table["number"],pool,row.get("single_methods",[]),row.get("single_gap",row["status"]))')
 md('## CHECK · Your functions are load-bearing\nRemove RDBLearn access from a copy of the policy and rerun. The foundation comparison must change. This is a policy sensitivity exercise, not a new model experiment.')
 code('tables=parse_tables((packet/"paper-v1.html").read_text())\npolicy=json.loads((packet/"method-policy.json").read_text())\npolicy["RDBLearn"]["access"]="unverified"\nchanged=reconstruct(tables,policy)\nassert changed["tables"][0]["pools"]["foundation"]["single_methods"]==["Griffin"]\nprint("Changed eligible baseline:",changed["tables"][0]["pools"]["foundation"]["single_methods"])')
 md('## EXIT · Defend one comparison\nWrite the six-part contract: source/date, task coverage, metric units, pool, selection rule and evidence lane. Explain one source discrepancy and one missing comparator. Connect this baseline choice to your L190 research question. Submit your own argument to the teacher.\n\n## NEXT · Full model reproduction\nFresh inference stays NOT_RUN. Before a model run, obtain pinned weights or immutable service identity, query/data versions, complete protocol, historical context/seed details, and a bounded price. A new service prediction does not by itself recover the historical paper experiment. No paid/model command is hidden in this notebook.')
 code('submission=dict(status="PENDING_WRITTEN_DEFENSE",source_and_date="",task_coverage="",metric_units="",pool="",selection_rule="",evidence_lane="",source_discrepancy="",missing_comparator="",research_question="",fresh_run_requirements="")\nPath("l191-submission.json").write_text(json.dumps(submission,indent=2))')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l191-{i:03}'
 return book
for solution in [False,True]:
 path=P/('solutions' if solution else '')/(S+'.ipynb');book=notebook(solution)
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in book.cells]:book=old
 nb.write(book,path)
print('Built lesson, reference, student and solution notebooks')
