"""Deterministic lesson, reference, log and portable notebook builder."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
from relkit.literature_l188 import replay,review_log
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l188';S='0188-systematic-literature-tracking'
r=replay(E/'packet');reviews=json.loads((E/'screening-review.json').read_text());rows=review_log(r,reviews)
(E/'reviewed-log.json').write_text(json.dumps(rows,indent=2)+'\n')
supp=json.loads((P/'sources/l188/supplemental-log.json').read_text())
status='**Observed:** 2/4 query traversals complete; 31 retrieved records → 30 unique papers. Overall quarterly collection **INCOMPLETE**. Frozen-packet replay **COMPLETE**. Seven abstract-screened reading candidates; 23 deferred. Published model-result reproduction **NOT_RUN**. New cloud/API spend **$0**.'
results='| Query | Received / advertised | Coverage |\n|---|---:|---|\n'+'\n'.join(f"| {k} | {v['received']} / {v['total'] if v['total'] is not None else 'unknown'} | {v['status']} |" for k,v in r['queries'].items())+'\n\nFour failed API attempts are retained (HTTP 429 or timeout). Zero records from the two failed queries is not a zero-result finding. All 31 successful records were checked independently, yielding 30 unique identities. Seven are admitted for reading and 23 explicitly deferred; all results remain reported, not reproduced. The successful pages each fit within one 100-record request. Synthetic multi-page tests verify pagination behavior separately. [Raw receipt](../labs/evidence/l188/packet/collection.json) · [Independent audit](../labs/_verify_l188_results.json).'
papers='| Primary source | Reading decision | Next evidence needed |\n|---|---|---|\n'+'\n'.join(f"| [{x['title']}](https://arxiv.org/abs/{x['id']}) ({x['submitted']}) | {x['decision']}: {x['relevance'].replace('_',' ')} | {x['reason']} |" for x in supp)
log=['# L188 Q3-2026 paper log','',status,'','The API corpus and supplemental discovery lane are separate. One supplemental paper (2609.03880) is also in the API corpus; do not add their counts. Every decision below is abstract-level screening, not a method/code audit.','', '## Complete retrieved corpus','', '| Paper / submission date | Decision | Reason / next action |','|---|---|---|']
for row in rows:
 log.append(f"| [{row['id']}: {row['title']}](https://arxiv.org/abs/{row['id']}) / {row['published']} | {row['decision']} | {row['reason']} {row.get('next_action','Read the abstract and methods, classify relevance, then record a reason.')} |")
log+=['','## Supplemental verified primary pages','',papers,'','## Evidence boundary','','Source hashes and version URLs are in packet/manifest.json and reviewed-log.json. Updated timestamps remain in raw Atom records. Distinct IDs can share experiments; deduplication is not an independence audit. No paper results reproduced; no historical leaderboard changes established.']
(E/'paper-log.md').write_text('\n'.join(log)+'\n')
(E/'alerts.opml').write_text('''<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0"><head><title>Relational research alerts</title></head><body>
<outline text="arXiv machine learning" type="rss" xmlUrl="https://rss.arxiv.org/rss/cs.LG"/>
<outline text="arXiv artificial intelligence" type="rss" xmlUrl="https://rss.arxiv.org/rss/cs.AI"/>
<outline text="arXiv statistical learning" type="rss" xmlUrl="https://rss.arxiv.org/rss/stat.ML"/>
</body></opml>
''')
(E/'watch-config.json').write_text(json.dumps(dict(rss=['https://rss.arxiv.org/rss/cs.LG','https://rss.arxiv.org/rss/cs.AI','https://rss.arxiv.org/rss/stat.ML'],keywords=['relational foundation model','tabular foundation model','RelBench','TabArena'],boards=['https://star-project.stanford.edu/relbench/leaderboard/','https://huggingface.co/spaces/TabArena/leaderboard'],cadence='Weekly manual review; monthly deep read; quarterly reconciliation',activated=False,query_protocol='packet/config.json'),indent=2)+'\n')
# A compact measured flow; no model architecture is introduced in this lesson.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=P/'figures/l188';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l188'})
fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained');ax.axis('off');fig.patch.set_facecolor('#f6f9f8')
for x,title,big,detail,col in [(0.02,'DISCOVERY','2 / 4','queries complete\n2 failed → INCOMPLETE','#996027'),(.36,'IDENTITY','31 → 30','records → unique papers\n1 cross-query duplicate','#1d655d'),(.70,'SCREENING','7 + 23','include for reading + defer\n0 model results reproduced','#294f76')]:
 ax.text(x,.92,title,transform=ax.transAxes,fontsize=10,color=col,weight='bold');ax.text(x,.61,big,transform=ax.transAxes,fontsize=28,color=col);ax.text(x,.26,detail,transform=ax.transAxes,fontsize=11,linespacing=1.7)
ax.text(.02,.02,'Measured Q3 packet • complete replay preserves the incomplete discovery status',transform=ax.transAxes,fontsize=10,color='#40595d')
fig.savefig(F/'evidence-flow.png',dpi=160,metadata={'Software':'L188'});fig.savefig(F/'evidence-flow.svg',metadata={'Date':None});plt.close(fig)
fig,ax=plt.subplots(figsize=(4.2,6),layout='constrained');ax.axis('off');fig.patch.set_facecolor('#f6f9f8')
for y,title,big,detail,col in [(.98,'DISCOVERY','2 / 4','queries complete; 2 failed','#996027'),(.65,'IDENTITY','31 → 30','records → unique papers','#1d655d'),(.32,'SCREENING','7 + 23','include for reading + defer','#294f76')]:
 ax.text(.04,y,title,transform=ax.transAxes,fontsize=11,color=col,weight='bold',va='top');ax.text(.04,y-.08,big,transform=ax.transAxes,fontsize=27,color=col,va='top');ax.text(.04,y-.20,detail,transform=ax.transAxes,fontsize=11,va='top')
ax.text(.04,.01,'Measured packet • no model results reproduced',transform=ax.transAxes,fontsize=9,color='#40595d')
fig.savefig(F/'evidence-flow-mobile.svg',metadata={'Date':None});plt.close(fig)
figcaption='Measured L188 packet: two complete queries, 31 records, 30 unique papers, seven reading candidates and 23 deferred. No model-result reproduction.'
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',results).replace('[[PAPERS]]',papers)
 widgets={'WARMUP':('Recall validation selection, complete keys and replay versus training.','<div id="warmup"></div>'),'PREDICT':('Predict: is a first page of 100 out of 203 advertised results a complete search? Explain before continuing.','<div id="predict"></div>'),'TRIAGE':('Try a verified baseline candidate; now remove source verification. Should INCLUDE become DEFER? The notebook implements this rule.','<div id="triage"></div><noscript>Verified and reviewed baseline: INCLUDE for reading. Missing verification: DEFER. An incremental application: EXCLUDE. All three have NOT_RUN model reproduction.</noscript>'),'TEACHBACK':('Write the EXIT defense below and ask the teacher to review it.','<div id="teachback"></div>')}
 for k,(plain,widget) in widgets.items():s=s.replace('[['+k+']]',plain if portable else widget)
 src='data:image/png;base64,'+base64.b64encode((F/'evidence-flow.png').read_bytes()).decode() if portable else '../labs/figures/l188/evidence-flow.svg'
 picture='<img src="'+src+'" alt="'+figcaption+'">'
 if not portable:picture='<picture><source media="(max-width:500px)" srcset="../labs/figures/l188/evidence-flow-mobile.svg">'+picture+'</picture>'
 s=s.replace('## 6 · Run the complete approved lane','## 6 · Run the complete approved lane\n\n<figure>'+picture+'<figcaption>'+figcaption+'</figcaption></figure>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 h=render(body).replace('<table>','<div class="lit-table" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','literature-triage','l188-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/literature-triage.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav><header><p class="lit-kicker">Year 5 · Quarter 3 · Lesson 188</p><h1>'+title+'</h1></header>'+h+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Systematic literature tracking',prose(),True))
ref='''A reproducible log records queries, date window, raw responses, retrieval timestamps, exact versions, hashes, screening reasons and unknowns.

| Contract | Question |
|---|---|
| Search scope | Which expressions, fields, source and first-submission window? |
| Pagination | Are offsets contiguous, totals stable and identities unique? |
| Identity | Which arXiv base ID groups the retained versions? |
| Dates | First submission, revision and retrieval are separate clocks. |
| Admission | Verified source + reason + SOTA/failure/baseline relevance? |
| Evidence | Reported, replayed or reproduced under a named protocol? |
| Board comparison | Same task suite, version, metric, tuning and adaptation? |
| Stop | Missing page or source → INCOMPLETE, not zero papers. |

**Weekly:** check last successful fetch, screen new entries, retain rejects and unknowns. **Monthly:** audit one candidate's protocol/artifacts. **Quarterly:** reconcile candidates with the research agenda. Track revisions separately; the frozen Q3 collector filters first submissions.

**Synthetic example:** offsets 0/100/200, lengths 100/100/3, stable total203 and203 distinct IDs pass. Omit offset100 and fail. Valid hashes cannot replace missing records.

**Reading rule:** INCLUDE is permission to spend attention, not proof of a scientific claim. Separate IDs need not be independent experiments. A benchmark shell is not an extracted score table.

'''+status+'''\n\n[Lesson](../lessons/0188-systematic-literature-tracking.html) · [Paper log](../labs/evidence/l188/paper-log.md) · [Alert configuration](../labs/evidence/l188/watch-config.json) · [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html) · [RSS guide](https://info.arxiv.org/help/rss.html).
'''
(R/'reference/systematic-literature-tracking.html').write_text(doc('Literature tracking · field guide',ref))
# Deterministic, authenticated offline payload. No fetch in the learner replay.
buf=io.BytesIO();files={}
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for path in sorted((E/'packet').rglob('*'))+[E/'screening-review.json']:
  if not path.is_file():continue
  name=str(path.relative_to(E));raw=path.read_bytes();files[name]=hashlib.sha256(raw).hexdigest();info=zipfile.ZipInfo(name,date_time=(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
payload=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
source=(P/'relkit/literature_l188.py').read_text();tree=ast.parse(source);functions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
header=source[:source.index('def canonical_id')]
expected=json.dumps(r,sort_keys=True)

def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 188 · Systematic literature tracking\n\n'+prose(True))
 md('## PROVIDED · Portable offline bootstrap\nThis tool/API lab uses real arXiv metadata; CHECK fixtures are explicitly synthetic. No training dataset or model is introduced. The embedded packet contains every collected response, including failures. All replay code uses the Python standard library; no cloud account, repository checkout or network is needed.')
 code('# @colab-bootstrap\nimport base64, hashlib, io, json, tempfile, zipfile\nfrom pathlib import Path\nworkspace=Path(tempfile.mkdtemp(prefix="l188-replay-"))')
 code("payload=base64.b64decode('"+payload+"')\nassert hashlib.sha256(payload).hexdigest()=="+repr(digest)+"\nwith zipfile.ZipFile(io.BytesIO(payload)) as archive:\n    for name in archive.namelist():\n        assert (workspace/name).resolve().is_relative_to(workspace.resolve())\n    archive.extractall(workspace)\nexpected_files="+repr(files)+"\nfor name,digest in expected_files.items():\n    assert hashlib.sha256((workspace/name).read_bytes()).hexdigest()==digest\npacket=workspace/'packet'",['data-payload'])
 code(header)
 tasks=[('canonical_id','value','Validate official arXiv URLs and old/new IDs; group versioned copies without erasing identity digits.',"assert canonical_id('https://arxiv.org/abs/2607.12345v12')=='2607.12345'\nassert canonical_id('hep-th/9901001v2')=='hep-th/9901001'"),('coverage','pages','Reject missing/overlapping pages, changing totals and repeated identities. A genuine zero-result feed must be present.',"assert coverage([{'start':0,'total':3,'ids':['2607.00001','2607.00002']},{'start':2,'total':3,'ids':['2607.00003']}])=='COMPLETE'\nassert coverage([{'start':0,'total':3,'ids':['2607.00001','2607.00002']}])=='INCOMPLETE'\nassert coverage([])=='INCOMPLETE'"),('triage','record','Use verified identity, completed screening, quarter membership, written reason and relevance. Admission does not promote evidence.',"candidate=dict(verified=True,reviewed=True,in_window=True,reason='Required comparator',relevance='baseline',evidence='reported')\nassert triage(candidate)=='INCLUDE'\nassert triage(dict(candidate,verified=False))=='DEFER'\nassert triage(dict(candidate,relevance='incremental'))=='EXCLUDE'")]
 for name,arg,task,check in tasks:
  md('## TODO · '+name+'\n'+task+' Predict a counterexample before writing code.')
  code(functions[name] if solution else f'def {name}({arg}):\n    # TODO: implement the stated contract.\n    raise NotImplementedError("{name}")')
  md('### CHECK · Synthetic examples only');code(check+"\nprint('Contract examples passed')")
 md('## PROVIDED · Parse the source bytes\nRead the namespace-qualified fields and retain first-submission/revision timestamps. A malformed HTTP200 response must not count as success.')
 code(functions['parse_atom'])
 md('## PROVIDED · Replay and review\nThe pipeline calls your three functions. Source screening is a separate, versioned annotation; raw discovery bytes are unchanged.')
 code(functions['replay']+'\n\n'+functions['review_log'])
 md('## CHECK · Complete real packet\nPrint the status before interpreting it. All retrieved records are processed; failed queries remain incomplete.')
 code("report=replay(packet,canonical_id,coverage,triage)\nreviews=json.loads((workspace/'screening-review.json').read_text())\nlog=review_log(report,reviews,triage)\nassert report==json.loads("+repr(expected)+")\nassert sum(x['decision']=='INCLUDE' for x in log)==7\nassert len(log)==30\nPath('l188-report.json').write_text(json.dumps(report,indent=2))\nPath('l188-reviewed-log.json').write_text(json.dumps(log,indent=2))\nprint('Collection:',report['collection_status'],'| Replay:',report['replay_status'])\nfor name,q in report['queries'].items():\n    print(name,q['received'],q['total'],q['status'])\nprint('Unique papers:',len(log),'| Reading candidates:',sum(x['decision']=='INCLUDE' for x in log))")
 md('## EXIT · Your evidence-backed decision\nWrite 150–250 words: scope; source; relevance; missing protocol; next experiment. Explain why replay does not repair missing discovery, and why INCLUDE does not mean reproduced. Leave personal mastery pending until the teacher reviews your attempt.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','search_boundary':'','candidate_and_source':'','inclusion_reason':'','missing_protocol':'','next_experiment':'','what_would_change_my_mind':''}\nPath('l188-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## NEXT STEP · Fresh collection is separate\nRun the repository command in section6 into a new directory. The full collector source is below for inspection. It is not executed in this offline notebook. A fresh result may differ; keep its timestamp, failures and all pages. No recurring task is activated. Paper-result reproduction stays NOT_RUN. Ask the teacher for feedback; revisit after one day and one week.\n\n```python\n'+(P/'_collect_l188.py').read_text()+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l188-{i:03}'
 return book
for solution in (False,True):
 path=P/('solutions' if solution else '')/(S+'.ipynb');book=notebook(solution)
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in book.cells]:
   book=old
 nb.write(book,path)
print('Built L188 lesson, reference, complete log, alert config and notebooks')
