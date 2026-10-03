"""Deterministic lesson/reference and portable notebook builder."""
from pathlib import Path
import ast,base64,hashlib,io,json,re,zipfile
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l184';S='0184-gelgt-temporal-attention'
r=json.loads((E/'report.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
status='Selected GelGT reproduction **INCOMPLETE_SOURCE_TEMPORAL_GATE**. Fresh training **NOT_RUN**. Local source/mechanism audit and all **8,712 labels PASS**. Cloud/API **$0**. Learner **PENDING_WRITTEN_DEFENSE**.'
rows=['| Split | Complete queries | Entity-only entries | Lost distinct entries |','|---|---:|---:|---:|']
for s,v in r['cache_collisions'].items():rows.append(f"| {s} | {v['queries']:,} | {v['unique_entities']:,} | {v['overwritten_query_entries']:,} |")
results='\n'.join(rows)+'\n\nAll labels reconstruct with maximum error **0**. Original attention: maximum forward error **4.44×10⁻¹⁶** and input-gradient error **2.15×10⁻¹⁵** in a float64, dropout-zero controlled test. Synthetic source probe: **2 queries → 1 cache entry**, admitting one future node to the early query.'
captions={'architecture':'Paper/release GelGT architecture. The released outer hybrid loop has one block; published depth remains unresolved. No full model fit ran.','cache':'Synthetic original-source cache counterexample: raw days5and15, cutoffs10and20. The early query incorrectly reuses the late context.','attention':'Synthetic Gaussian feature and one-head attention trace. Equal content scores and identity projection; not a trained performance result.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l184'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l184/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for key,plain,html in [('WARMUP','From memory: define complete query identity, validation-only selection and temporal availability.','<div id="warmup"></div>'),('PREDICT','Predict before reading: can two cutoff contexts survive in a dictionary keyed by driver alone?','<div id="predict"></div>'),('EXPLORER','Use the TRY cell to change center, width and projection. Keep values/content scores fixed.','<div id="gelgt-viz"></div><noscript>At cutoff 10, day 15 is excluded. With lags 0, 2, 4, center 2 and width 2, the feature is 0.368, 1, 0.368. A zero projection restores uniform attention; a negative projection reverses the preference.</noscript>'),('TEACHBACK','Defend access, selection and weighting in the EXIT response below.','<div id="teachback"></div>')]:s=s.replace('[['+key+']]',plain if portable else html)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','gelgt-viz'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','gelgt-viz','l184-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 184 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav><header><p class="route-kicker">Year5 · Quarter3 · Lesson 184</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('GelGT: choose context, then weight time',prose(),True))
ref='''**Access → selection → weighting.** Full query key=(entity,cutoff). Admit only available context; select among admitted nodes; attention weights affect contribution, not permission.

**Structure:** breadth-first traversal covers one hop before two. An induced graph retains edges between surviving nodes. A budget smaller than protected nodes requires an explicit policy.

**Semantics:** dot(seed,neighbor) ranks distant learned embeddings. Protecting local nodes is a policy choice; score similarity alone does not guarantee useful information.

**Time:** r=exp(-((absolute_lag-center)/width)²), then a learned linear projection produces each head's bias. No factor1/2. Release effective width=abs(raw)+1e-5. Negative projection weights can reverse preferences. softmax(QKᵀ/√head_width+bias)V yields the output.

**Release fusion:** sigmoid(w)×GNN+(1−sigmoid(w))×attention. Gaussian weighting never fixes future rows or missing cutoff keys.

**Observed L184:**8712labels reconstructed; original attention arithmetic checked; cache counterexample executed. Table2driver-position3.7345±.1200MAE remains cited, not reproduced. Full reproduction INCOMPLETE_SOURCE_TEMPORAL_GATE; training NOT_RUN; no new cloud spend.

[Lesson](../lessons/0184-gelgt-temporal-attention.html) · [Student lab](../labs/0184-gelgt-temporal-attention.ipynb) · [Protocol](../labs/l184-reproduction.md) · [Paper](https://arxiv.org/html/2605.15575v2)
'''
(R/'reference/gelgt-temporal-attention.html').write_text(doc('GelGT — quick reference',ref))
(E/'report.md').write_text('# L184 evidence\n\n'+status+'\n\n'+results+'\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(manifest['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(E/'packet'/name).read_bytes())
encoded=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
source=(P/'relkit/gelgt_l184.py').read_text();tree=ast.parse(source);functions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
def onefunc(file,name):
 text=(P/file).read_text();return next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
def book(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 184 · GelGT\n\nOne skill: separate legal context, retained context and temporal attention. PROVIDED cells expose the implementation; TODO cells are yours; CHECK invokes your actual functions; EXIT needs a written defense. TierB: complete real F1 database/task packet; TierC: controlled mechanism counterexamples. This is an executable source audit, not a completed full GelGT reproduction.')
 code('# @colab-bootstrap\nfrom pathlib import Path\nimport ast,base64,hashlib,io,json,math,random,typing,importlib.util,zipfile\nfrom collections import defaultdict,deque\nfrom types import SimpleNamespace\nimport numpy as np\nimport pandas as pd\nimport pyarrow,duckdb,torch\nprint("Dependencies loaded; full training NOT_RUN; cloud $0")',['colab-bootstrap'])
 md(prose(True))
 md('## PROVIDED · Authenticate the complete packet\nNo network required. Dependencies must already be installed: numpy,pandas,pyarrow,duckdb,torch. Archive hashes establish bytes, not historical usage. Embedded source is unmodified. Extraction permits only relative packet paths.')
 code('PACKET='+repr(encoded)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\nP=Path('l184-packet');P.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(P)\nmanifest="+repr(manifest),['data-payload'])
 tasks=[('temporal_bfs','Preserve access before selection','Return seed-first, sorted, undirected two-hop BFS. Budget includes seed. Admit only finite timestamps strictly below cutoff; exclude untimed nodes. Refuse invalid inputs. Reversing the edge list must not change the answer.',"assert temporal_bfs([(0,1),(1,2)],[0,5,15],0,10,3)==[0,1]"),('gaussian_features','Compute temporal features','Accept an array of lags and finite scalar center/positive effective width. Return the Gaussian radial feature array. Reject nonpositive/nonfinite width and nonfinite inputs. Your function is used inside the actual source-bias parity audit.',"assert np.allclose(gaussian_features([0,2,4],2,2),[math.exp(-1),1,math.exp(-1)])"),('keyed_mae','Keep the cutoff in evaluation','Input rows are(entity,cutoff,value). Join truth and predictions on the complete key, ignoring order. Reject duplicate/empty/missing populations or nonfinite values. Return mean absolute error.',"assert keyed_mae([(1,2,1.),(1,3,5.)],[(1,3,5.),(1,2,3.)])==1.")]
 for name,title,instructions,check in tasks:
  md('## TODO · '+title+'\n**Goal:** '+instructions+'\n\n**Why:** a silent access, arithmetic or identity change invalidates model comparisons.')
  node=ast.parse(functions[name]).body[0];code(functions[name] if solution else 'def '+name+'('+ast.unparse(node.args)+'):\n    raise NotImplementedError("Implement '+name+'")')
  md('### CHECK · Immediate feedback');code(check+"\nprint('Immediate CHECK passed')")
 md('## PROVIDED · Semantic selection and attention trace\nThese functions use the same numbers as the lesson. The semantic budget refusal is a course policy; it is not patched into the original release.')
 code(functions['semantic_refine']);code(functions['attention_trace'])
 md('## CHECK · Independent scalar arithmetic and adversarial contracts\nThree incorrect implementations must fail. One hundred random Gaussian cases are checked with scalar math; one hundred shuffled keyed populations retain the same score.')
 code(onefunc('_check_l184.py','checks'));code(onefunc('_verify_l184.py','independent184'));code("print(independent184(temporal_bfs,gaussian_features,keyed_mae))\nassert semantic_refine([[1,0],[0,1],[.1,0],[2,0]],[0,1,2,2],3)==[0,1,3]")
 md('## PROVIDED + CHECK · Execute the original source counterexample\nOnly the process Pool is replaced by a serial adapter. Original function bodies stay unchanged. The attention module is imported from authenticated bytes. Independent QK/softmax/V arithmetic checks output and input gradients at dropout0. Raw SQL reconstructs every real label. This audit calls your Gaussian implementation; synthetic scoring checks use your complete-key MAE.')
 code(onefunc('_audit_l184.py','audit184'));code("report=audit184(P,manifest,gaussian_features)\nPath('l184-report.json').write_text(json.dumps(report,indent=2))\nprint(pd.DataFrame(report['cache_collisions']).T.to_string())\nprint(report['labels'],'labels; max error',report['label_max_error'])\nprint(report['synthetic_source_probe'])\nprint(report['selected_experiment'])")
 md('## TRY · Change a cause, predict a consequence\nBefore executing, predict the output when the projection becomes zero or negative. Then compare. The three already-legal values are fixed; this does not alter the cache counterexample.')
 code("for projection in [1,0,-1]:\n    trace=attention_trace([0,2,4],projection=projection)\n    print(projection,trace)\nassert abs(attention_trace([0,2,4],projection=0)['output']-13/3)<1e-12\nassert semantic_refine([[1,0],[0,1],[2,0],[.1,0]],[0,1,2,2],3)==[0,1,2]")
 md('## CHECK · Tampered source is rejected before execution')
 code("bad=json.loads(json.dumps(manifest));bad['files']['upstream/utils.py']='0'*64\ntry:audit184(P,bad,gaussian_features)\nexcept ValueError:print('Tampered source rejected')\nelse:raise AssertionError('Tampered source accepted')")
 md('## EXIT · Write and defend\nExplain why an accurate Gaussian calculation and correct labels do not certify a valid benchmark. Propose the first necessary cache change and one independent intervention that would catch a remaining time error. Ask the teaching agent for feedback; automatic tests do not establish mastery.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','access_vs_weighting':'','complete_key_failure':'','raw_timestamp_intervention':'','what_a_repaired_run_would_establish':''}\nPath('l184-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## NEXT STEP · Full selected reproduction admission\nDefault OFF. Turning this on re-runs the authenticated audit and stops; it cannot spend cloud money or silently train a repaired model. Repo command: `.venv/bin/python labs/_run_l184.py --preset paper`. Source/config/temporal gates must be resolved under a declared new protocol before a full runner can be validated. Whole-paper reproduction NOT_RUN.')
 code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    decision=audit184(P,manifest,gaussian_features)\n    raise RuntimeError(decision['selected_experiment']+': no training dispatch')\nprint('Full training NOT_RUN; post-gate cloud operator NOT_VALIDATED')")
 md('## Appendix · Full original model and trainer, visible and pinned\nThe code below is archived for inspection, not executed as notebook training. Library calls remain visible. The course sampler is deliberately separate; no hidden repair is made. Inspect `filter_neighbors`, the fusion operation, `local_nodes_hetero`, and validation checkpoint selection together.')
 for f in ['model.py','local_module.py','encoders.py','codebook.py','utils.py','main_node_ddp.py']:
  md('### Original '+f+'\n```python\n'+(E/'packet/upstream'/f).read_text()+'\n```')
 n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(n.cells):c.id=f'l184-{i:03}'
 return n
nb.write(book(False),P/(S+'.ipynb'));nb.write(book(True),P/'solutions'/(S+'.ipynb'))
print('Built lesson, reference and portable notebooks')
