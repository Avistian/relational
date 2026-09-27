"""Deterministic lesson/reference and portable notebook builder."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0125-pytorch-frame-deep-dive';TITLE='PyTorch Frame: from typed columns to graph node features'
canonical=(P/'relkit/frame_l125.py').read_text();tree=ast.parse(canonical)
checks=(P/'_check_l125.py').read_text();check_nodes={n.name:ast.get_source_segment(checks,n) for n in ast.parse(checks).body if isinstance(n,ast.FunctionDef)}
tasks={'numeric_tokens':('check_numeric','Reconstruct normalization, mean imputation and the column-specific affine map.'),'categorical_indices':('check_categories','Reserve padding index zero and separate column vocabularies with offsets.'),'align_rows':('check_alignment','Gather using immutable entity IDs, preserving duplicate requests and rejecting duplicate sources.'),'fit_and_convert':('check_fit','Fit on the training population, then reuse the converter for query rows.')}
summary=json.loads((P/'evidence/l125/summary.json').read_text());audit=json.loads((P/'_audit_l125_results.json').read_text())
results='**Author-reference execution: complete course feature path, not a paper-score reproduction.**\n\n| Table | Fitting rows | Encoded rows | Columns | Output width |\n|---|---:|---:|---:|---:|\n'
for name,item in summary['tables'].items():results+=f"| {name} | {item['fit_rows']:,} | {item['rows']:,} | {len(item['columns'])} | 8 |\n"
results+=f"\nTotal **{summary['total_rows']:,} rows**. Maximum difference from the upstream Frame forward pass with identical weights: **{audit['maximum_output_error']}**. All IDs and finite-value checks passed. The separate one-step exercise used **23 training queries**, **18,389 eligible result rows**, and updated **58 parameter tensors**. [Measured report](../labs/evidence/l125/summary.json) · [Independent audit](../labs/_audit_l125_results.json).\n"
captions={'typed':'Six semantic-type routes expose raw cells, materialized representations and common-width tokens. Text counts are a fixed illustrative adapter.', 'architecture':'The executed course architecture: separate driver/result ResNets, one mean-GraphSAGE relation, 23 seed predictions and a differentiable loss.', 'identity':'Entity IDs preserve alignment under reordered and repeated queries; duplicate requests accumulate gradients to their source row.'}
fallback='Static trace: with fitting values 10,20,30, mean=20 and population scale≈8.165. Value30 produces token≈[2.949,−0.725]. Missing maps to bias[0.5,0.5]. Including future value1000 changes the fitted mean to265.'
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,cap in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l125/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l125/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 text=text.replace('[[WARMUP]]','Recall: what does an entity ID identify, and why must a query carry a cutoff?' if portable else '<div id="warmup"></div><noscript><p>Recall: what does an entity ID identify, and why must a query carry a cutoff?</p></noscript>')
 text=text.replace('[[FRAME_WIDGET]]',fallback if portable else '<div id="l125-frame"></div><noscript><p>'+fallback+'</p></noscript>')
 text=text.replace('[[TEACHBACK]]','Explain why tensor shapes cannot establish correct preprocessing or entity alignment.' if portable else '<div id="l125-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','task-table'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0124-entity-task-tables.html">Lesson 124</a></nav><header><p class="stream-kicker">Year 4 · Quarter 1 · Lesson 125</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','frame-contract-viz','l125-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap: optional proposed Colab runtime; local author runtime is recorded separately.
import sys,subprocess,platform
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q',
        'torch==2.5.1','pytorch-frame==0.3.0','numpy==1.26.4','pandas==2.2.3','pyarrow==18.1.0'])
import torch,torch_frame,pandas as pd
assert torch_frame.__version__=='0.3.0', 'Use the pinned Frame API'
print({'python':platform.python_version(),'torch':torch.__version__,'frame':torch_frame.__version__,'pandas':pd.__version__})
'''
payload={name:base64.b64encode((P/'evidence/l125'/name).read_bytes()).decode() for name in ['f1-db.zip','f1-task.zip']}
full_run='''# RUN: complete real-data feature path through YOUR functions. No downloads.
import base64,json,numpy as np
from pathlib import Path
payload = '''+repr(payload)+'''
tables=load_f1_archive(base64.b64decode(payload['f1-db.zip']))
models,frames,exports,table_report=encode_f1_tables(tables)
arrays={}
for name,item in exports.items():
    arrays[name+'_ids']=np.asarray(item['ids'])
    arrays[name+'_vectors']=item['vectors'].numpy()
np.savez_compressed('l125-encoded-reg.npz',**arrays)
step=gradient_step(tables,models,frames,base64.b64decode(payload['f1-task.zip']))
report={'status':'PASS','tables':table_report,'total_rows':sum(len(t) for t in tables.values()),'gradient':step,
        'paper_result':'NOT_RUN','learner':'PENDING_WRITTEN_DEFENSE'}
assert report['total_rows']==74063 and len(tables)==9
assert step['training_queries']==23 and step['changed_parameter_tensors']>0
Path('l125-feature-report.json').write_text(json.dumps(report,indent=2))
print('Encoded rows:',report['total_rows'],'across',len(tables),'tables')
print(json.dumps(step,indent=2))
'''
parity_source=(P/'_source_check_l125.py').read_text();parity=ast.get_source_segment(parity_source,next(n for n in ast.parse(parity_source).body if isinstance(n,ast.FunctionDef) and n.name=='check_model'))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 125 · '+TITLE+'\n\n'+('Solution/reference' if solution else 'Student')+' notebook. Solve four functions; checks and the complete F1 path call your implementations. Full historical Table 2 reproduction is NOT_RUN. No hidden repository import or remote trainer is used.\n\n'+prose(True)),nbf.v4.new_markdown_cell('## SETUP\n\nThe proposed Colab bootstrap installs dependencies only in Colab. Live Colab is NOT_CHECKED. Locally, use the recorded author environment. The pinned Frame conversion may emit a pandas3 read-only NumPy warning; our numeric/categorical kernels do not mutate those inputs.'),nbf.v4.new_code_cell(bootstrap)]
 first=next(n for n in tree.body if isinstance(n,ast.FunctionDef));preamble='\n'.join(canonical.splitlines()[:first.lineno-1]);cells.append(nbf.v4.new_code_cell(preamble.strip()))
 for node in tree.body:
  if node.lineno<first.lineno:continue
  body=ast.get_source_segment(canonical,node)
  task=node.name if isinstance(node,ast.FunctionDef) and node.name in tasks else None
  if task:
   checker,hint=tasks[task];cells.append(nbf.v4.new_markdown_cell('## TODO · `'+task+'`\n\n'+hint+' Predict the CHECK output before running it.'))
   if not solution:body=body.splitlines()[0]+'\n    raise NotImplementedError("TODO: '+task+'")'
  elif isinstance(node,(ast.FunctionDef,ast.ClassDef)):
   cells.append(nbf.v4.new_markdown_cell('### PROVIDED · `'+node.name+'`\n\nRead the implementation and trace its inputs to outputs.'))
  cells.append(nbf.v4.new_code_cell(body))
  if task:cells.append(nbf.v4.new_code_cell(check_nodes[checker]+'\n'+checker+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## CHECK · pinned Frame parity\n\nThis independently compares typed tokens, row vectors, parameter gradients and one Adam update on a six-type fixture. It also checks row permutation and row locality in evaluation mode.'),nbf.v4.new_code_cell('from torch_frame.nn.models import ResNet\n'+parity+'\nprint(check_model())'),nbf.v4.new_markdown_cell('## RUN · the complete F1 feature export and real training-query gradient step\n\nEmbedded archives are checksum verified. Every row through the declared 2010 cutoff is encoded. Statistics use the declared earlier fitting population. Static creation history is missing. This does not train or score the historical Stack Exchange model.'),nbf.v4.new_code_cell(full_run)])
 cells.append(nbf.v4.new_markdown_cell('## Source appendix · visible library primitives and historical candidate\n\nThe following sources are read-only, **not executed**. Frame 0.3.0 operator/model identity and the historical candidate are separate. Original MIT licenses are retained. The missing historical data and protocol prevent a validated Table 2 training lane. See the reproduction contract for recovery hashes and commands.'))
 for group,files in [('frame',['LICENSE','torch_frame/nn/models/resnet.py','torch_frame/nn/encoder/stype_encoder.py','torch_frame/nn/encoder/stypewise_encoder.py']),('relbench',['LICENSE','relbench/external/graph.py','relbench/external/nn.py','examples/model.py','examples/gnn_node.py','examples/text_embedder.py'])]:
  for name in files:
   text=(P/'sources/l125'/group/name).read_text();cells.append(nbf.v4.new_markdown_cell('### '+group+'/'+name+'\n\n```'+('python' if name.endswith('.py') else '')+'\n'+text+'\n```'))
 cells.append(nbf.v4.new_markdown_cell('## EXIT\n\nSubmit your four implementations, `l125-encoded-reg.npz`, `l125-feature-report.json`, and your written answers from the lesson. Make one controlled intervention and explain the resulting failure. Author execution leaves learner PENDING_WRITTEN_DEFENSE.'))
 for i,c in enumerate(cells):c.id=f'l125-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
reference='''## The row-encoder contract

| Boundary | Preserve / verify |
|---|---|
| Raw table → TensorFrame | Declared semantic types; train-only fitted dictionaries/statistics |
| Numeric → token | Mean imputation, stored scale, column-specific affine map |
| Categorical → token | Disjoint vocabulary offsets; missing/unseen index0 |
| Typed blocks → tokens | TensorFrame type order and fitted column names |
| Tokens → row vector | ResNet flattens columns; same output width across tables |
| Rows → graph | Immutable entity IDs; one ID per exported source vector |
| Query → prediction | Repeated requested IDs allowed; only seeds supervised |
| Loss → encoder | Recompute live vectors; detached caches cut gradients |

## Core equations

`numeric = ((fill(x, μ) − μ) / scale)[...,None] × W + b`

`category_index = 0 if missing else local_ID + column_offset + 1`

`offset[j] = sum(cardinality[k] for k < j)`

`row = decoder(residual_blocks(flatten(tokens)))`

`neighbor_mean[target] = sum(source_vectors) / max(1, degree[target])`

## Evidence boundaries

74,063 F1 rows: complete course feature export and original Frame parity. Real training-query gradient step:23 seeds. Frame Table2 Stack Exchange experiment:NOT_RUN (missing archives and unresolved historical protocol). No predictive-performance claim. Static creation histories unavailable; frozen hashed text adapter is not pretrained. Learner:PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0125-pytorch-frame-deep-dive.html) · [Notebook](../labs/0125-pytorch-frame-deep-dive.ipynb) · [Protocol](../labs/l125-reproduction.md) · [Source manifest](../labs/_sources_l125.json).
'''
(R/'reference/pytorch-frame-deep-dive.html').write_text(document('PyTorch Frame · reference',reference))
print('Built L125 lesson, reference and standalone notebooks')
