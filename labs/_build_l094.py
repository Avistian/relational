"""Canonical lesson and portable notebook builder; retains outputs only for identical code."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;SLUG='0094-hin-survey';TITLE='HIN survey: map the mechanism, audit the evidence'
manifest=json.loads((P/'_sources_l094.json').read_text());result=json.loads((P/'_paper_l094_results.json').read_text())

def measured():
 s='| NN statistic | Paper Table 1 | Complete released NN | Status |\n|---|---:|---:|---|\n'
 for k,v in result['comparison'].items():s+=f"| {k} | {v['paper']:,} | {v['released']:,} | {v['status']} |\n"
 st=result['statistics'];s+=f"\nThe release has **{st['forward_edges']:,} forward entries**, including **{st['unlisted_forward_edges']:,} field-hierarchy entries**. Adding the verified reverse entries gives **{st['all_stored_edges']:,}**. Its five listed forward edge families sum to **{st['listed_forward_edges']:,}**. These are distinct counting conventions, not alternative model scores."
 return s

def arithmetic():
 s='| Printed row | Sum of five node types − printed node total | Sum of five edge families − printed edge total |\n|---|---:|---:|\n'
 for k,v in result['printed_table_arithmetic'].items():s+=f"| {k} | {v['node_delta']:,} | {v['edge_delta']:,} |\n"
 return s+'\nThe CS node sum differs by 15,238. The NN listed edge sum differs by −1,794,907. These checks expose accounting questions; they do not establish which entry or convention is wrong.'

prior='''| Existing artifact | Named target / executed coverage | Evidence boundary |
|---|---|---|
| L091 | AIFB Table 2; 10 complete release-port runs | Historical backend/seed identity INCOMPARABLE |
| L092 | ACM Table 3 KNN; one encoder, 40 KNN evaluations | Repeated probes are not independent encoder runs; original MAT identity unestablished |
| L093 | CS Table 2 paper-field L2; zero completed CS fits | NOT_RUN; NN teaching fits cannot substitute for CS |
'''

def prose(portable=False):
 s=(R/'lessons/content'/f'{SLUG}.md').read_text().replace('[[MEASURED_TABLE]]',measured()).replace('[[ARITHMETIC_TABLE]]',arithmetic()).replace('[[PRIOR_EVIDENCE]]',prior)
 for tag,name,caption in [('FAMILY_FIG','family-map','Encoder and route are separate axes; follow each input through its mechanism to the task handoff.'),('ROUTE_FIG','route-trace','Synthetic path-count trace with four authors. The same IDs yield different pair connectivity under different routes.')]:
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l094/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l094/{name}.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 for tag,id_,fallback in [('WARMUP','warmup','**Cold retrieval:** answer the three questions below before reading their model answers.'),('ROUTE_WIDGET','route-widget','**Portable intervention:** use the route-matrix cells below to remove Bo–P1; predict Ada–Bo counts first.'),('PREDICTION','prediction','**Commit first:** can HAN belong to both the explicit-meta-path and GNN categories? Explain before reading the map.'),('TEACHBACK','teachback','**Teach back:** write the EXIT defense before checking the reference map.')]:s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id_}"></div>')
 if portable:
  s=re.sub(r'(?<=href=")(00\d\d-[^"]+\.html)',r'https://avistian.github.io/relational/lessons/\1',s)
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 94 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','hin-survey'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0093-hgt.html">Lesson 93</a> · <a href="../reference/hin-taxonomy.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 094</p><h1>'+TITLE+'</h1></header>'
footer='</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','arch-family-viz','l094-lesson'])+'</body></html>'
(R/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
source=(P/'relkit/hin_l094.py').read_text();tree=ast.parse(source)
tasks={
'compose_path':('Validate typed continuity; compute path multiplicities. A relation matrix has source rows and target columns.',"ap=np.array([[1,1,0],[0,1,1],[0,0,0]])\nr={('A','writes','P'):ap,('P','by','A'):ap.T}\nnp.testing.assert_array_equal(compose_path(r,list(r)),[[2,1,0],[1,2,0],[0,0,0]])\ntry:\n    compose_path(r,[('A','writes','P'),('A','writes','P')])\nexcept ValueError:\n    print('PASS: invalid path rejected; counts and isolated row correct')\nelse:\n    raise AssertionError('Type continuity must be checked')"),
'classify_method':('Return labels for encoder and route independently. Supported encoder values: lookup, message_passing, fixed. Supported routes: explicit_path, one_hop, branching.',"assert classify_method({'encoder':'message_passing','route':'explicit_path'})==['GNN','explicit meta-path']\nassert classify_method({'encoder':'lookup','route':'explicit_path'})==['lookup embedding','explicit meta-path']\nassert classify_method({'encoder':'message_passing','route':'one_hop'})==['GNN','one-hop composition']\nprint('PASS: HAN crosses categories; a shared route does not imply a shared encoder')"),
'compare_protocols':('Inspect dataset_hash, task, split_hash, features, target_mask, selection, budget, metric and aggregation. Known conflicts mean INCOMPARABLE; absent/None/empty/UNKNOWN fields mean NOT_ESTABLISHED unless a conflict already proves incomparability. Otherwise COMPARABLE. Return status, differences and optional unknown fields.',"p={k:'fixed' for k in ['dataset_hash','task','split_hash','features','target_mask','selection','budget','metric','aggregation']}\nassert compare_protocols(p,p)=={'status':'COMPARABLE','differences':[]}\nassert compare_protocols(p,dict(p,metric='changed'))['status']=='INCOMPARABLE'\nassert compare_protocols({}, {})['status']=='NOT_ESTABLISHED'\nassert compare_protocols(dict(p,metric=None),dict(p,metric=None))['status']=='NOT_ESTABLISHED'\nprint('PASS: missing information is not agreement')")}
loader=(P/'relkit/oag_read_l094.py').read_text().replace('from hin_l094 import file_sha256\n','')
table=json.loads((P/'sources/hin-l094/table1.json').read_text())
# Freeze exact prior evidence once; do not refresh on regeneration.
archive=P/'sources/hin-l094';snapshot=archive/'prior-evidence.json'
if not snapshot.exists():
 records={}
 for n in [91,92,93]:
  path=P/f'_paper_l{n:03}_results.json';b=path.read_bytes();records[str(n)]={'filename':path.name,'sha256':hashlib.sha256(b).hexdigest(),'record':json.loads(b)}
 snapshot.write_text(json.dumps(records,indent=2)+'\n')
prior_records=json.loads(snapshot.read_text())
for item in prior_records.values():
 raw=item['record']; item['record']={k:raw[k] for k in ['status','historical_exact_parity','historical_parity','full_paper_parity','training_runs','knn_per_training_run','completed_CS_fits','planned_CS_fits','source_revision','data_sha256'] if k in raw}
 if 'runs' in raw:item['record']['recorded_training_runs']=len(raw['runs'])
 item['scope']='Selected fields; full parsed record is archived in sources/hin-l094/prior-evidence.json'
bootstrap="""# @colab-bootstrap
import os,sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','pandas==2.3.2','dill==0.3.8'])
import json,hashlib,urllib.request
from pathlib import Path
import numpy as np
import importlib.metadata as metadata
print({k:metadata.version(k) for k in ['numpy','pandas','dill']})
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nStandalone survey lab. Complete three TODO functions and their CHECK cells. The full NN graph audit is a separate explicit switch; frozen author evidence is labeled. No local relkit imports or model-training dependencies are required.'),nbf.v4.new_code_cell(bootstrap)]
 cells += [nbf.v4.new_markdown_cell(s) for s in re.split(r'(?=\n## )',prose(True)) if s.strip()]
 cells += [nbf.v4.new_markdown_cell('## Exercises: visible implementation\n\nPROVIDED functions implement the complete data audit. Your TODO functions are used by the route intervention, taxonomy export and comparison below.')]
 for node in tree.body:
  if isinstance(node,(ast.Import,ast.ImportFrom)) or isinstance(node,ast.Expr):continue
  name=node.name;code=ast.get_source_segment(source,node)
  if name in tasks:
   desc,check=tasks[name];cells.append(nbf.v4.new_markdown_cell('### TODO · '+name+'\n\n'+desc))
   if not solution:
    signature=code[:code.index('\n')];code=signature+'\n    raise NotImplementedError("Complete this task")'
   cells.append(nbf.v4.new_code_cell(code,metadata={'task':name}));cells.append(nbf.v4.new_code_cell(check,metadata={'check':name}))
  else:cells +=[nbf.v4.new_markdown_cell('### PROVIDED · '+name+'\n\n'+(ast.get_docstring(node) or 'Streaming SHA-256 identifies exact bytes.')),nbf.v4.new_code_cell(code)]
 cells +=[nbf.v4.new_markdown_cell('### CHECK · intervention uses your route composer'),nbf.v4.new_code_cell("ap=np.array([[1,1,0,0],[0,1,1,0],[0,0,0,1],[0,0,0,0]])\npv=np.array([[1,0],[0,1],[0,1],[1,0]])\ndef routes(ap):\n    r={('A','writes','P'):ap,('P','by','A'):ap.T,('P','at','V'):pv,('V','hosts','P'):pv.T}\n    return compose_path(r,[('A','writes','P'),('P','by','A')]),compose_path(r,[('A','writes','P'),('P','at','V'),('V','hosts','P'),('P','by','A')])\npaper,venue=routes(ap);assert (paper[0,1],venue[0,1],paper[0,2],venue[0,2])==(1,2,0,1)\nchanged=ap.copy();changed[1,1]=0;p2,v2=routes(changed);assert (p2[0,1],v2[0,1])==(0,1)\nprint('Baseline shared paper:\\n',paper,'\\nShared venue:\\n',venue,'\\nAfter removal, Ada–Bo:',p2[0,1],v2[0,1])")]
 cells +=[nbf.v4.new_markdown_cell('### PROVIDED · exact published statistics and prior-run records\n\nThese are author-reference evidence, not newly trained models. Their byte hashes identify the source files; the notebook stores selected fields for portable inspection; full parsed records remain in the provenance archive.'),nbf.v4.new_code_cell('PAPER_TABLE = json.loads('+repr(json.dumps(table))+')\nAUTHOR_AUDIT = json.loads('+repr(json.dumps(result))+')\nPRIOR_EVIDENCE = json.loads('+repr(json.dumps(prior_records))+')\nMANIFEST = json.loads('+repr(json.dumps(manifest))+')'),nbf.v4.new_code_cell("for name,row in PAPER_TABLE.items():print(name,table_arithmetic(row))\nassert table_arithmetic(PAPER_TABLE['CS'])['node_delta']==15238\nassert table_arithmetic(PAPER_TABLE['NN'])['edge_delta']==-1794907\nassert PRIOR_EVIDENCE['93']['record']['status']=='NOT_RUN'\nassert PRIOR_EVIDENCE['92']['record']['training_runs']==1\nprint('Prior evidence inspected; fresh training fits in this notebook: 0')")]
 cells +=[nbf.v4.new_markdown_cell('### PROVIDED · complete read-only OAG loader\n\nCompatibility code is copied from the audited L093 loader. Only the SHA-256-pinned author archive is loaded. Old defaultdict factories are not executed. This is a read-only port, not proof of historical preprocessing identity.'),nbf.v4.new_code_cell(loader)]
 cells +=[nbf.v4.new_markdown_cell('### Full published-data target · Table 1 NN\n\nSet RUN_FULL_GRAPH=True to audit every entry. The 656 MB download needs sufficient disk/RAM; no GPU is needed. The default keeps downloads explicit. Existing local data can be supplied through L094_DATA_PATH. Result JSON includes every relation count and a comparison with the printed row. CS and OAG are NOT_RUN in this notebook.'),nbf.v4.new_code_cell("RUN_FULL_GRAPH = os.environ.get('L094_RUN_FULL_GRAPH') == '1'\nif RUN_FULL_GRAPH:\n    path=Path(os.environ.get('L094_DATA_PATH','graph_NN.pk'))\n    if not path.exists():\n        temporary=path.with_suffix('.part')\n        urllib.request.urlretrieve(MANIFEST['nn_data']['url'],temporary)\n        assert file_sha256(temporary)==MANIFEST['nn_data']['sha256']\n        temporary.replace(path)\n    graph=load_oag(path,MANIFEST['nn_data']['sha256'])\n    observed=count_graph(graph)\n    assert observed==AUTHOR_AUDIT['statistics'], 'Graph content or counting implementation changed'\n    assert observed['reverse_content_matches']\n    Path('l094-inline-graph.json').write_text(json.dumps(observed,indent=2))\n    for key,value in observed['table_columns'].items():\n        print(key,'paper:',PAPER_TABLE['NN'][key],'released:',value)\n    print('Full released NN audited; historical parity remains NOT_ESTABLISHED')\nelse:\n    print('Fresh full-graph audit NOT_RUN in this kernel; author evidence above is frozen')")]
 cells +=[nbf.v4.new_markdown_cell('## EXIT · taxonomy and written defense\n\nRun the export, then add parameterization, feature/supervision assumptions and a primary-source locator for every row. Explain why a compatible protocol is not a reproduction claim. Finally propose one falsifiable experiment for the database mission. Status remains PENDING_WRITTEN_DEFENSE until you submit it.'),nbf.v4.new_code_cell("methods={'R-GCN':{'encoder':'message_passing','route':'one_hop'},'HAN':{'encoder':'message_passing','route':'explicit_path'},'HGT':{'encoder':'message_passing','route':'one_hop'},'metapath2vec':{'encoder':'lookup','route':'explicit_path'}}\ntaxonomy={name:classify_method(record) for name,record in methods.items()}\nPath('l094-taxonomy.json').write_text(json.dumps(taxonomy,indent=2))\nprint(json.dumps(taxonomy,indent=2))\n# Different dataset and metric: never compare these as a leaderboard.\nleft=dict(p,dataset_hash='AIFB',metric='accuracy');right=dict(p,dataset_hash='ACM',metric='KNN F1')\nassert compare_protocols(left,right)['status']=='INCOMPARABLE'\nprint('EXIT: PENDING_WRITTEN_DEFENSE')")]
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});path=P/('solutions' if solution else '')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 for i,c in enumerate(nb.cells):c.id=f'l094-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if solution:
  html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{SLUG}.html').write_text(html)
print('Built lesson, student and solution notebooks, and lab HTML')
