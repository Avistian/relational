"""Build canonical HTML and self-contained notebooks from shared prose and code."""
import ast,base64,hashlib,json,re
from pathlib import Path
from _walkthrough_delivery import snapshot, finalize
snapshot(96)
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0096-multi-relational-data';T='Multi-relational data: SQL foreign keys become graph routes'
result=json.loads((P/'_experiment_l096_results.json').read_text())
def prose(portable=False):
    s=(R/'lessons/content'/f'{S}.md').read_text()
    src='data:image/png;base64,'+base64.b64encode((P/'figures/l096/schema.png').read_bytes()).decode() if portable else '../labs/figures/l096/schema.svg'
    s=s.replace('[[SCHEMA_FIG]]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="Four-table schema with typed primary-key maps, FK routes, row counts, preserved line nodes, and SQL query audit."></div><figcaption>The entire course fixture. Teal links are forward child-to-parent roles; reverse stores are derived views.</figcaption></figure>')
    s=s.replace('[[INTERVENTION]]','**Portable intervention:** use the prediction and intervention cell after the implementation.' if portable else '<div id="schema-intervention"></div>')
    s=s.replace('[[RESULTS]]',f"**Author execution: {result['status']}.** All {result['databases']} declared databases have exact equality of graph and SQL paths, totals and customer-order counts. These are frozen author results; your notebook recomputes them from scratch.")
    if portable:s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s).replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
    return s
head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 096 — '+T+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','hetero-graph-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0095-bipartite-graphs.html">Lesson 95</a> · <a href="../reference/schema-graph-contract.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 096</p><h1>'+T+'</h1></header>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose())+'</article><script src="../assets/schema-graph-viz.js"></script><script src="../assets/l096-lesson.js"></script></body></html>')
source=(P/'relkit/schema_l096.py').read_text();tree=ast.parse(source)
checks={
'key_index':"assert key_index([{'a':100,'b':2},{'a':100,'b':1}],('a','b'))=={(100,1):0,(100,2):1}\nfor rows in [[{'a':1},{'a':1}],[{'a':None}]]:\n    try: key_index(rows,('a',))\n    except ValueError: pass\n    else: raise AssertionError('Invalid PK accepted')\nprint('PASS: composite keys, stable ordering, invalid keys rejected')",
'fk_edges':"e=fk_edges([{'p':None},{'p':30}],('p',),{(10,):0,(30,):1},True)\nassert e.tolist()==[[1],[1]]\nassert fk_edges([],('p',),{},True).shape==(2,0)\nfor value,nullable in [(999,True),(None,False)]:\n    try: fk_edges([{'p':value}],('p',),{(10,):0},nullable)\n    except ValueError: pass\n    else: raise AssertionError('Invalid FK accepted')\nprint('PASS: NULL, orphan and empty relation contracts')",
'graph_query':"g=build_graph(fixture())\nanswer=graph_query(g)\nassert answer=={'paths':[[10,100,1,10,2],[10,100,2,10,3],[10,100,3,50,1],[30,200,1,50,4]],'totals':[[10,10,5],[10,50,1],[30,50,4]],'customer_order_counts':[[10,2],[30,1],[90,0]]}\nprint('PASS: all paths, quantities and isolated customer preserved')"}
bootstrap="""# @colab-bootstrap
import sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','torch==2.8.0','torch-geometric==2.6.1'])
import importlib.metadata as metadata
print({k:metadata.version(k) for k in ['torch','torch-geometric']})
"""
for solution in [False,True]:
    cells=[nbf.v4.new_markdown_cell('# '+T+'\n\nTier C · full declared schema/query experiment. No repository imports or dataset downloads. Three live tasks; CPU only. PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
    cells.extend(nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=\n## )',prose(True)) if x.strip())
    cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nRead the schema constants first. A supplied SQL engine will serve as an independent oracle; it never calls your graph evaluator.'))
    prelude='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))
    cells.append(nbf.v4.new_code_cell(prelude))
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        code=ast.get_source_segment(source,node);name=node.name
        cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if name in checks else '### PROVIDED · ')+name+'\n\n'+(ast.get_docstring(node) or '')))
        if name in checks and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement the declared contract")'
        cells.append(nbf.v4.new_code_cell(code,metadata={'task':name} if name in checks else {}))
        if name in checks:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+name),nbf.v4.new_code_cell(checks[name])])
    cells.extend([nbf.v4.new_markdown_cell('## Predict → intervene → explain\n\nMove line (100,2) to product 50. Write the new customer 10 totals before executing. Which keys and row counts stay fixed?'),nbf.v4.new_code_cell("changed=fixture();changed['line_item'][1]['product_id']=50\nafter=graph_query(build_graph(changed))\nassert after==sql_query(changed)\nassert after['totals']==[[10,10,2],[10,50,4],[30,50,4]]\nprint(after)"),nbf.v4.new_markdown_cell('## Complete reproduction run\n\nRun all 33 declared databases afresh. The hash comparison below uses labeled frozen author references; SQLite still independently recomputes every expected query.'),nbf.v4.new_code_cell('fresh=run_suite()\nAUTHOR_RECORDS_SHA256='+repr(hashlib.sha256(json.dumps(result['records'],sort_keys=True).encode()).hexdigest())+'\nassert hashlib.sha256(json.dumps(fresh["records"],sort_keys=True).encode()).hexdigest()==AUTHOR_RECORDS_SHA256\nfrom pathlib import Path\nPath("l096-fresh.json").write_text(json.dumps(fresh,indent=2))\nprint(f"PASS: {fresh[\'databases\']} complete databases, every path, total and outer count")'),nbf.v4.new_markdown_cell('## EXIT · PENDING_WRITTEN_DEFENSE\n\nSubmit the graph specification, the complete-suite output and your section 8 written defense. Explain the two role names, composite identity, the 1/2/5 distinction, NULL versus orphan, and one temporal leakage route. Ask the agent for feedback. Teacher execution is not learner mastery.')])
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in cells if c.cell_type=='code']
        if [c.source for c in before]==[c.source for c in after]:
            for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
    for i,c in enumerate(cells):c.id=f'l096-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
    nbf.validate(nb);nbf.write(nb,path)
    if solution:
        html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
print('Built lesson, student, solution and HTML')

finalize(96, preview='solution')
