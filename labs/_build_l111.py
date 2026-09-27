"""Build the OGB protocol lesson and portable visible-code notebook pair."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
from nbconvert import HTMLExporter
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0111-ogb-benchmark-contract'
source=(P/'relkit/ogb_contract_l111.py').read_text();source=source.split("if __name__ == '__main__':")[0]
report=json.loads((P/'evidence/l111/summary.json').read_text())
text=(R/'lessons/content'/f'{S}.md').read_text()
results='| Split | Official nodes | Majority accuracy |\n|---|---:|---:|\n'+''.join(f'| {k} | {report["split_counts"][k]:,} | {100*v:.4f}% |\n' for k,v in report['accuracy'].items())
results+='\nChosen training class: **'+str(report['chosen_class'])+'**. [Full identity and score report](../labs/evidence/l111/summary.json).'
text=text.replace('[[RESULTS]]',results)
caption='One graph, three label roles. Label splitting and feature visibility are separate contracts.'
fig=f'<figure class="stream-figure" tabindex="0"><img src="../labs/figures/l111/contract.svg" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
text=text.replace('[[FIG:contract]]',fig)
def doc(title,body):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 111 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0110-temporal-gnn-checkpoint.html">Lesson 110</a></nav><h1>'+title+'</h1>'+render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article></body></html>'
(R/'lessons'/f'{S}.html').write_text(doc('OGB setup and the benchmark contract',text))
ref='''## Five objects

Dataset: graph and features. Split: official node identities. Predictor: inputs to answers. Evaluator: aligned answers to a metric. Protocol: all choices defining the experiment.

## Accuracy

Correct class IDs / evaluated nodes. OGB arxiv uses [N,1] class IDs. Index labels and predictions by the same official split.

## Information boundary

Train labels fit; validation selects; test evaluates. A chronological label split does not certify a historically available feature graph. Transductive propagation can use held-out features without fitting on held-out labels.

## Before comparing scores

Record release, split identities, transformations, model, schedule, selection, metric and runs. Constant majority is a plumbing baseline, not a GCN reproduction.

[Lesson](../lessons/0111-ogb-benchmark-contract.html) · [Protocol](../labs/l111-reproduction.md) · [Official specification](https://ogb.stanford.edu/docs/nodeprop/)
'''
(R/'reference/ogb-benchmark-contract.html').write_text(doc('OGB benchmark contract · reference',ref))
portable=text.replace('../labs/figures/l111/contract.svg','data:image/png;base64,'+base64.b64encode((P/'figures/l111/contract.png').read_bytes()).decode())
portable=portable.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
portable=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
checks=(P/'_check_l111.py').read_text();checks=re.sub(r'^from relkit[^\n]*\n','',checks,flags=re.M)
for solution in [False,True]:
 code=source
 if not solution:
  tree=ast.parse(code)
  for node in reversed(tree.body):
   if isinstance(node,ast.FunctionDef) and node.name in ['audit_splits','majority_predictions']:
    lines=code.splitlines();lines[node.lineno-1:node.end_lineno]=[lines[node.lineno-1],'    """TODO: implement the contract described in the lesson."""','    raise NotImplementedError("TODO: '+node.name+'")'];code='\n'.join(lines)+'\n'
 cells=[nb.v4.new_markdown_cell('# Lesson 111 · OGB benchmark contract\n\n'+portable),nb.v4.new_code_cell("# @colab-bootstrap\nimport sys, subprocess\nif 'google.colab' in sys.modules:\n    subprocess.check_call([sys.executable,'-m','pip','install','ogb==1.3.6'])"),nb.v4.new_markdown_cell('## Implement, then check\n\nThe two TODO functions below are used by the full-data run. First predict the effect of overlapping split IDs and changed held-out labels.'),nb.v4.new_code_cell(code),nb.v4.new_code_cell(checks),nb.v4.new_markdown_cell('## Full official-data course baseline\n\nDownloads the hash-pinned OGB archive when absent. CPU only; no neural training. Each invocation parses raw data using the official loader. This report is separate from L112 GCN reproduction.'),nb.v4.new_code_cell("report=run_contract(os.environ.get('L111_ARCHIVE','l111-data/arxiv.zip'))\nPath('l111-report.json').write_text(json.dumps(report,indent=2))\nprint(json.dumps(report,indent=2))")]
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 dest=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and dest.exists():
  old=nb.read(dest,as_version=4);a=[c for c in notebook.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   for c,d in zip(a,b):c.outputs=d.outputs;c.execution_count=d.execution_count
 nb.write(notebook,dest)
 if solution and all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(notebook);(P/'html'/f'{S}.html').write_text(html)
print('Built L111 lesson, reference and notebooks')
