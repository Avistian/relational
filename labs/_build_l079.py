"""Build the writing capstone and standalone executable audit from canonical sources."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from bs4 import BeautifulSoup
from _figures_l079 import build as figures
LAB=Path(__file__).resolve().parent;ROOT=LAB.parent
SLUG='0079-neural-tabular-decision-guide';TITLE='Year 2 essay: choose a model you can defend'
CAPTIONS={'decision':'Worked fraud case: information eligibility precedes evaluation, measured feasibility and validation-only selection. Each gate carries a concrete requirement.','ranks':'Reanalysis of corrected L060 v2 predictions: seed means ranked within each of the same three underlying datasets, then averaged. No causal interpretation or fresh training.','budget':'Synthetic fixture: the 5 ms constraint excludes TabM and ICL despite lower validation losses. These are illustrative latencies, not measured model comparisons.'}
SRC=(LAB/'relkit/decision_guide.py').read_text()
NODES={n.name:ast.get_source_segment(SRC,n) for n in ast.parse(SRC).body if isinstance(n,ast.FunctionDef)}
CHECKS={
'select_feasible':"""candidates=[dict(model='Trees',validation_loss=.32,p95_ms=2),dict(model='TabM',validation_loss=.29,p95_ms=8),dict(model='ICL',validation_loss=.27,p95_ms=35)]
assert [select_feasible(candidates,b) for b in [1,5,10,40]]==[None,'Trees','TabM','ICL']
assert select_feasible([dict(model='A',validation_loss=.2,p95_ms=2),dict(model='B',validation_loss=.2,p95_ms=1)],2)=='A'
changed=[dict(r,test_loss=i*10) for i,r in enumerate(candidates)]
assert select_feasible(changed,40)=='ICL', 'Test losses must not select candidates'
try: select_feasible(candidates,float('nan'))
except ValueError: pass
else: raise AssertionError('Reject invalid budgets')
print('CHECK: feasible set, minimum validation loss, tie rule and test isolation pass')""",
'matched_datasets':"""fixture={'random':['A/random','B/random','C/random'],'temporal':['C/temporal','A/temporal']}
assert matched_datasets(fixture)=={'random':['A/random','C/random'],'temporal':['A/temporal','C/temporal']}
try: matched_datasets({'random':['A/random'],'temporal':['B/temporal']})
except ValueError: pass
else: raise AssertionError('Do not invent paired data')
print('CHECK: matched underlying identities, deterministic order, empty intersection pass')"""}

def manuscript(notebook=False):
    s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text();r=json.loads((LAB/'_verify_l079_results.json').read_text())
    table='**Fresh author reanalysis of frozen predictions; not new training.**\n\n| Method | Random (11) | Matched random (3) | Temporal (3) |\n|---|---:|---:|---:|\n'
    for arm in r['summary']['random']['arms']:
        vals=[r['summary']['random']['mean_ranks'][arm],r['matched']['random']['mean_ranks'][arm],r['matched']['temporal']['mean_ranks'][arm]]
        table+='| '+arm+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |\n'
    s=s.replace('<!--results-->',table)
    for name,caption in CAPTIONS.items():
        src='data:image/png;base64,'+base64.b64encode((LAB/'figures/l079'/f'{name}.png').read_bytes()).decode() if notebook else f'../labs/figures/l079/{name}.png'
        graphic=f'<img src="{src}" alt="{caption}">'
        if not notebook:graphic=f'<span class="dg-scroll-note">On small screens, scroll the figure horizontally for full-size labels.</span><div class="dg-image" tabindex="0" role="region" aria-label="Scrollable {name} figure">{graphic}</div>'
        s=s.replace('<!--figure:'+name+'-->',f'<figure class="dg-figure">{graphic}<figcaption>{caption}</figcaption></figure>')
    if notebook:
        s=re.sub(r'<div id="[^"]+"></div>','',s)
        s=re.sub(r'\((00\d[^)]*\.html)\)',r'(../lessons/\1)',s)
    return s

def notebook(solution=False):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# Lab 079 · A decision guide with reproducible evidence\n\n**Core deliverable: your one-page essay.** Optional executable route: audit the complete corrected L060 v2 predictions, then apply a synthetic constrained-selection rule. No training or new paper reproduction. The complete compressed input is embedded below so the notebook does not depend on an unpublished repository checkout.')
    md('### PROVIDED · standalone environment\n\nOnly analysis dependencies are needed. On Colab the cell installs the exact author-observed versions. Locally use `requirements-l079-observed.txt`. Live Colab remains NOT_CHECKED. No model weights or remote dataset downloads are required.')
    code("# @colab-bootstrap — standalone L079 analysis; no repository clone\nimport sys, subprocess\nif 'google.colab' in sys.modules:\n    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.5.0','pandas==3.0.3','scipy==1.18.0'])\nprint('Analysis environment:',sys.version)")
    for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
        if part.strip():md(part)
    md('## PROVIDED · analysis imports\n\nThe toy scenario has lower-is-better validation loss and complete-path p95 latency. Python input order is the declared tie breaker.')
    code('import math, json, gzip, base64, hashlib, platform\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport scipy')
    for name,task in [('select_feasible','Validate a finite nonnegative budget and finite nonnegative losses/latencies; reject duplicate names. Filter candidates at or below the budget, then return the name with minimum validation loss. Preserve input order on exact ties; return None when none qualifies. Ignore any test-loss field.'),('matched_datasets','Require random and temporal panels with unique dataset/regime IDs. Intersect the underlying dataset names, sort them, and return both panels restricted to those names. Reject an empty intersection or malformed regime suffix.')]:
        md('## TODO · '+name+'\n\n'+task)
        code(NODES[name] if solution else NODES[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
        code('# CHECK — your live function\n'+CHECKS[name])
    md('## PROVIDED · full frozen evidence\n\nAll 210 records include saved targets, predictions, row IDs, candidate validation losses, selected indices and timings. This is Tier A/B evidence from L060, not newly sampled data. Check compressed and uncompressed hashes before parsing. The SHA256 pins bytes; the following audit checks their internal consistency. It cannot prove an unobserved historical training process was leakage-free.')
    m=json.loads((LAB/'_sources_l079.json').read_text());payload=base64.b64encode((LAB/'data/l079/l060-v2.json.gz').read_bytes()).decode()
    code('manifest='+repr(m)+'\ncompressed=base64.b64decode('+repr(payload)+')\nassert hashlib.sha256(compressed).hexdigest()==manifest["local_sha256"]["data/l079/l060-v2.json.gz"]\nraw=gzip.decompress(compressed)\nassert hashlib.sha256(raw).hexdigest()==manifest["evidence_sha256"]\nevidence=json.loads(raw)\nprint(len(evidence["records"]),"complete frozen prediction records")')
    source=(LAB/'relkit/comparison_l060.py').read_text();tree=ast.parse(source)
    code('\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))))
    explanations={'validate_partitions':'Reject overlapping or duplicate train/validation/test IDs.','choose_validation':'Reconstruct the declared first-minimum validation selection.','score_predictions':'Recompute binary log loss or regression RMSE from saved predictions.','aggregate_panel':'Require the exact dataset × arm × seed design; average seeds before dataset ranks. Conditional seed SD and exploratory rank tests remain separate.','paired_effect':'Retained canonical helper for a same-dataset paired seed difference; does not combine incompatible metrics.','audit_records':'Check row order, shared targets, candidate selection and all recomputed metrics, then build separate regime panels.'}
    for n in tree.body:
        if isinstance(n,ast.FunctionDef):
            md('### PROVIDED · '+n.name+'\n\n'+explanations[n.name]);code(ast.get_source_segment(source,n))
    md('## RUN · full audit and matched-population analysis\n\nThis calls your live matching function and the visible complete audit. No stored author summary is used to compute ranks. A passed check does not certify original paper parity or availability of historical source features.')
    code("""summary=audit_records(evidence)
paired=matched_datasets(evidence['design']['panels'])
matched={regime:aggregate_panel([r for r in evidence['records'] if r['dataset'] in ids],ids,evidence['design']['arms'],evidence['config']['seeds']) for regime,ids in paired.items()}
table=pd.DataFrame({'random11':summary['random']['mean_ranks'],'matched_random3':matched['random']['mean_ranks'],'temporal3':matched['temporal']['mean_ranks']})
print(table.round(3).to_string())
print('Underlying paired datasets:',paired)
results={'evidence_sha256':manifest['evidence_sha256'],'records':len(evidence['records']),'summary':summary,'matched':matched,'choices':{str(b):select_feasible(candidates,b) for b in [1,5,10,40]},'scope':'Frozen prediction analysis only; no fresh predictive training'}
Path('l079-student-results.json').write_text(json.dumps(results,indent=2)+'\\n')
print('Saved l079-student-results.json')""")
    md('## EXIT · the one-page guide is the assessed artifact\n\nWrite the table from section 5 in the blank cell below, followed by your four-sentence fraud recommendation, evidence ledger and reproduction footer. Include a paper-supported challenger absent from the local evidence and say what you would need to measure. Explain why matching three dataset names is necessary but insufficient for a causal claim. The agent grades your reasoning; the CHECK cells only grade code behavior.')
    md((ROOT/'solutions/l079-example-guide.md').read_text() if solution else '**YOUR GUIDE — write here.**\n\nRegime | Baseline/challenger | Split/metric | Budget | Evidence/scope | Falsifier\n\nReproduction footer: ...\n\nFraud recommendation: ...\n\nSelf-score and uncertainty: ...')
    md('## Reproduction extension · fresh training is a separate experiment\n\nFrom a full course checkout, the corrected L060 operator can train all five arms on the declared 14 cells with `.venv/bin/python labs/_verify_l060.py --preset lab --output /tmp/l060-l079-fresh.json` (choose a new path). See [L060 contract](../labs/l060-reproduction.md) for data acquisition, caps and source variants. This is the full **local** experiment, not a TabM/RealMLP/TabArena paper reproduction. L079 does not run it or embed its trainers; follow the source-visible [L060 notebook](../labs/0060-broad-model-comparison.ipynb). See [L079 contract](../labs/l079-reproduction.md) for exact analysis commands and evidence boundaries.')
    for i,c in enumerate(cells):c.id=f'l079-{i:03d}'
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})

def build():
    from _walkthrough_delivery import snapshot, finalize
    snapshot(79)
    figures()
    head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/decision-guide.css"></head><body><article>'
    nav=f'<nav><a href="../index.html">Course</a> · <a href="0078-message-passing-preview.html">Lesson 78</a></nav><header><p>Year 2 · Quarter 4 · Lesson 079</p><h1>{TITLE}</h1><p>From model familiarity to a falsifiable recommendation.</p></header><aside><a href="../labs/l079-decision-template.md">Essay template</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/{SLUG}.ipynb">Notebook</a> · <a href="../reference/{SLUG}.html">Reference</a></aside>'
    body=render(manuscript()).replace('<table>','<div class="dg-table"><table>').replace('</table>','</table></div>')
    scripts=''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','teachback','decision-guide-viz','l079-lesson'])
    (ROOT/'lessons'/f'{SLUG}.html').write_text(head+nav+body+'</article>'+scripts+'</body></html>')
    for solution in [False,True]:
        nb=notebook(solution);p=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb'
        if solution and p.exists():
            old=nbf.read(p,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
            if [c.source for c in a]==[c.source for c in b]:
                for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count
        nbf.write(nb,p)
    preview,_=HTMLExporter().from_notebook_node(notebook());soup=BeautifulSoup(preview,'html.parser')
    for a in soup.find_all('a',href=True):
        if a['href'].startswith('../'):a['href']='../'+a['href']
    (LAB/'html'/f'{SLUG}.html').write_text(str(soup)+'\n')
    ref='''# Decision guide · one-page reference

**Order:** available information → deployment split → feasible shortlist → validation selection → locked test.

**Shortlist:** trees as a strong baseline; RealMLP/TabM for strong MLP recipes and efficient ensembling; FT-Transformer for a feature-token interaction hypothesis; pinned TabPFN/TabICL for supported in-context tasks; honest OOF ensembles for useful diversity. Text and relational history first require an explicit availability-safe representation.

**A defensible row:** regime | baseline/challenger and mechanism | split/metric | budget | evidence and scope | change-my-mind result.

**Evidence arithmetic:** average model seeds within dataset → rank methods within dataset (ties get mean rank) → average datasets equally. Never pool incompatible losses. Match dataset names before comparing split panels; matched names still do not make a causal intervention.

**Constrained selection:** filter by measured complete-path latency/memory, then minimize validation loss. Prespecify tie rule. An empty feasible set means revise the plan. Test scores never choose the candidate.

**Reproduction footer:** model/checkpoint/source revision; data/split identities; availability cutoff; preprocessing fit scope; model/search seeds; environment; selection rule; metric/aggregation; measured serving context; command, hashes and deviations.

**L079 evidence:** all 210 corrected L060 v2 frozen predictions audited. Local reduced recipes; FT-Transformer/ICL absent. Fresh training and published-paper reproduction NOT_RUN. Synthetic latency values are not serving measurements.

**Exit:** 6–8 rows, ≤650 words, fraud recommendation, evidence ledger and footer. Six 0–2 criteria: split, mechanism, scoped evidence, cost, falsifier, reproduction. Target ≥10/12 and no leakage/test-selection/invented-reproduction claim.
'''
    (ROOT/'reference'/f'{SLUG}.html').write_text(head+render(ref+f'\n[Lesson](../lessons/{SLUG}.html) · [Blank guide](../labs/l079-decision-template.md) · [Protocol](../labs/l079-reproduction.md)')+'</article></body></html>')
    r=json.loads((LAB/'_verify_l079_results.json').read_text());data={'all':r['summary']['random']['mean_ranks'],'matched':r['matched']['random']['mean_ranks'],'temporal':r['matched']['temporal']['mean_ranks']}
    js="RetrievalBank.mount(document.getElementById('warmup'),{upTo:79,count:3});\nDecisionGuideViz.budget(document.getElementById('budget-viz'));\nDecisionGuideViz.cohort(document.getElementById('cohort-viz'),"+json.dumps(data)+");\nTeachback.mount(document.getElementById('teachback'),{prompt:'Why can the best-ranked model be the wrong deployment choice?',points:['Deployment split and available information come first','The published or local population may differ','Task support and full-path serving cost constrain candidates','Validation selects; test remains locked'],model:'A mean rank applies only to its benchmark population and protocol. I first match information availability and the deployment split, then compare supported candidates within measured resource limits. Validation selects the feasible model; a frozen test evaluates it.'});\n"
    (ROOT/'assets/l079-lesson.js').write_text(js)
    print('Built L079 lesson, reference, student/solution notebooks, preview and figures')
    finalize(79)

if __name__=='__main__':build()
