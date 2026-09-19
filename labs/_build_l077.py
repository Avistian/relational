"""Build L077 from prose and canonical implementation; retain solution outputs."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _figures_l077 import build as figures
LAB=Path(__file__).resolve().parent;ROOT=LAB.parent;SLUG='0077-single-table-ceiling';TITLE='Single-table ceiling: an information audit'
SOURCE=(LAB/'relkit/ceiling_l077.py').read_text();TREE=ast.parse(SOURCE)
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,ast.FunctionDef)}
IMPORTS='\n'.join(ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)))
CAPTIONS={
'collision':'Worked example: the same three eligible amounts in opposite time order. Count, sum, mean and maximum coincide; last-minus-first has opposite signs.',
'bound':'Four-example illustration, distinct from the balanced experiment. Each collision class gets one prediction: majority counts 2 and 1 give 75% overall.',
'reach':'Two illustrative database worlds. Own features and order amounts agree; merchant risk differs two edges away. This shows information access, without claiming every graph reducer preserves the signal.',
'results':'Author-reference measurement: all five complete paired-history runs. Flat test accuracy 0.5; restored test accuracy 1.0. These are synthetic experiments under one generator.'}
GOALS={
'flat_rows':'Construct five columns in customer order: own, count, sum, mean and maximum of eligible amounts. Use the provided identity/eligibility helper. Empty-history aggregates are zero. IDs and labels must stay outside the feature vector.',
'collision_ceiling':'Partition examples by exactly equal feature vectors and return the best fraction of correct labels possible with one prediction per vector. Support arbitrary class proportions. Reject empty or misaligned inputs and nonfinite features.',
 'temporal_delta':'Return last-minus-first eligible amount for each customer, using event time rather than physical row order. Empty and singleton histories give zero. Reject tied eligible event times because the fixture declares no tie-breaking order.'}
CHECKS={
'flat_rows':'''c=[dict(customer_id=9,own=4),dict(customer_id=2,own=7)]
e=[dict(customer_id=9,event_day=3,available_day=3,amount=8),dict(customer_id=9,event_day=1,available_day=1,amount=2),dict(customer_id=9,event_day=2,available_day=11,amount=99)]
assert np.array_equal(flat_rows(c,e,10),[[4,2,10,5,8],[7,0,0,0,0]]), 'Check eligibility, row order and empty history'
assert np.array_equal(flat_rows(c[::-1],e[::-1],10),[[7,0,0,0,0],[4,2,10,5,8]]), 'Route by identity'
print('CHECK: eligible aggregation and empty histories pass')''',
'collision_ceiling':'''assert collision_ceiling([[1],[1]],[0,1])==.5, 'Identical inputs need one prediction'
assert collision_ceiling([[1],[1],[1],[2]],[0,0,1,1])==.75, 'Weight by examples, not classes'
assert collision_ceiling([[1],[2]],[0,1])==1., 'Unique rows give an empirical ceiling of one'
for xx,yy in [([],[]),([[0]],[0,1]),([[float('nan')]],[0])]:
    try: collision_ceiling(xx,yy)
    except ValueError: pass
    else: raise AssertionError('Reject undefined input')
print('CHECK: exact majority-class ceiling passes')''',
 'temporal_delta':'''assert temporal_delta(c,e,10).tolist()==[6,0], 'Sort eligible events by event time'
assert temporal_delta(c,e[::-1],10).tolist()==[6,0], 'Storage order has no meaning'
tied=e+[dict(customer_id=9,event_day=1,available_day=1,amount=5)]
try: temporal_delta(c,tied,10)
except ValueError: pass
else: raise AssertionError('Tied eligible times need a declared rule')
print('CHECK: time order, availability and ambiguity pass')'''}

def manuscript(notebook=False):
    t=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
    runs=json.loads((LAB/'_verify_l077_results.json').read_text())['runs']
    rows='\n'.join(f"| {r['seed']} | {r['scores']['flat']['test']:.3f} | {r['scores']['restored']['test']:.3f} | {r['ceilings']['flat']:.3f} / {r['ceilings']['restored']:.3f} |" for r in runs)
    table='**Author-reference evidence — measured locally; not output from your current kernel.**\n\n| Seed | Flat test accuracy | Restored test accuracy | Exact ceilings, flat / restored |\n|---|---|---|---|\n'+rows+'\n| Mean ± sample SD | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.500 / 1.000 |\n\nSQL feature oracle, future/late-arrival invariance, pair isolation, storage-order invariance and independent depth-three tree control: **PASS**.\n'
    t=t.replace('<!--results-->',table)
    for name,caption in CAPTIONS.items():
        src='data:image/png;base64,'+base64.b64encode((LAB/'figures/l077'/f'{name}.png').read_bytes()).decode() if notebook else f'../labs/figures/l077/{name}.png'
        t=t.replace('<!--figure:'+name+'-->',f'<figure class="ceiling-figure"><div tabindex="0" role="region" aria-label="Scrollable {name} figure"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
    if notebook:
        t=re.sub(r'<div id="[^"]+"></div>','',t)
        t=re.sub(r'\((00\d[^)]*\.html)\)',r'(../lessons/\1)',t)
    return t

def notebook(solution=False):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# Lab077 · '+TITLE+'\n\n**Learning contract:** build and certify an aggregation collision, then repair it while holding the learner fixed. Tier C synthetic data is required to prove the finite-support claim. Full reproduction: 1,000 pairs × five seeds, executed after EXIT; no scaled-down benchmark substitute.\n\n[Lesson](../lessons/'+SLUG+'.html) · [Reproduction protocol](../labs/l077-reproduction.md)')
    cells.extend(nbf.from_dict(c) for c in bootstrap_cells())
    for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
        if part.strip():md(part)
    md('<a id="lab-exercises"></a>\n## Implement and certify the information interface\n\nThe next cells show the complete generator, split, eligibility filter, model and training loop. Your three TODO functions remain live in `run_experiment`. The notebook imports no hidden experiment model. Work in `labs/` locally; the bootstrap selects it on Colab.')
    code(IMPORTS+'\nfrom pathlib import Path')
    for name,why in [('make_database','The target describes eligible history direction. Random metadata never enters a predictor. Each excluded record deliberately probes a time boundary.'),('split_pairs','Pairs are grouping units. Keeping each pair together preserves exact conditional balance in every partition.'),('eligible_histories','This is the shared identity and time boundary. FKs must refer to known PKs; both event and availability times are checked.')]:
        md('### PROVIDED · '+name+'\n\n'+why);code(NODES[name])
    for name in GOALS:
        md('### TODO · '+name+'\n\n'+GOALS[name]+'\n\nPredict the hand fixture output before running the CHECK.')
        code(NODES[name] if solution else NODES[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
        code('# CHECK — do not edit\n'+CHECKS[name])
    for name,why in [('fit_stump','Every threshold is derived from training values. Both orientations are tested; ties use the first candidate. This is the entire training procedure.'),('predict_stump','One threshold comparison per row. The function sees only a model and its supplied feature matrix.'),('run_experiment','The runner resolves your three helpers above. It fits on training, then reports validation/test scores. The ceiling audit consumes test labels only after fitting.')]:
        md('### PROVIDED · '+name+'\n\n'+why);code(NODES[name])
    md('### CHECK · a prediction-time intervention\n\nPredict the effect of replacing every future or late-arriving amount with a million. Then reverse physical storage order. Both representations must be unchanged after accounting for customer order.')
    code('''customers,events=make_database(30,77)
flat=flat_rows(customers,events,10)
delta=temporal_delta(customers,events,10)
changed=[dict(e,amount=1e6) if e['event_day']>10 or e['available_day']>10 else dict(e) for e in events]
assert np.array_equal(flat,flat_rows(customers,changed,10)), 'Excluded events changed flat features'
assert np.array_equal(delta,temporal_delta(customers,changed,10)), 'Excluded events changed delta'
assert np.array_equal(delta[::-1],temporal_delta(customers[::-1],events[::-1],10)), 'Identity or order bug'
train,valid,test=split_pairs(customers,77)
pair_sets=[{customers[i]['pair_id'] for i in part} for part in (train,valid,test)]
assert not(pair_sets[0]&pair_sets[1] or pair_sets[0]&pair_sets[2] or pair_sets[1]&pair_sets[2])
y=np.array([c['target'] for c in customers])
for part in (train,valid,test): assert collision_ceiling(flat[part],y[part])==.5
print('CHECK: invariance and pair-group isolation pass')''')
    md('### EXIT · your ceiling certificate\n\nWrite 150–250 words: name the exact feature interface, exhibit opposite-label histories, derive the bound, describe a permissible repair, and propose a real-data test that could weaken the broader thesis. Then construct another collision preserving count and sum. Explain whether delta repairs your new target. Send this text with the JSON below for grading; code execution alone does not grade the explanation.')
    md('## NEXT STEP · reproduce the complete declared experiment\n\nRun all five seeds below. This is the same 1,000-pair protocol as the author reference, with no downloads, GPU or additional model package. Train/validation/test sizes are 1,200/400/400 per seed. Every displayed result below comes from your live functions. Allow roughly a minute on a small CPU. The tolerance is exact accuracy equality: 0.5 flat and 1.0 restored.')
    code('''runs=[run_experiment(1000,seed) for seed in range(5)]
print('seed | flat test | restored test | flat bound | restored bound')
for r in runs:
    print(f"{r['seed']:4d} | {r['scores']['flat']['test']:.3f}     | {r['scores']['restored']['test']:.3f}         | {r['ceilings']['flat']:.3f}      | {r['ceilings']['restored']:.3f}")
    assert r['scores']['flat']['test']==r['ceilings']['flat']==.5
    assert r['scores']['restored']['test']==r['ceilings']['restored']==1.
Path('l077-student-results.json').write_text(json.dumps(runs,indent=2)+'\\n')
print('MATCH: complete declared synthetic experiment; no published benchmark reproduction claimed.')''')
    md('### Conclusion ledger\n\n| Bucket | Status | What this establishes |\n|---|---|---|\n| Local construction | Recompute above | Exact information ceiling and successful feature repair under the declared generator |\n| Published RelBench results | Cited, NOT_RUN here | Real-data predictive comparisons require their own protocol |\n| Larger synthetic run | Full declared run above | All 1,000 pairs per seed; increasing this count cannot establish real-world prevalence |\n\nFor the real-data sequel, use [L076\'s complete historical model/trainer notebook](../labs/html/0076-encoder-predictor-stack.html) and its [pinned replay contract](../labs/l076-reproduction.md), which include local, Colab and Modal commands. This synthesis introduces no replacement model and does not relabel that unrun benchmark as reproduced.\n\n**Spaced retrieval:** tomorrow, derive the majority-class bound without opening this notebook. One week later, give a counterexample to the claim that a graph always preserves cardinality.')
    for i,c in enumerate(cells): c['id']=f'l077-{i:03d}'
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})

def build():
    from _walkthrough_delivery import snapshot, finalize
    snapshot(77)
    figures()
    head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson077 — '+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/ceiling-viz.css"><link rel="stylesheet" href="../assets/flatten-loss-viz.css"></head><body><article>'
    nav=f'<nav><a href="../index.html">Course</a> · <a href="0076-encoder-predictor-stack.html">Lesson76</a></nav><header><p>Year2 · Quarter4 · Lesson077</p><h1>{TITLE}</h1><p>Exhibit a collision. Prove its bound. Restore the missing information.</p></header><aside><a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/{SLUG}.ipynb" download>Download notebook</a> · <a href="../reference/{SLUG}.html">Reference</a> · <a href="../labs/l077-reproduction.md">Reproduction contract</a></aside>'
    body=render(manuscript()).replace('<table>','<div class="ceiling-table"><table>').replace('</table>','</table></div>')
    scripts=''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','flatten-loss-viz','ceiling-viz','l077-lesson'])
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
    ref='''# Single-table ceiling · audit card

**Representation:** the exact information a predictor receives. State whether IDs, other rows, history, time and learned context are accessible.

**Collision:** different histories share a vector. A deterministic predictor gives them one shared output.

**Exact empirical ceiling:** sum the majority-label count within each equal-vector group; divide by total examples. This is an information audit using labels after fitting, not a deployable classifier.

**Balanced paired construction:** [10,30,50] and [50,30,10] both give count3, sum90, mean30, max50, with opposite direction labels. The ceiling is 0.5. Last-minus-first gives +40 and −40; adding it permits 1.0.

**Qualification:** random predictions have the same expected bound. Unique IDs make the finite-sample bound vacuous. Approximate float equality requires its own declared tolerance. These fixtures use integer-valued, exact aggregates.

**Time:** event time and availability time must both precede or equal cutoff10. Empty histories map to zero aggregates/delta; tied eligible times are rejected.

**Reproduction:** 1000 pairs, 5 seeds, disjoint pair-group 60/20/20 split; visible exhaustive stump training; exact bound; independent SQL and sklearn control. Synthetic evidence only.

**Four questions:** What is discarded? Can a permitted feature restore it? Does the predictor exploit it? Does it remain useful on new entities or time periods at acceptable cost?

**Reduction traps:** mean loses multiplicity; count+sum can lose distribution; one hop cannot read a two-hop attribute in the declared graph; relational access alone does not enforce joint constraints.

**Research boundary:** evidence of an information gap is not evidence that learned relational features outperform adequate manual features. Measure that separately.

[Primary motivation](https://proceedings.mlr.press/v235/fey24a.html) · [Real-data study](https://arxiv.org/html/2407.20060v1)
'''
    (ROOT/'reference'/f'{SLUG}.html').write_text(head+render(ref+f'\n[Full lesson](../lessons/{SLUG}.html) · [Lab](../labs/html/{SLUG}.html)')+'</article></body></html>')
    print('Built lesson, student/solution notebooks, preview, reference and four portable figures')
    finalize(77)
if __name__=='__main__':build()
