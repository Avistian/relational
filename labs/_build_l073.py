"""Build L073 from manuscript and canonical visible Python definitions."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _figures_l073 import build as figures
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
SLUG='0073-when-ssl-helps';TITLE='When SSL actually helps'
SOURCE=(ROOT/'relkit/ssl_regimes_l073.py').read_text();TREE=ast.parse(SOURCE)
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
OLD=(ROOT/'relkit/contrastive_l072.py').read_text()
for n in ast.parse(OLD).body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['SCARF','corrupt','draw_view','scarf_loss']:NODES[n.name]=ast.get_source_segment(OLD,n)
IMPORTS='\n'.join(ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)) and not (isinstance(n,ast.ImportFrom) and n.module=='relkit.contrastive_l072'))
CAPTIONS={'boundary':'Illustrative access ledger at 10% labels. Trace which rows can affect fitted parameters before test scoring.', 'nesting':'Illustrative IDs: the same permutation yields 2, 5 and 10 labeled rows. Larger budgets retain every earlier labeled row.', 'curves':'Author measurements. Curves are means; faint ribbons show sample SD over three paired split/model repetitions. All six recipes receive the same available label budget.', 'gains':'Author measurements. Small points are individual paired gains; bars are descriptive paired t95 intervals over three repetitions. They are neither simultaneous bands nor independent-dataset uncertainty.', 'ranks':'Author rank audit at 40% labels. Average seeds before ranking datasets equally. The critical difference is exploratory; only three selected datasets are represented.'}
CHECKS={
'nested_labels':"g=nested_labels(list(range(20)),[.1,.5,1.],73)\nassert list(map(len,g))==[2,10,20]\nassert set(g[0])<set(g[1])<set(g[2])\nassert g==nested_labels(list(range(20)),[.1,.5,1.],73)",
'paired_gains':"rr=[{'seed':2,'arm':'ssl','accuracy':.9},{'seed':1,'arm':'base','accuracy':.7},{'seed':1,'arm':'ssl','accuracy':.6},{'seed':2,'arm':'base','accuracy':.6}]\nassert np.allclose(paired_gains(rr,'ssl','base'),[-.1,.3])\nfor bad in [rr[:-1],rr+[rr[0]]]:\n    try:paired_gains(bad,'ssl','base')\n    except ValueError:pass\n    else:raise AssertionError('Reject incomplete or duplicated seed pairs')",
'crossing_brackets':"assert crossing_brackets([.1,.2,.4,1.],[.02,-.01,.03,-.01])==[[.1,.2],[.2,.4],[.4,1.]]\nassert crossing_brackets([.1,.2,1.],[.02,0.,-.01])==[]\nassert crossing_brackets([.1,.2,1.],[.02,.01,.01])==[]"}
GOALS={'nested_labels':'Return floor(f*n) prefixes of one seeded label-blind permutation for increasing fractions. Reject invalid fractions and duplicate IDs. Never read labels.', 'paired_gains':'For one dataset/fraction, join treatment and control by seed, sort seeds, and return accuracy differences. Reject missing and duplicate pairs.', 'crossing_brackets':'Return every adjacent fraction pair whose gains have strictly opposite signs. Check lengths, increasing fractions and finite gains. Keep exact ties out.'}

def evidence():
 r=json.loads((ROOT/'_verify_l073_results.json').read_text())
 text='**Author-reference primary gains: SCARF fine-tuned minus scratch, percentage points.** These are saved measurements, not your current kernel output.\n\n| Dataset | 10% | 20% | 40% | 70% | 100% |\n|---|---|---|---|---|---|\n'
 for d in r['config']['datasets']:
  ss=[next(s for s in r['contrasts'] if s['dataset']==d and s['fraction']==f and s['treatment']=='scarf_ft' and s['control']=='scratch') for f in r['config']['fractions']]
  text+='| '+d+' | '+' | '.join(f"{100*s['mean']:+.2f}" for s in ss)+' |\n'
 text+='\n**Observed strict adjacent primary brackets:** '+ '; '.join(c['dataset']+': '+(str(c['brackets']) if c['brackets'] else 'none')+', ties '+str(c['ties']) for c in r['crossings'] if c['comparison']=='scarf_ft - scratch')+'.\n'
 text+='\nNeither these brackets nor their direction establishes a population threshold. Inspect the bars and individual gains below.\n'
 return text

def manuscript(notebook=False):
 t=(REPO/'lessons/content'/f'{SLUG}.md').read_text()
 for name,caption in CAPTIONS.items():
  p=ROOT/'figures/l073'/f'{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if notebook else f'../labs/figures/l073/{name}.png'
  t=t.replace('<!--figure:'+name+'-->',f'<figure><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable {name} figure"><img src="{src}" alt="{caption}"></div><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>')
 t=t.replace('<!--results-->',evidence())
 if notebook:
  t=re.sub(r'<div id="[^"]+"[^>]*></div>','',t)
  t=t.replace('Then complete the spaced warm-up.','Keep your answers for tutor review.')
  t=t.replace('**Prediction before moving the control:**','**Recalculate on paper:**')
  t=t.replace('(0071-', '(../lessons/0071-').replace('(0072-','(../lessons/0072-')
 return t

def notebook(solution=False):
 cells=[]
 def md(x):cells.append(nbf.v4.new_markdown_cell(x))
 def code(x):cells.append(nbf.v4.new_code_cell(x))
 md('# Lab 073 · When SSL actually helps\n\nImplement the experiment before interpreting the curve. Three TODOs feed fresh execution. CPU protocol: 3 real datasets × 3 paired seeds × 5 budgets × 6 arms. Fixed 40 pretraining / 60 supervised epochs. Original benchmark reproduction: INCOMPARABLE. Read the full self-contained explanation below, then implement the three tasks.\n\n[Lesson](../lessons/'+SLUG+'.html) · [Contract](../labs/l073-reproduction.md)')
 for c in bootstrap_cells():cells.append(nbf.from_dict(c))
 for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
  if part.strip():md(part)
 md('<a id="lab-exercises"></a>\n## Implement the experimental design\n\nPROVIDED cells expose all model and training code. The printed author tables above are reference evidence; your run below generates its own results.')
 code(IMPORTS+'\nimport json\nfrom pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\ntorch.set_num_threads(1)')
 for name,goal in GOALS.items():
  md('### TODO · '+name+'\n\n'+goal+'\n\nPredict the arithmetic fixture before running CHECK. Explain which bias a careless implementation would introduce.')
  code(NODES[name] if solution else NODES[name].split('\n')[0]+'\n    raise NotImplementedError("'+name+'")')
  code('# CHECK\n'+CHECKS[name]+'\nprint("'+name+': passed")')
 for names,title in [(['corrupt','scarf_loss','draw_view','SCARF'],'SCARF carried forward from Lesson 72: follow the donor columns, N-way loss and encoder/projector boundary'),(['load_regime','initialize'],'Feature access and paired initialization: identify every fit-row set'),(['predict_probe','predict_tuned'],'Frozen versus trainable encoders: find the reset and the validation-only selection'),(['summarize_regimes','run_regimes'],'The runner: find the nested subsets, seed join and crossing functions you implemented')]:
  md('### PROVIDED · '+title)
  for name in names:code(NODES[name])
 md('### RUN · commit predictions first\n\nWrite the sign of the primary gain at 10% and 100% for each dataset. Then run. The experiment typically takes a few CPU minutes; runtime depends on hardware.')
 code("assert run_regimes.__globals__['nested_labels'] is nested_labels\nassert summarize_regimes.__globals__['paired_gains'] is paired_gains\nassert summarize_regimes.__globals__['crossing_brackets'] is crossing_brackets\nresult=run_regimes()\nPath('l073-student-results.json').write_text(json.dumps(result,indent=2))\ndisplay(pd.DataFrame(result['summary'])[['dataset','fraction','arm','mean','sd']])\ndisplay(pd.DataFrame(result['crossings']))")
 code("# Plot YOUR fresh paired gains\nfig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)\nfor ax,d in zip(axes,result['config']['datasets']):\n    ss=[c for c in result['contrasts'] if c['dataset']==d and c['treatment']=='scarf_ft' and c['control']=='scratch']\n    for c in ss:\n        ax.scatter([100*c['fraction']]*len(c['gains']),100*np.array(c['gains']),s=18)\n    ax.plot([100*c['fraction'] for c in ss],[100*c['mean'] for c in ss],'k.-')\n    ax.axhline(0,color='gray');ax.set(title=d,xlabel='Training labels (%)')\naxes[0].set_ylabel('SCARF FT − scratch (percentage points)')\nplt.tight_layout();plt.show()")
 md('### EXIT · measurements plus argument\n\nSubmit your curves, paired gains, label counts, ties and brackets. Explain one failed prediction in 150–250 words; include a practical comparator, interval limits, validation-label cost and one controlled follow-up. No crossing is an acceptable finding.')
 code("assert len(result['records'])==270\nfor s in result['splits']:\n    tr,va,te=map(set,[s['train'],s['validation'],s['test']])\n    assert not tr&va and not tr&te and not va&te\n    previous=set()\n    for g in s['labeled_groups']:\n        assert previous<=set(g)<=tr\n        previous=set(g)\nfor r in result['records']:\n    assert abs(np.mean(np.array(r['prediction'])==r['target'])-r['accuracy'])<1e-12\n    assert r['total_development_labels']==r['train_labels']+r['validation_labels']\n    if r['arm'] in ['scratch','scarf_ft']:assert r['encoder_delta']>0\nprint('Structural EXIT passed. Written explanation needs tutor review.')")
 md('**Write your interpretation here.** Tomorrow, reconstruct the evaluation boundary from memory.')
 md('### Optional teaching extension · predeclared convergence audit\n\nHold datasets, budgets and method choices fixed; run five seeds with 200 pretraining and supervised epochs using your live functions. This changes the optimization budget and repetition count, not benchmark fidelity. Use fresh evaluation data before turning a test-derived crossover into a deployment decision. The larger run is gated off by default.')
 code("RUN_LONGER=False\nif RUN_LONGER:\n    longer=run_regimes(seeds=tuple(range(5)),pre_epochs=200,fine_epochs=200)\n    Path('l073-longer-results.json').write_text(json.dumps(longer,indent=2))\nelse:\n    print('Longer run NOT_RUN; original paper benchmarks INCOMPARABLE; live Colab NOT_CHECKED.')")
 from _paper_tracks_071_074 import paper_cells
 cells.extend(paper_cells(73))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(nb.cells):c.id=f'l073-{i:03d}'
 return nb

def build():
 figures()
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 073 — '+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/contrastive-views.css"></head><body><article>'
 nav=f'<nav><a href="../index.html">Course</a> · <a href="0072-scarf-subtab-contrastive-views.html">Previous lesson</a></nav><header><p>Year 2 · Quarter 4 · Lesson 073</p><h1>{TITLE}</h1><p>Match the comparison → account for labels → measure the curve → audit the crossover.</p></header><aside class="lab-access"><nav><a href="../labs/html/{SLUG}.html">Lab preview</a><a href="../labs/{SLUG}.ipynb">Student notebook</a><a href="../reference/{SLUG}.html">Reference</a><a href="../labs/_verify_l073_results.json">Measured evidence</a><a href="../labs/l073-reproduction.md">Reproduction contract</a></nav><p>Local package. Live Colab and deployment NOT_CHECKED.</p></aside>'
 body=render(manuscript()).replace('<table>','<div class="result-scroll"><table>').replace('</table>','</table></div>')
 scripts=''.join(f'<script src="../assets/{n}.js"></script>' for n in ['retrieval-pool','retrieval-bank','predict','teachback','label-budget-viz','l073-lesson'])
 (REPO/'lessons'/f'{SLUG}.html').write_text(head+nav+body+'<div id="teachback"></div></article>'+scripts+'</body></html>')
 for sol in [False,True]:
  p=ROOT/('solutions' if sol else '')/f'{SLUG}.ipynb';nb=notebook(sol)
  if sol and p.exists():
   old=nbf.read(p,as_version=4)
   if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in nb.cells if c.cell_type=='code']:
    codes=iter(c for c in old.cells if c.cell_type=='code')
    for c in nb.cells:
     if c.cell_type=='code':o=next(codes);c.outputs=o.outputs;c.execution_count=o.execution_count
  nbf.write(nb,p)
 preview,_=HTMLExporter().from_notebook_node(notebook());soup=BeautifulSoup(preview,'html.parser')
 for a in soup.find_all('a',href=True):
  if a['href'].startswith('../'):a['href']='../'+a['href']
 (ROOT/'html'/f'{SLUG}.html').write_text('\n'.join(line.rstrip() for line in str(soup).splitlines())+'\n')
 ref='''# Reference · SSL label-budget audits

**Define help:** fine-tuned SSL minus matched scratch tests pretraining; frozen SSL minus random features tests fixed representation utility; raw logistic and trees test practical competitiveness.

**Boundary:** fit preprocessing and pretraining on training features only. Supervise on the labeled prefix. Select on validation. Freeze choices before test. Count validation labels separately from training labels.

**Nested labels:** one label-blind permutation; prefix size floor(f × n_train). Larger budgets retain earlier rows. Never repair a missing class by quietly searching seeds.

**Worked cost:** 700 train / 100 validation / 200 test; f=.1 gives 70+100=170 development labels, 21.25% of development rows. Test annotations are a separate evaluation expense.

**Pairing:** join by seed within dataset and budget. Subtract first, average second. Reject missing and duplicate pairs. Accuracy differences ×100 are percentage points.

**Crossover:** report all adjacent strict mean-sign reversals; report exact ties separately. No observed bracket does not establish no population crossing. Never select on a test-derived threshold and reuse that test as confirmation.

**Uncertainty:** three overlapping split/model repetitions give descriptive variation, not independent-dataset or simultaneous interval guarantees. Rank datasets after averaging their seeds; do not count budgets as datasets.

**Diagnosis:** pretraining changes compute as well as initialization; frozen and fine-tuned comparisons can disagree. Vary unlabeled quantity, shift, corruption and training length separately in follow-ups.

**Sources:** [SCARF](https://arxiv.org/html/2106.15147v2), [tabular SSL survey](https://arxiv.org/html/2402.01204v3), [evaluation recommendations](https://arxiv.org/html/1804.09170v2).
'''
 (REPO/'reference'/f'{SLUG}.html').write_text(head+render(ref+f'\n[Full lesson](../lessons/{SLUG}.html) · [Lab](../labs/html/{SLUG}.html)')+'</article></body></html>')
 print('Built L073 lesson, reference, notebooks, preview and five figures')
if __name__=='__main__':build()
