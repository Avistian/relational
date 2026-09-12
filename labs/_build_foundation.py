"""Regenerate the thirteen authored lessons, references, notebooks and prepared HTML."""
import html,json,os,re
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _foundation_config import SLUGS,TITLES,PRIMARY,WIDGET,PREDICT,ARCH,TASKS,QUIZ
from _notebook_foundation import build as notebook
from _foundation_access import launcher,lab_plan,build_directory,build_static_gallery
from _lesson_depth import enrich_html

ROOT=Path(__file__).resolve().parents[1];LABS=ROOT/'labs'


def evidence(n):
    r=json.loads((LABS/f'_verify_l{n:03}_results.json').read_text())
    text='';table=''
    if n==58:
        a=r['mean_ranks'];b=r['xgb_favored_45_ranks'];text=f"The pinned tables contain {r['datasets']} complete task rows for this six-model panel. {min(a,key=a.get)} has the smallest full-panel mean rank ({min(a.values()):.3f}); outcome-selecting 45 tasks changes both the question and the ranking. Rounded published means do not recover seed-level observations."
        table=pd.DataFrame({'full_mean_rank':a,'selected_45_rank':b}).to_html(float_format=lambda v:f'{v:.3f}')
    elif n==59:
        first,last=r['summary'][0],r['summary'][-1]
        text=f"Searching 256 noise candidates reduces mean selected validation error from {first['validation_error']:.3f} to {last['validation_error']:.3f}, while mean independent test error is {last['test_error']:.3f}. The mean optimism is {last['optimism']:.3f}; its Monte Carlo standard error is {last['monte_carlo_se']:.4f}. This controlled result establishes selection bias without introducing predictive signal."
        table=pd.DataFrame(r['summary']).to_html(index=False,float_format=lambda v:f'{v:.4f}')
    elif n in [60,70]:
        lines=[]
        for regime,s in r['summary'].items():
            ranks=s['mean_ranks'];winner=min(ranks,key=ranks.get)
            lines.append(f"{regime.capitalize()}: {s['datasets']} task blocks; {winner} has the smallest mean rank ({ranks[winner]:.3f}); Friedman p={s['friedman_p']:.4f}, Nemenyi CD={s['nemenyi_cd']:.3f}.")
        text=' '.join(lines)+' These restricted panels and small budgets support a local baseline decision, not universal superiority. Intervals below describe model-seed variation on fixed rows.'
        details=[dict(regime=regime,**d) for regime,s in r['summary'].items() for d in s['details']]
        table=pd.DataFrame(details)[['regime','dataset','arm','mean','sd','seconds']].to_html(index=False,float_format=lambda v:f'{v:.4f}')
    elif n==61:
        a=[v['in_support_mae'] for v in r['runs']];b=[v['extrapolation_mae'] for v in r['runs']]
        text=f"Across three trained seeds, mean absolute posterior error averages {np.mean(a):.4f} on the checked in-range grids and {np.mean(b):.4f} at unseen context length 64. The plot shows each seed's range around its mean. The learned approximation is useful but does not inherit the analytic formula's exact extrapolation."
        table=pd.DataFrame([{'seed':v['seed'],'in_range_MAE':v['in_support_mae'],'length_64_MAE':v['extrapolation_mae']} for v in r['runs']]).to_html(index=False,float_format=lambda v:f'{v:.4f}')
    elif n==63:
        values=r['downstream_mean_absolute_change'];text=f"Removing the selected edge leaves both ancestors exactly unchanged. Mean absolute changes at nodes 2 and 3 are {values[2]:.4f} and {values[3]:.4f}, respectively, on the paired synthetic samples. This isolates the implemented mechanism; it does not identify a real table's causal graph."
    elif n==66:
        f=pd.DataFrame(r['records']);s=f.groupby('context')[['log_loss','predict_seconds']].mean()
        text=f"With identical 60 query rows, mean log loss is {s.loc[60,'log_loss']:.4f} at context 60 and {s.loc[540,'log_loss']:.4f} at context 540. Mean prediction time rises from {s.loc[60,'predict_seconds']:.3f}s to {s.loc[540,'predict_seconds']:.3f}s. One task and three nested context draws do not establish the paper's large-data benchmark claim."
        table=s.to_html(float_format=lambda v:f'{v:.4f}')
    elif n==68:
        f=pd.DataFrame(r['pfn_ablation']['records']);s=f.groupby(['arm','condition']).log_loss.agg(['mean','std'])
        text='The changing-edge prior lowers mean loss on the matched drifting task collection in this reduced experiment. Its stationary-task result must be inspected separately. The figure shows three model seeds; the 200 shared tasks are paired evaluation cases, not independent pretrained models. The full published temporal model and benchmark remain unrun.'
        table=s.to_html(float_format=lambda v:f'{v:.4f}')
    elif n==69:
        f=pd.DataFrame(r['records']);s=f.groupby(['dataset','arm','condition']).log_loss.agg(['mean','std'])
        text=f"The three-task corruption panel retains every test row and pairs each intervention with its clean baseline. The separate synthetic unseen-class fixture has all-row log loss {r['unseen_class_log_loss_example']:.3f} with epsilon 1e−12. It is a class-support diagnostic, not measured v2 open-set detection performance."
        table=s.to_html(float_format=lambda v:f'{v:.4f}')
    else:
        f=pd.DataFrame(r['records']);s=f.groupby(['dataset','arm']).error.agg(['mean','std'])
        if n==62:text=f"The actual historical v1 checkpoint ran on all three task/seed combinations. Maximum observed alone-versus-batched query probability difference is {r['query_batch_max_delta']:.2g}. This checks the measured inference path; it does not validate unseen preprocessing settings or reproduce the paper benchmark."
        elif n==64:text='These are the exact v2 and XGBoost predictions from the five-task checkpoint, reused for a focused historical-version comparison. They are not additional independent fits. No four-hour baseline tuning was performed.'
        elif n==65:text='The native v2 predictor has lower seed-mean loss than the final-layer cross-fitted linear head on all three tasks here. The head remains a valid representation experiment; this result does not reproduce the paper\'s selected intermediate-layer and 29-task comparison. A clear failed gain is preferable to asserting the paper\'s outcome for a different protocol.'
        elif n==67:text='Exact retrieval alone does not uniformly improve the global v1 baseline. Six-step local fine-tuning changes the actual pretrained parameters and has the best mean rank in this three-task, 24-test-row-per-task panel. The small test sample and conditional seeds make this exploratory; no broad adaptation superiority follows.'
        table=s.to_html(float_format=lambda v:f'{v:.4f}')
    return text,table


def figure(n,name,reference=False):
    caption={'mechanism':'Synthetic numerical trace. Values are generated by the displayed operator; this is not a measured benchmark.',
             'comparison':'Measured paired seed gaps and conditional 95% t intervals on fixed rows.',
             'ranks':'Dataset-block mean ranks and Nemenyi critical distance for the complete declared pool.',
             'results':'Author-reference measurements. See the protocol, seed points and interpretation; these are not your notebook kernel outputs.'}[name]
    return f'<figure class="foundation-figure"><div class="foundation-image-scroll"><img src="../labs/figures/l{n:03}/{name}.png" alt="{html.escape(caption)}" loading="lazy"></div><figcaption>{caption}</figcaption></figure>'


def architecture_html(n):
    return '<ol class="foundation-route">'+''.join(f'<li><strong>{html.escape(a)}</strong>{html.escape(b)}<small>{html.escape(c)}</small></li>' for a,b,c in ARCH[n])+'</ol>'


def contract(n):
    title,url=PRIMARY[n];slug=f'{n:04}-{SLUGS[n]}'
    text=f'''# L{n:03} reproduction and evidence contract

Primary source: [{title}]({url}). [Lesson](../lessons/{slug}.html).

## What ships

Student notebook, local executed teacher solution (ignored by Git under the course convention),
prepared student HTML, numerical mechanism/architecture/result figures, visible canonical code,
behavioral checks, source/checkpoint provenance and committed author evidence.

## Three separate claims

1. **Operator/architecture:** the specified local functions implement the named computation;
   reduced PFNs are not official checkpoint architectures. Read their explicit omissions.
2. **Measured evidence:** `_verify_l{n:03}_results.json` contains the actual local run or frozen-result
   reanalysis. Score outputs identify which implementation produced them.
3. **Paper results:** full original pretraining/benchmark replication is NOT_ESTABLISHED.
   No resource preset silently changes this verdict into MATCH.

## Regeneration

From the repository root, install `requirements-labs.txt`, then:

```bash
.venv/bin/python labs/_run_foundation.py --lesson {n} --preset lab --output labs/data/cache/foundation/l{n:03}-rerun.json
.venv/bin/python labs/_build_l{n:03}.py
```

The default notebook track needs no pretrained download. The explicit post-EXIT gate uses
an isolated package directory for historical packages; it may download immutable checkpoints.
Historical package/checkpoint hashes are in `_sources_foundation.json`; TALENT source tables
are pinned separately in `_sources_l058.json`. TabReD and public-table loaders retain their
earlier source/data identities. Do not treat a package version as a checkpoint identity.

The `smoke` preset is a small execution check. `closer` increases supported training budgets,
or reruns an already full frozen-result audit. Historical checkpoint experiments retain their
declared small protocol unless their runner explicitly says otherwise. `paper` deliberately
raises an error: the full heterogeneous paper protocols are not implemented.

For unattended execution, the supplied CPU operator is:

```bash
modal run --detach modal/foundation_repro.py --lesson {n} --preset closer
```

No Modal job or live Colab browser run is claimed by packaging this command.

## Evaluation boundaries

Read the lesson for exact dataset roster/caps, split seeds, model/inference seeds, candidates,
metric direction, context policy and omitted paper components. Compare methods on aligned
rows. Average model seeds inside datasets before ranking. Conditional seed intervals do not
cover dataset or temporal-split uncertainty. Frozen result tables are not new model fits.

The common local checkpoint records predictions, targets, selection traces and costs.
L064 explicitly reuses L070 predictions; do not count those as additional evidence.
Current-version arms have separate statuses; historical-v2 scores cannot stand in for v3.

## Delivery evidence

See `_execution_foundation_results.json`, `_source_check_foundation_results.json` and
`_delivery_foundation_results.json` for performed checks. Browser rendering, copied Pages
staging, live Colab and post-push deployment are independent checks. User mastery remains
unassessed until a completed EXIT and explanation are reviewed.
'''
    if n==70:
        text+='\n## Rebuild the complete seven-arm panel\n\nRun both environments, then join only matching row contracts:\n\n```bash\n.venv/bin/python labs/_run_foundation.py --lesson 70 --preset lab --output /tmp/l070-historical.json\n.venv/bin/python labs/_run_foundation.py --lesson 70 --preset lab --current --output /tmp/l070-current.json\n.venv/bin/python labs/_assemble_foundation_checkpoint.py --historical /tmp/l070-historical.json --current /tmp/l070-current.json --output /tmp/l070-complete.json\n```\n\nThe committed `_verify_l070_results.json` is this complete panel. Historical and current subpanel JSON files preserve their distinct environment identities. The 2.5 arm is the explicit synthetic-pretrained checkpoint.\n'
    (LABS/f'l{n:03}-reproduction.md').write_text(text)


def build(n,notebooks=True,render=True):
    if n == 69:
        from _build_l069 import build_package
        return build_package(notebooks=notebooks,render=render)
    if n == 68:
        from _build_l068 import build_package
        return build_package(notebooks=notebooks,render=render)
    if n == 67:
        from _build_l067 import build_package
        return build_package(notebooks=notebooks,render=render)
    if n == 64:
        from _build_l064 import build_package
        return build_package(notebooks=notebooks, render=render)
    if n == 66:
        from _build_l066 import build_package
        return build_package(notebooks=notebooks,render=render)
    if n == 65:
        from _build_l065 import build_package
        return build_package(notebooks=notebooks, render=render)
    if n == 63:
        from _build_l063 import build_package
        return build_package(notebooks=notebooks, render=render)
    if n == 62:
        from _build_l062 import build_package
        return build_package(notebooks=notebooks, render=render)
    if n == 61:
        from _build_l061 import build_package
        return build_package(notebooks=notebooks, render=render)
    if n == 60:
        from _build_l060 import build_package
        return build_package(notebooks=notebooks, render=render)
    if n == 59:
        from _build_l059 import build_package
        return build_package(notebooks=notebooks, render=render)
    if n == 58:
        from _build_l058 import build_package
        return build_package(notebooks=notebooks, render=render)
    slug=f'{n:04}-{SLUGS[n]}';manuscript=(ROOT/'lessons/content'/f'{slug}.md').read_text();text,table=evidence(n)
    cfg=dict(lesson=n,mode=WIDGET[n],prompt=PREDICT[n][0],answer=PREDICT[n][1],quiz=QUIZ[n])
    if n==59:cfg['selection']=json.loads((LABS/'_verify_l059_results.json').read_text())['summary']
    body=markdown2html_mistune(manuscript)
    for name in ['mechanism','architecture','results','comparison','ranks']:
        if name=='architecture':replacement=architecture_html(n) if n in ARCH else ''
        elif name=='mechanism':replacement=f'<div data-foundation-viz="{WIDGET[n]}"></div>'+figure(n,name)
        elif name in ['comparison','ranks']:replacement=figure(n,name)
        else:replacement=figure(n,name)+f'<div class="foundation-evidence"><p><strong>Measured interpretation.</strong> {html.escape(text)}</p><details><summary>Inspect the numerical evidence table</summary><div class="foundation-scroll">{table}</div></details></div>'
        body=body.replace(f'<!--figure:{name}-->',replacement)
    prev='0057-cross-family-ensembling' if n==58 else f'{n-1:04}-{SLUGS[n-1]}'
    nextlink=f' · <a href="{n+1:04}-{SLUGS[n+1]}.html">Lesson {n+1:03} →</a>' if n<70 else ''
    head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson {n:03} · {html.escape(TITLES[n])}</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/foundation-course.css"><link rel="stylesheet" href="../assets/lab-access.css"></head><body><article>'
    opening=f'<nav><a href="../index.html">Course</a> · <a href="{prev}.html">← Lesson {n-1:03}</a>{nextlink} · <a href="../reference/{slug}.html">Reference</a></nav><p class="mission-tag">Year 2 · Quarter {2 if n<=60 else 3} · Lesson {n:03}</p><h1>{TITLES[n]}</h1>{launcher(n)}<p class="subtitle">Understand the computation. Challenge the evidence. Produce a defensible result.</p><p>Core reading: 35–50 minutes. Lab: 45–75 minutes plus declared experiment time. This sequence builds the strong single-table baseline and information discipline required to test the relational thesis.</p><section id="retrieval"><h2>Retrieve before reading</h2><p>Without notes: {html.escape(PREDICT[n][0])} Explain why, then recall a related failure mode from an older lesson.</p><div id="warmup"></div></section><div id="prediction"></div>'
    closing=f'''<section id="lab"><h2>Run the companion lab</h2>{lab_plan(n)}<p><a href="../labs/{slug}.ipynb">Student notebook</a> · <a href="../labs/html/{slug}.html">Prepared lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{slug}.ipynb">Open in Colab</a></p><p>{len(TASKS[n])} substantive code tasks feed the live experiment, followed by behavioral CHECKs and a written EXIT artifact. The notebook contains the explanation and portable figures so it is teachable independently.</p><div class="foundation-contract"><strong>Keep the evidence boundaries visible.</strong> <a href="../labs/l{n:03}-reproduction.md">Reproduction contract</a> · <a href="../labs/_verify_l{n:03}_results.json">Measured evidence</a> · <a href="../labs/_sources_foundation.json">Source/checkpoint identities</a> · <a href="../labs/relkit/foundation_core.py">Visible model mechanisms</a>. These artifacts do not certify full paper reproduction or personal mastery.</div><div id="teachback"></div><p>Ask me follow-up questions about any operation, CHECK or conclusion you cannot defend. Paste the EXIT artifact and your explanation for review. Revisit the retrieval question tomorrow and again in a week, before reopening the reference.</p></section><nav><a href="../reference/{slug}.html">Printable reference</a> · <a href="../notebooks.html">Notebook gallery</a></nav></article><script id="foundation-config" type="application/json">{json.dumps(cfg).replace('<','\\u003c')}</script>'''
    scripts=''.join(f'<script src="../assets/{v}.js"></script>' for v in ['retrieval-pool','retrieval-bank','predict','teachback','foundation-viz','foundation-lesson'])
    (ROOT/'lessons'/f'{slug}.html').write_text(head+opening+body+closing+scripts+'</body></html>')
    refbody=f'<nav><a href="../lessons/{slug}.html">Lesson {n:03}</a> · <a href="../labs/html/{slug}.html">Open lab</a> · <a href="../labs/html/foundation-sequence.html">All 58–70 materials</a> · <a href="glossary.html">Glossary</a></nav><h1>{TITLES[n]}</h1><p class="subtitle">A computation and evidence reference</p><h2>Core distinction</h2><p>{html.escape(PREDICT[n][1])}</p><h2>Implementation contract</h2><table><thead><tr><th>Operator</th><th>Required behavior</th></tr></thead><tbody>'+''.join(f'<tr><td><code>{a}</code></td><td>{html.escape(d)}</td></tr>' for a,b,c,d in TASKS[n])+'</tbody></table>'+figure(n,'mechanism')+f'<h2>Measured evidence and limit</h2><p>{html.escape(text)}</p><h2>Before making a claim</h2><ol><li>Identify source, model/checkpoint and data versions.</li><li>Trace which labels enter each computation.</li><li>State what is fixed, varied and measured.</li><li>Use the right uncertainty and aggregation unit.</li><li>Name the unrun comparison and a falsifying experiment.</li></ol><p>Primary reading: <a href="{PRIMARY[n][1]}">{html.escape(PRIMARY[n][0])}</a>. <a href="../labs/l{n:03}-reproduction.md">Exact regeneration contract</a>.</p>'
    (ROOT/'reference'/f'{slug}.html').write_text(head.replace(f'Lesson {n:03}','Reference')+refbody+'</article></body></html>')
    enrich_html(n)
    contract(n)
    if notebooks:notebook(n);notebook(n,True)
    if render:
        nb=nbformat.read(LABS/f'{slug}.ipynb',as_version=4);page,_=HTMLExporter().from_notebook_node(nb)
        soup=BeautifulSoup(page,'html.parser')
        for tag in soup.find_all(['a','img'],href=True)+soup.find_all('img',src=True):
            key='href' if tag.has_attr('href') else 'src';value=tag[key]
            if value and not value.startswith(('#','data:','http:','https:','mailto:','javascript:')):
                path,sep,fragment=value.partition('#');tag[key]=os.path.relpath((LABS/path).resolve(),LABS/'html')+(sep+fragment if sep else '')
        # The preview must say where to run it before the notebook reading starts.
        soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'))
        first_task=next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1'))
        anchor=soup.new_tag('span',id='lab-exercises');first_task.insert_before(anchor)
        soup.body.insert(0,BeautifulSoup(launcher(n,prepared=True),'html.parser'))
        (LABS/'html'/f'{slug}.html').write_text(str(soup))
    build_directory();build_static_gallery()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--lesson',type=int);p.add_argument('--keep-notebooks',action='store_true');args=p.parse_args()
    for n in ([args.lesson] if args.lesson else range(58,71)):build(n,notebooks=not args.keep_notebooks)
