"""Build B19 lesson, reference, portable notebooks and explanatory figures."""
import ast,base64,gzip,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent;S='b19-benchmark-evidence';F=P/'figures/b19'
r=json.loads((P/'evidence/b19/diagnostic.json').read_text());audit=json.loads((P/'evidence/b19/source-audit.json').read_text())
(R/'assets/b19-evidence.js').write_text('/* Generated complete course evidence; no paper scores. */\nwindow.B19_EVIDENCE='+json.dumps(r,separators=(',',':'))+';\n')
plt.rcParams.update({'font.size':12,'figure.facecolor':'#fafcf9','axes.facecolor':'#fafcf9','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(6.4,6.7),sharex=True)
for ax,regime in zip(axes,['random','grouped']):
 for model,offset,color,label in [('group_memory',-.14,'#347f67','Group memory'),('signal',.14,'#ae672e','Signal')]:
  vals=[a['brier'] for a in r['arms'] if a['regime']==regime and a['model']==model]
  ax.bar([s+offset for s in range(3)],vals,width=.26,color=color,label=label)
  for seed,v in enumerate(vals):ax.text(seed+offset,v+.008,f'{v:.3f}',ha='center',fontsize=10)
 ax.set_ylim(0,.35);ax.set_ylabel('Brier loss ↓');ax.set_title(('Familiar groups · 48 test rows' if regime=='random' else 'Unseen groups · 64 test rows'),loc='left',weight='bold');ax.set_xticks([0,1,2],['Seed 0','Seed 1','Seed 2']);ax.grid(axis='y',alpha=.2)
axes[0].legend(loc='upper left',fontsize=10,ncol=2);fig.tight_layout();fig.savefig(F/'split.png',dpi=145);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(6.4,5.7),sharex=True)
for ax,policy,title in zip(axes,['measured','rf'],['Common measured support · 3 datasets','Explicit RF substitution · 4 datasets']):
 means=r['policies'][policy]['means'];ax.barh(['A','B','RF'],[means[x] for x in ['A','B','RF']],color=['#347f67','#55799b','#ae672e']);ax.invert_yaxis();ax.set_xlim(0,.50);ax.set_title(title,loc='left',fontsize=12,weight='bold')
 for i,m in enumerate(['A','B','RF']):ax.text(means[m]+.01,i,f'{means[m]:.3f}',va='center')
axes[1].set_xlabel('Mean constructed loss ↓');fig.tight_layout();fig.savefig(F/'missing.png',dpi=145);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(6.4,6))
for i,d in enumerate(['a','b','c']):
 vals=[x['delta'] for x in r['deltas'] if x['dataset']==d];axes[0].scatter([i]*3,vals,color='#347f67');axes[0].plot([i-.2,i+.2],[sum(vals)/3]*2,color='#ac622c',lw=3)
axes[0].axhline(0,color='#777',ls='--');axes[0].set_xticks([0,1,2],['Dataset a','Dataset b','Dataset c']);axes[0].set_ylabel('A−B loss');axes[0].set_title('Average paired seeds within each dataset',loc='left',fontsize=12,weight='bold')
for name,values,color in [('Dataset SE',[r['uncertainty']['se_dataset'],r['duplicated_seeds']['se_dataset']],'#347f67'),('Naive row SE',[r['uncertainty']['se_naive_rows'],r['duplicated_seeds']['se_naive_rows']],'#ac622c')]:axes[1].plot([9,18],values,'o-',label=name,color=color)
axes[1].set_xticks([9,18],['9 original rows','18 duplicated rows']);axes[1].set_ylim(0,.055);axes[1].set_ylabel('Standard error');axes[1].legend(fontsize=10);axes[1].set_title('Duplication adds no independent dataset',loc='left',fontsize=12,weight='bold');fig.tight_layout();fig.savefig(F/'uncertainty.png',dpi=145);plt.close(fig)

def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b19/'+name+'.png'
 return f'<figure class="evidence-figure" tabindex="0" role="region" aria-label="{caption}" style="overflow-x:auto"><img src="{src}" alt="{caption}" style="min-width:560px;max-width:100%;height:auto"><figcaption>{caption} On narrow screens, scroll the figure horizontally.</figcaption></figure>'

def board(kind,title,controls,extra=''):
 return f'<section class="evidence-board" data-{kind}-board><h3>{title}</h3><div class="evidence-controls">{controls}<button type="button">Reset</button></div>{extra}<output aria-live="polite">Enable JavaScript to explore saved measurements. The adjacent table and figure remain available without it.</output><noscript><p>Read the complete static worked example below.</p></noscript></section>'

def select(name,label,values,default):
 return '<label>'+label+f' <select name="{name}">'+''.join(f'<option value="{k}"'+(' selected' if k==default else '')+'>'+v+'</option>' for k,v in values)+'</select></label>'
split=board('split','Who is familiar at test time?',select('seed','Seed',[(str(i),str(i)) for i in range(3)],'0')+select('regime','Test population',[('random','Familiar groups'),('grouped','Unseen groups')],'grouped'),'<div class="evidence-groups" aria-label="24 groups; train and test row counts"></div>')
missing=board('missing','Does the declared policy change the winner?',select('policy','Missing-result policy',[('measured','Common measured support'),('rf','Explicit RF substitution')],'measured'))
uncertainty=board('uncertainty','Add rows without adding information',select('replicas','Seed records',[('1','Original nine rows'),('2','Duplicate to eighteen rows')],'1'))
pipe='''<section class="evidence-flow" aria-label="Evaluation computation"><div class="evidence-step"><strong>1 · Declare deployment</strong>Will the next customer be familiar or unseen? This fixes the test population.</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step"><strong>2 · Split before fitting</strong>Outer training → inner selection and preprocessing. Outer test labels remain outside both.</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step"><strong>3 · Produce keyed evidence</strong>Train-only predictor + legal test features → (dataset, split, row ID, probability). Join labels only for scoring.</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step"><strong>4 · State what was measured</strong>Record measured / missing / substituted cells. Pair comparable support, average seeds within dataset, then summarize datasets.</div></section>'''
version='''<section class="evidence-flow" aria-label="Versioned evidence"><div class="evidence-step"><strong>Original paper target</strong>v1 Figure F.2 · original six families · grouped and IID · original full split/variant set</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step"><strong>Later release</strong>September Linear rerun · newer TFM entries · current default core subset. Record these as changes.</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step"><strong>Comparison decision</strong>Match all relevant identities for a historical claim, or label a new comparison. A familiar method name cannot bridge changed protocols.</div></section>'''
split_table='| Seed | Random: memory | Random: signal | Grouped: memory | Grouped: signal |\n|---:|---:|---:|---:|---:|\n'
for seed in range(3):
 vals=[next(a['brier'] for a in r['arms'] if a['seed']==seed and a['regime']==regime and a['model']==m) for regime in ['random','grouped'] for m in ['group_memory','signal']]
 split_table+='| '+str(seed)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |\n'
paper_table='| Evidence item | B19 status |\n|---|---|\n| Complete course diagnostic | 12 arms, 672 predictions independently reconstructed |\n| Released grouped scores | 9,540 records; twelve default aggregates checked |\n| Full Figure F.2 | INCOMPLETE_SOURCE_PROTOCOL_GATE / NOT_RUN |\n| Fresh benchmark / pretraining | NOT_RUN |\n| Learner defense / live Colab | PENDING_WRITTEN_DEFENSE / NOT_CHECKED |\n'
matrix='''| Source or lane | Task/split and exposure | Selection and missing scores | Access / comparability |
|---|---|---|---|
| BeyondArena v1 | Mixed IID/grouped/temporal; audit each model's training exposure | Traditional search vs TFM ICL; specified RF fallbacks | Public source/results; historical F.2 IID scores not recovered here |
| Current pinned BeyondArena registry | Original and newer suites coexist; core differs from all splits | Linear rerun replaces default registry entry | Version change must be recorded before comparing ranks |
| TabDPT v3 | Real-data pretraining; Appendix B.1 overlap screening | Follow its own split, retrieval and selection protocol | Screening evidence is not proof of zero contamination |
| Enterprise study v1 | Selected internal business tasks; audit curation and semantic types | Nontrivial/signal filters affect target population | Internal data prevent an independent B19 reproduction; SAP affiliation disclosed |
| B19 synthetic diagnostic | Known generator; grouped/random populations; no pretraining | Fixed rules, all conditions, separate imputation fixture | Full local reconstruction; no real-data benchmark ranking |
'''
figs={'SPLIT_FIG':('split','Measured course diagnostic: shared groups yield perfect memory; unseen-group results vary by seed.'),'MISSING_FIG':('missing','Constructed losses: common measured support favors A; explicit RF substitution favors B.'),'UNCERTAINTY_FIG':('uncertainty','Constructed paired differences: seed duplication shrinks only the naive row-based uncertainty.')}
replacements={'PIPELINE':pipe,'SPLIT_WIDGET':split,'MISSING_WIDGET':missing,'UNCERTAINTY_WIDGET':uncertainty,'VERSION_FLOW':version,'SPLIT_RESULTS':split_table,'MATRIX':matrix,'PAPER_RESULTS':paper_table}
body=(R/'lessons/content'/f'{S}.md').read_text()
def fill(text,portable=False):
 for key,val in replacements.items():
  if portable and key.endswith('WIDGET'):val='**Predict before running:** trace the worked example, then compare your notebook outputs with this author reference.'
  text=text.replace('{{'+key+'}}',val)
 for key,(name,caption) in figs.items():text=text.replace('{{'+key+'}}',figure(name,caption,portable))
 return text

def document(title,text,interactive=False):
 html=render(text).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable evidence table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','b19-evidence','benchmark-evidence'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/benchmark-evidence.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B19 · Benchmark evidence',fill(body),True))
ref='''# B19 · Benchmark evidence reference

[Lesson](../lessons/b19-benchmark-evidence.html) · [Notebook](../labs/b19-benchmark-evidence.ipynb) · [Full contract](../labs/b19-reproduction.md)

| Question | Required evidence |
|---|---|
| What will be unfamiliar? | Declare familiar entity, unseen entity, future period, or joint restriction |
| Is outer-test separation enough? | Inner selection and preprocessing must respect intended deployment too |
| Did pretraining see the test? | Audit manifests, exact/derived overlaps; record UNKNOWN when unestablished |
| Which score was measured? | Preserve measured / missing / imputed origin per dataset, split and method |
| What is averaged? | Common paired support, declared metric normalization and equal-dataset weighting |
| How many independent units? | Dataset means after seed averaging; audit related datasets too |
| Is this a historical reproduction? | Match paper, weights, data, splits, preprocessing, search and aggregation |
| Is a later improvement contradictory? | No: first align model and evaluation versions |

**Worked trace:** A=[.1,.1,.1,missing], B=[.2,.2,.2,.2], RF=[.25,.25,.25,.9]. Common support: A=.1, B=.2. Explicit RF fill: A=.3, B=.2. A's fourth model score remains unmeasured.

**Independent unit:** dataset means [.10,−.04,.02] give mean .02667 and descriptive SE .04055. Nine seed rows give a naive .02048. Duplicating rows changes no dataset evidence.

'''+matrix+'\n'+paper_table+'''

Preregister before test access: full IDs and snapshots; availability; split; baseline/challenger versions; candidate/seed budgets; selection; metric; missing-run rule; uncertainty; cost cutoff; falsification test. Use the [learner template](../labs/b19-preregistration.md).

Primary reads: [BeyondArena v1](https://arxiv.org/html/2606.30410v1), [TabDPT v3 Appendix B.1](https://arxiv.org/html/2410.18164v3#A2.SS1), [Enterprise study v1](https://arxiv.org/html/2606.30452v1).
'''
(R/'reference'/f'{S}.html').write_text(document('B19 · Evidence reference',ref))
source=(P/'relkit/evidence_b19.py').read_text();tree=ast.parse(source);parts={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
tests=(P/'_test_b19.py').read_text().split('class Boundaries',1)[1].split("if __name__",1)[0];tests='class Boundaries'+tests
repro=(P/'_reproduce_b19.py').read_text();replay=next(ast.get_source_segment(repro,n) for n in ast.parse(repro).body if isinstance(n,ast.FunctionDef) and n.name=='replay_grouped')
blob=gzip.compress((P/'evidence/b19/released-grouped-scores.json').read_bytes(),mtime=0);digest=hashlib.sha256(blob).hexdigest();encoded=base64.b64encode(blob).decode()
recap=fill(body.split('## Lab ·')[0],True);recap=re.sub(r'<div id="b19-[^"]+"></div>','',recap)
recap=re.sub(r'\]\(\.\./([^)]*)\)',r'](https://avistian.github.io/relational/\1)',recap);recap=re.sub(r'\]\((b\d[^)]*\.html|0004-[^)]*\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',recap)
goals={'split_audit':'Reject duplicate, overlapping, empty or unknown row IDs. Return train/test counts, sorted overlap_groups and group_disjoint. Do not reject group overlap: report it so the caller can assess its intended deployment.','paired_scores':'Validate explicit measured/missing origins and unique dataset/seed/model identities. For measured policy retain common measured support. For rf policy fill only declared missing cells using a measured same-key RF score, retaining origin imputed:RF. Return keys, cells, means and order. Reject absent identities rather than silently inventing missing runs.','dataset_summary':'Reject duplicate dataset/seed keys and nonfinite differences. Average seeds within dataset; return equal-dataset mean, dataset means/count and sample-SD/sqrt(n) standard error. Also return the explicitly naive row-level SE for comparison; return None when fewer than two units exist.'}
for solution in [False,True]:
 cells=[]
 def md(x):cells.append(nb.v4.new_markdown_cell(x))
 def py(x):cells.append(nb.v4.new_code_cell(x))
 md(recap)
 md('## Run contract\n\nSelf-contained NumPy + Python standard library. No repository paths, downloads, credentials or installs are required. Source-visible operations below produce the course experiment. Author figures above are saved references; your current kernel outputs below are new local diagnostic execution. No model benchmark is trained. Tested environment is recorded in the repository. Live Colab frontend remains NOT_CHECKED.')
 py('# @colab-bootstrap\nimport itertools,math,json,unittest,base64,gzip,hashlib\nfrom pathlib import Path\nimport numpy as np')
 md('## PROVIDED · hand-computable behavioral checks\n\nChecks exercise duplicate identities, group overlap, common support, imputation provenance and dataset weighting. They do not grant learner mastery.')
 py(tests)
 for name,test in [('split_audit','test_split'),('paired_scores','test_pairs'),('dataset_summary','test_datasets')]:
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+goals[name])
  py(parts[name] if solution else parts[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
  md('### CHECK · run before continuing');py(f"Boundaries('{test}').debug()\nprint('{name}: passed')")
 md('## PROVIDED · fixed synthetic data\n\nNo learner functions are hidden in an import. The group labels are generated once per seed; predictor inputs exclude query targets. Two held-out rows per group versus eight held-out groups deliberately test different populations. The group ID shortcut is an illustration, not a claim about musk features or a TFM.')
 py(parts['make_fixture'])
 md('## PROVIDED · prediction and evidence aggregation\n\nTrace the two prediction rules, then the scoring-only use of y. All three learner functions are called here. The separate uncertainty fixture has three constructed datasets, unlike the three seeds of the group fixture.')
 py(parts['run_experiment'])
 md('## RUN · the complete course experiment')
 py("report=run_experiment()\nPath('b19-diagnostic.json').write_text(json.dumps(report,indent=2)+'\\n')\nassert len(report['arms'])==12\nassert sum(len(a['predictions']) for a in report['arms'])==672\nfor a in report['arms']:\n    print(a['seed'],a['regime'],a['model'],round(a['brier'],6))\nprint(report['uncertainty'])\nprint({p:r['means'] for p,r in report['policies'].items()})")
 md('## Paper lane · portable saved-score replay\n\nThe embedded compact packet contains all 9,540 selected grouped score records, with dataset/fold/configuration keys and original suite labels. Its SHA256 checks packet identity; visible code checks complete coverage and recomputes twelve default means. This portable lane does not authenticate the full source Parquet extraction: the repository audit does that separately. No IID-ablation scores or raw predictions are embedded, and no paper model is fitted.')
 py('compact=base64.b64decode('+repr(encoded)+')\nassert hashlib.sha256(compact).hexdigest()=='+repr(digest)+'\nrecords=json.loads(gzip.decompress(compact))\nassert len(records)==9540')
 py(replay)
 py("grouped_summary=replay_grouped(records)\nPath('b19-grouped-summary.json').write_text(json.dumps(grouped_summary,indent=2)+'\\n')\nfor row in grouped_summary:\n    print(row['dataset'],row['model'],row['folds'],row['metric'],round(row['mean_error'],6))\nprint('COMPLETE_SELECTED_SCORE_TABLE_AUDIT; full Figure F.2 INCOMPLETE_SOURCE_PROTOCOL_GATE / NOT_RUN')")
 md('## EXIT · produce your own preregistration\n\nChoose an untouched accessible grouped or temporal task. Before test access, specify its deployment population, full identities and label-availability times, inner/outer splits, baseline/challenger versions and pretraining exposure, accessible support/features, preprocessing/search/refit rules, seed and total compute budget, primary metric, paired support, missing-run policy, independent unit and falsification criterion. This notebook has not run that new task. Explain in 150–200 words why the 9,540 saved scores do not establish full Figure F.2 reproduction. Ask the teacher to review your contract and revisit after 1/7/30 days.')
 py("print('Author diagnostic complete; learner PENDING_WRITTEN_DEFENSE; paid compute USD0; full paper NOT_RUN')")
 n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),language_info=dict(name='python',version='3.12')))
 nb.write(n,(P/'solutions'/f'{S}.ipynb') if solution else P/f'{S}.ipynb')
print('Built B19 lesson, reference, three figures and portable student/solution notebooks')
