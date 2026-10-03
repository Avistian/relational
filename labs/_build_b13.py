"""Build portable B13 teaching package from visible source and measured evidence."""
import ast,base64,hashlib,io,json,re,zipfile,sys
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;S='b13-synthetic-relational-data';E=P/'evidence/b13';F=P/'figures/b13';SRC=P/'sources/b13';F.mkdir(parents=True,exist_ok=True)
INK='#183c50';TEAL='#16877c';PURPLE='#6557a5'

def canvas(title,subtitle,size=(11,7)):
    f,a=plt.subplots(figsize=size);f.patch.set_facecolor('#fbfcfd');a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
    a.text(.03,.95,title,fontsize=21,weight='bold',color=INK);a.text(.03,.89,subtitle,fontsize=11,color='#52616b');return f,a
def box(a,x,y,w,h,title,body,color='#eaf6f2'):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',facecolor=color,edgecolor='#b4cdd2'))
    a.text(x+.014,y+h-.035,title,fontsize=11,weight='bold',color=INK,va='top');a.text(x+.014,y+h-.10,body,fontsize=10.5,color=INK,va='top',linespacing=1.4)
def arrow(a,x,y,u,v):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=INK,lw=1.7))
def save(f,name):f.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(f)

f,a=canvas('RDB-PFN · relational worlds, tabular episodes','Top: pretraining data route. Bottom: selected released-checkpoint prediction route.')
box(a,.03,.61,.27,.20,'Synthetic prior','Schema: LayerDAG\nStructure: keys + states\nContent: attributes')
box(a,.37,.61,.27,.20,'Linearize each target','Joins + DFS aggregations\nExample: count / mean\n→ fixed feature vector','#eef0fb')
box(a,.71,.61,.25,.20,'Two-stage training','Single-table warm-up\n→ relational episodes\nLoss → weight updates','#fff1dd')
arrow(a,.31,.71,.35,.71);arrow(a,.65,.71,.69,.71)
box(a,.03,.27,.27,.22,'Real query + support','Same representation\n512 known-label examples\nQuery label hidden')
box(a,.37,.27,.27,.22,'6 tabular blocks','Scalar → 96-wide token\nFeature attention ↔\nsupport-reading attention','#eef0fb')
box(a,.71,.27,.25,.22,'Target-token decoder','4 heads · FFN 192\nQuery token → logits\n→ class probability','#fff1dd')
arrow(a,.31,.38,.35,.38);arrow(a,.65,.38,.69,.38);arrow(a,.84,.59,.52,.51)
a.text(.03,.13,'Relational information enters through constructed features; no FK adjacency mask enters this predictor.',fontsize=10.5,color=INK)
a.text(.03,.055,'B13: full saved inference replay. No new pretraining or checkpoint inference.',fontsize=11,color=TEAL);save(f,'rdbpfn')
f,a=canvas('PluRel → RT · keep the cell relationships','Generator makes databases; the Relational Transformer learns masked-cell prediction.')
box(a,.03,.61,.27,.20,'1 · Schema → 2 · keys','DAG of tables\nClustered bipartite row links\n→ parent / child identities')
box(a,.37,.61,.27,.20,'3 · Conditional values','Parent values + noise\nType-specific SCMs\n→ tables in topological order')
box(a,.71,.61,.25,.20,'Sample a query context','Follow row / FK paths\nFuture rows excluded\n≤ 1024 cells','#fff1dd')
arrow(a,.31,.71,.35,.71);arrow(a,.65,.71,.69,.71);arrow(a,.84,.59,.84,.51)
box(a,.71,.27,.25,.22,'Cell embeddings','Typed value + name\nHidden value → mask\n1024 × 256 vectors','#eef0fb')
box(a,.37,.27,.27,.22,'12 RT blocks','Column / feature / neighbor\nQK normalization · 8 heads\nNo full-attention stage','#eef0fb')
box(a,.03,.27,.27,.22,'Typed head + loss','Numeric → Huber loss\nBoolean → cross-entropy\nLoss → parameter updates','#fff1dd')
arrow(a,.69,.38,.66,.38);arrow(a,.35,.38,.31,.38)
a.text(.03,.13,'Real deployment: the same context → embedding → relational blocks → head; hide the queried value.',fontsize=10.5,color=INK)
a.text(.03,.055,'Table 1 adds leave-one-database-out continued pretraining. B13 audits its contract; training NOT_RUN.',fontsize=10.5,color=TEAL);save(f,'plurel')
f,a=canvas('A key chooses a value before the mean','Hand-worked fixture; this is the course rule, not either paper model.',size=(10,4.5))
box(a,.03,.48,.25,.30,'Parent A','row 0 → 2  ← chosen\nrow 1 → 6')
box(a,.37,.48,.25,.30,'Parent B','row 0 → 10\nrow 1 → 14  ← chosen')
box(a,.71,.48,.25,.30,'Query','local x0 = 4\nkeys: A0, B1','#fff1dd')
a.text(.05,.30,'Gather [2, 14]  →  mean 8  →  0.5 × 4 + 8 = 10',fontsize=18,color=TEAL)
a.text(.05,.16,'Change only A0 → A1:  [6, 14]  →  mean 10  →  prediction 12',fontsize=13,color=INK)
a.text(.05,.04,'Foreign keys still resolve. A feature-only component stays 0.5 × 4 = 2.',fontsize=11,color=INK);save(f,'trace')
report=json.loads((E/'diagnostic.json').read_text());replay=json.loads((E/'rdbpfn-replay.json').read_text())
table='| Predictor | Training families | Intact MSE mean ± SD | Shuffled-FK MSE mean ± SD |\n|---|---:|---:|---:|\n'
for arm in ['feature-only','relational']:
    for d in [1,3]:
        rows=[r for r in report['conditions'] if r['arm']==arm and r['diversity']==d]
        vals=[r['mse'] for r in rows];broken=[r['corrupted_mse'] for r in rows]
        table+=f'| {arm} | {d} | {np.mean(vals):.5f} ± {np.std(vals,ddof=1):.5f} | {np.mean(broken):.5f} ± {np.std(broken,ddof=1):.5f} |\n'
table+='\nMeasured author evidence. SD uses three seed values; it is not a confidence interval. Lower MSE is better.'
f,axes=plt.subplots(1,2,figsize=(10,4.6),sharey=True);f.patch.set_facecolor('#fbfcfd')
for ax,key,title in zip(axes,['mse','corrupted_mse'],['Intact foreign keys','Shuffled test foreign keys']):
    for arm,color,off in [('feature-only',PURPLE,-.06),('relational',TEAL,.06)]:
        for seed in range(3):
            vals=[next(r[key] for r in report['conditions'] if (r['seed'],r['diversity'],r['arm'])==(seed,d,arm)) for d in [1,3]]
            ax.plot(np.array([1,3])+off+(seed-1)*.015,vals,'o-',color=color,alpha=.65,label=arm if seed==0 else None)
    ax.set(title=title,xlabel='Training schema families',xticks=[1,3],ylim=(-.03,1.12));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
axes[0].set_ylabel('Held-out diamond MSE (lower is better)');f.suptitle('12 fits · lines pair one seed across diversity levels',color=INK);f.tight_layout();save(f,'results')
rt='| Released configuration | Replayed AUROC mean ± SD | v5 paper target |\n|---|---:|---:|\n'
for name,row in replay['models'].items():rt+=f'| {name} | {row["mean"]:.6f} ± {row["sample_sd"]:.6f} | {row["paper_target"]:.4f} |\n'
rt+='\nTen support draws on one fixed test population; SD is across supports. All means are within the predeclared descriptive tolerance 0.02.'

paths={'rdbpfn':['Generate schema, row connectivity and conditional attributes.','Linearize relational neighborhoods into fixed feature vectors.','Pretrain on labeled-support / hidden-query episodes.','Real task: construct the same representation; normalize from support only.','Scalar and target tokens → six feature/example-attention blocks → query target decoder.'],
'plurel':['Generate an acyclic table schema, clustered row links and conditional values.','Sample a query-centered, temporally filtered context of at most 1024 cells.','Embed typed values and names; replace the hidden target with a mask.','Twelve blocks use column, feature and neighbor attention; QK normalization, no full attention.','Typed heads predict masked values; pretraining updates weights. Table1 compares real-only and synthetic-initialized continued pretraining.']}
def figure(name,caption):
    wide=' synthetic-wide' if name in paths or name=='results' else ''
    h=f'<figure class="synthetic-figure{wide}"><img src="../labs/figures/b13/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
    if name in paths:h+='<div class="synthetic-mobile"><strong>'+('RDB-PFN' if name=='rdbpfn' else 'PluRel → RT')+' architecture</strong><ol>'+''.join('<li>'+s+'</li>' for s in paths[name])+'</ol></div>'
    if name=='results':
        h+='<div class="synthetic-mobile" data-mobile-results><h3>All measured seeds</h3><p>Each line shows intact → shuffled-FK MSE. Lower is better; no refitting after the shuffle.</p>'
        for arm in ['feature-only','relational']:
            for d in [1,3]:
                rows=[r for r in report['conditions'] if r['arm']==arm and r['diversity']==d]
                h+=f'<h4>{arm} · {d} training '+('family' if d==1 else 'families')+'</h4><ul>'
                for r in rows:h+=f'<li>Seed {r["seed"]}: {r["mse"]:.5f} → {r["corrupted_mse"]:.5f}</li>'
                h+='</ul>'
        h+='<p>The same three seeds are paired across the two diversity levels. The relational diversity changes are small and mixed.</p></div>'
    return h
key_widget='''<div class="synthetic-board" data-fk-trace><h3>Predict, then change one key</h3><p>Query x0=4 and parent values stay fixed. This is an arithmetic illustration, not a fitted prediction.</p><label>Parent A row <select name="a"><option value="0">0: value 2</option><option value="1">1: value 6</option></select></label><label>Parent B row <select name="b"><option value="0">0: value 10</option><option value="1" selected>1: value 14</option></select></label><button type="button">Reset keys</button><output aria-live="polite">Default: (2+14)/2=8; prediction=10. Feature-only component=2.</output><noscript><p>Changing A to row1 reads6 and predicts12. Controls need JavaScript; the worked calculation above is complete.</p></noscript></div>'''
schema_widget='''<div class="synthetic-board" data-schema-identity><h3>Does a new name create a new schema?</h3><label>Topology <select name="family"><option value="chain">Chain</option><option value="out-star">Out-star</option><option value="in-star">In-star</option><option value="diamond">Diamond: held out</option></select></label><label>Rename all tables <select name="rename"><option value="no">Original names</option><option value="yes">Different names</option></select></label><div data-graph></div><button type="button">Reset schema</button><output aria-live="polite">Chain: A→B→C→D. Renaming preserves the topology. Diamond is held out.</output><noscript><p>Chain A→B→C→D and Z→X→W→Y are isomorphic. Diamond A→{B,C}→D is structurally different.</p></noscript></div>'''
replacements=dict(RDBPFN=figure('rdbpfn','RDB-PFN: generator and DFS feed a tabular episode learner. Released dimensions are explicit.'),PLUREL=figure('plurel','PluRel databases feed a cell-native RT. Generator and learner are distinct components.'),TRACE=figure('trace','A0 and B1 gather values 2 and 14; changing one foreign key changes the read.'),RESULT_FIGURE=figure('results','All three paired seeds, before and after the FK intervention; same vertical scale.'),RESULTS=table,REPLAY_RESULTS=rt,KEY_WIDGET=key_widget,SCHEMA_WIDGET=schema_widget,WARMUP='<div id="b13-warmup"></div>',PREDICT='<div id="b13-predict"></div>',TEACHBACK='<div id="b13-teachback"></div>')
source=(R/'lessons/content'/f'{S}.md').read_text();body=source
for key,value in replacements.items():body=body.replace('{{'+key+'}}',value)
def document(title,body,interactive=False):
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','synthetic-data'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/synthetic-data.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+render(body)+'</article>'+''.join('<script src="../assets/'+s+'.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B13 · Synthetic relational data',body,True))
reference='''# B13 · Generator, representation, learner

| Route | Generated object | Representation | Learner |
|---|---|---|---|
| RDB-PFN | Relational practice databases | DFS-derived feature vectors | Tabular in-context transformer |
| PluRel → RT | Schemas, keys and conditional cell values | Sampled relational cell contexts | Relational Transformer |
| B13 course | Four-table numeric fixtures | Query attributes ± parent mean | Ridge regression |

**Schema ≠ database instance ≠ row count.** A schema specifies table relationships; an instance supplies keys and values. New seeds may retain the same schema. Canonical directed graph identity rejects renamed copies. PluRel v1 assumes acyclic schemas.

**Worked gather:** A=[2,6], B=[10,14], keys=(0,1) → values=(2,14) → mean8. With query x0=4, .5*x0+mean=10. Change Akey to1 → mean10 →12. Row IDs are not attribute values.

**Fair holdout:** split schema families before fitting; keep data and target generation seeds distinct from test; count generated cells and keys separately; match fit/selection budgets; report every seed and paired query error. A sufficient-statistic feature can make an unseen topology easy without learning graph computation.

**Measured course result:**12fits,3072intact+3072corrupted predictions. Parent feature helps this constructed target; three training families do not consistently improve the relational arm. No real-data or published scaling claim follows.

**Paper evidence:** complete30-evaluation/21060-prediction L200 replay, not new inference. RDB-PFN Table9 is v5. PluRel v1 Table1 requires36real-data training runs and108task evaluations, plus synthetic-base reconstruction if required. Paper code exists; full historical seed mapping and paid execution remain gated. Learner defense pending.

[Lesson](../lessons/b13-synthetic-relational-data.html) · [Lab](../labs/b13-synthetic-relational-data.ipynb) · [Contract](../labs/b13-reproduction.md) · [RDB-PFN v5](https://arxiv.org/html/2603.03805v5) · [PluRel v1](https://arxiv.org/html/2602.04029v1)
'''
(R/'reference'/f'{S}.html').write_text(document('B13 · Synthetic data reference',reference))

# Complete source-only package, retaining repository-relative paths for the evaluator.
payload={}
for p in sorted(SRC.rglob('*')):
    if p.is_file() and p.suffix not in ['.zip','.pyc']:payload[str(p.relative_to(R))]=p.read_bytes()
with zipfile.ZipFile(SRC/'plurel-v1.0.0.zip') as z:
    for name in z.namelist():
        rel=Path(*Path(name).parts[1:])
        if not name.endswith('/') and (rel.parts and rel.parts[0] in ['plurel','rt','rustler','scripts','test'] or len(rel.parts)==1):
            payload['labs/sources/b13/plurel-paper/'+str(rel)]=z.read(name)
source_manifest=json.loads((P/'sources/l166/source-ledger.json').read_text())
for name,digest in source_manifest['files'].items():
    if name.startswith('upstream/'):
        p=P/'sources/l166'/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        payload['labs/sources/l166/'+name]=p.read_bytes()
for name in ['_fetch_l166.py','_run_l166.py','l166-requirements.txt','evidence/l166/prepared.npz','evidence/l166/input-manifest.json','sources/l166/source-ledger.json','b13-reproduction.md','_reproduce_b13.py']:
    payload['labs/'+name]=(P/name).read_bytes()
hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(payload.items())}
payload['source-manifest.json']=(json.dumps(hashes,indent=2)+'\n').encode()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(payload.items()):
        info=zipfile.ZipInfo(name,date_time=(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
packet=buf.getvalue();(E/'portable-sources.zip').write_bytes(packet)
(E/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')

module=(P/'relkit/synthetic_b13.py').read_text();nodes=ast.parse(module).body
funcs={n.name:ast.get_source_segment(module,n) for n in nodes if isinstance(n,ast.FunctionDef)}
tests=(P/'_test_b13.py').read_text();testfunc={n.name:ast.get_source_segment(tests,n) for n in ast.parse(tests).body if isinstance(n,ast.FunctionDef)}
def functions(path):
    s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef))
def embedded(name):return '!['+name+'](data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode()+')'
def archive_cell(data,name):
    return 'archive=base64.b64decode('+repr(base64.b64encode(data).decode())+')\nassert hashlib.sha256(archive).hexdigest()=='+repr(hashlib.sha256(data).hexdigest())+'\n'+name

for solution in [False,True]:
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(s))
    md('# B13 · Two roles for synthetic relational data\n\n**PROVIDED:** generator, source model listings, all saved evidence. **TODO:** three live functions. **CHECK:** immediate numerical feedback and complete experiments. **EXIT:** written defense.\n\nDefault execution is a NumPy course experiment plus complete saved RDB-PFN replay. It does not launch paper pretraining or fresh checkpoint inference. USD0 paid scope; a future rerun has its own time budget.')
    code('''# @colab-bootstrap — portable NumPy lab, no repository imports
import importlib.util,subprocess,sys
if importlib.util.find_spec('numpy') is None:subprocess.check_call([sys.executable,'-m','pip','install','numpy'])
import numpy as np
import base64,hashlib,io,itertools,json,copy,tempfile,zipfile
from pathlib import Path
workspace=Path(tempfile.mkdtemp(prefix='b13-lab-'))
print('NumPy',np.__version__,'; workspace',workspace)
''')
    narrative=source[source.index('B12 asked'):source.index('## 8 · Lab and defense')]
    for key,value in replacements.items():
        if key in ['RDBPFN','PLUREL','TRACE','RESULT_FIGURE']:value=embedded({'RDBPFN':'rdbpfn','PLUREL':'plurel','TRACE':'trace','RESULT_FIGURE':'results'}[key])
        elif key in ['WARMUP','PREDICT','TEACHBACK','KEY_WIDGET','SCHEMA_WIDGET']:value='**Recall / predict before reading:** distinguish the generated database, its representation, and the learner. Does renaming change topology? Must diversity help when the right feature is already supplied?'
        narrative=narrative.replace('{{'+key+'}}',value)
    narrative=re.sub(r'\[([^\]]+)\]\(\.\./[^)]+\)',r'\1 (included in the extracted source/evidence packet)',narrative)
    md('## Model architecture and concept overview\n\nThe following author-reference narrative and figures stand alone. Displayed results are author evidence, not your current kernel output.')
    for part in narrative.split('\n## '):md(part if part.startswith('B12') else '## '+part)
    instructions=[('canonical_schema','test_schema','Try all table permutations; preserve edge directions, reject duplicates/self-loops/cycles/dangling nodes. Return the lexicographically smallest sorted edge tuple. Four-node factorial enumeration is intentional.'),('parent_mean','test_parent_mean','Gather each parent array using its aligned child foreign-key array, then average gathered values across parents. Reject invalid indices and nonfinite attributes. Do not average whole tables.'),('fit_ridge','test_ridge','Fit mean and population SD on training X only; use scale1 for constant columns. Center y, solve regularized normal equations with alpha>0 and unpenalized mean intercept. Return mean, scale, coef, intercept. No query arrays are inputs.')]
    for i,(name,test,instruction) in enumerate(instructions,1):
        md(f'## TODO {i} · {name}\n\n'+instruction)
        code(funcs[name] if solution else funcs[name].split('\n')[0]+f'\n    raise NotImplementedError("Complete {name}")')
        code(testfunc[test]+'\n'+test+'()\nprint("CHECK passed: '+name+'")')
    md('## PROVIDED · Schema family and generator\n\nFour independent attributes per row; keys choose parent values. The target depends on the same parent-mean statistic at train and test. This is an intentionally easy transfer rule, not PluRel\'s full clustered temporal SCM. Your `canonical_schema` and `parent_mean` functions are called here.')
    schema=next(ast.get_source_segment(module,n) for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHEMAS' for t in n.targets))
    code(schema+'\n\n'+funcs['generate_database'])
    md('## PROVIDED · Target-free readout and predictor\n\nYour gather feeds the actual relational feature. The predictor never reads database target labels. Each corrupted FK column keeps the same multiset; targets stay fixed.')
    code(funcs['features']+'\n\n'+funcs['predict'])
    md('## CHECK · All 12 fits\n\nThe runner calls your fit function. Twelve training databases per arm; four new diamond test databases per seed. Do not tune alpha using these test results.')
    code(funcs['run_experiment']+'\nreport=run_experiment()\nprint(report["status"],report["fits"],report["intact_predictions"],report["corrupted_predictions"])')
    md('## CHECK · Independent readout and SVD oracle\n\nIndependent scalar gathers and an augmented least-squares solution check every saved prediction. Replacing query labels must leave features unchanged.')
    code(functions(P/'_verify_b13.py')+'\nverification=verify(report)\nprint(verification)')
    code('for r in report["conditions"]:\n    print(r["seed"],r["diversity"],r["arm"],"MSE",round(r["mse"],6),"shuffled",round(r["corrupted_mse"],6))\n(workspace/"course-results.json").write_text(json.dumps(report,indent=2))')
    md('## EXIT · Defend the mechanism\n\nExplain why an engineered sufficient statistic can transfer across a held-out topology without learned graph reasoning. Separate schemas, database draws and cells. Trace both paper pipelines, name two falsification tests, and explain why the corruption experiment tests use of keys rather than universal superiority. Revisit after1/7/30days. Ask the teaching agent for feedback; PENDING_WRITTEN_DEFENSE remains until you defend your work.')
    md('## NEXT STEP · Complete selected RDB-PFN replay\n\nThis cell extracts immutable L200 evidence into a fresh directory, then uses a new positive-negative pair oracle. It authenticates keys, labels, support derivation and all30evaluations. The same saved predictions are replayed; no model inference occurs.')
    code(functions(P/'_audit_b13.py'))
    code(archive_cell((E/'l200-reproducer.zip').read_bytes(),'replay=replay_archive(archive)\nprint(replay["status"],replay["evaluations"],replay["predictions"])\nfor arm,r in replay["models"].items():print(arm,r["mean"],r["sample_sd"])'))
    md('## PROVIDED · Full source packet and fresh-inference route\n\nThe source packet preserves full upstream generator/model/trainer code, Rust sampler, licenses, pinned evidence preparation and complete inference operators. Its source manifest authenticates every file. Extracting code does not execute it. Commands and unresolved PluRel Table1 requirements are in `labs/b13-reproduction.md`. Fresh inference and full training are outside the default notebook and the approved paid budget.')
    code(archive_cell(packet,'''source_root=workspace/'source-packet'
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(source_root)
source_manifest=json.loads((source_root/'source-manifest.json').read_text())
for name,digest in source_manifest.items():assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,name
print('Authenticated',len(source_manifest),'source files. Training NOT_RUN.')
'''))
    md('## Full reproduction contract\n\n'+(P/'b13-reproduction.md').read_text())
    md('## PROVIDED · Visible RDB-PFN selected model and training loop\n\nThese archival listings are readable source, not the course ridge model. L166 previously checked the visible numeric path against both released checkpoints; B13does not rerun that parity test. Upstream licensing is preserved in the packet. The model exposes normalization, feature/label encoders, attention axes and decoder. The trainer exposes synthetic episode fitting; fresh pretraining is NOT_RUN.')
    md('### Source reading map\n\nRead RDB-PFN\'s feature/example attention to locate the tabular representation boundary. For PluRel, follow `dag.py` (schema) → `utils.py` (bipartite keys and typed projections) → `scm.py` (conditional values) → `dataset.py` (generation orchestration). Then inspect RT\'s relational masks and trainer. These complete reference listings are optional deeper reading after the short lab, and remain separate from the executed course generator.')
    listings=[('Visible released-shape RDB-PFN',SRC/'rdbpfn_visible.py'),('RDB-PFN upstream training',P/'sources/l166/upstream/model_pretrain/src/training.py')]
    listings += [('PluRel paper generator · '+name,SRC/'plurel-paper/plurel'/name) for name in ['dag.py','utils.py','scm.py','dataset.py']]
    listings += [('PluRel paper RT model',SRC/'plurel-paper/rt/model.py'),('PluRel paper RT trainer',SRC/'plurel-paper/rt/main.py')]
    for label,path in listings:
        md('### '+label+'\n\nComplete source listing, split at top-level definitions. Use the source reading map above to locate this stage. This appendix does not execute the generator or paper training.')
        text=path.read_text();tree=ast.parse(text)
        # Keep exact full source incl. comments between definitions, in sequential chunks.
        lines=text.splitlines();starts=[0]+[n.lineno-1 for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]+[len(lines)]
        starts=sorted(set(starts))
        for start,end in zip(starts,starts[1:]):md('```python\n'+'\n'.join(lines[start:end])+'\n```')
    md('## Full-run gate\n\nPluRel full target:36real-data fits/108task evaluations plus the synthetic base. Paper code available; historical three-seed mapping and cost admission unresolved. Do not replace Table1with later leaderboard NMAE-selected checkpoints. LiveColab and deployment are NOT_CHECKED/NOT_RUN.')
    code('RUN_FULL_REPRO=False\nif RUN_FULL_REPRO:\n    raise RuntimeError("BLOCKED: USD0 scope; unresolved historical seed mapping and cost admission. Read the full contract.")')
    n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python')))
    # Stable IDs make repeated builds byte-reproducible.
    for i,c in enumerate(n.cells):c['id']='b13-'+str(i)
    dest=P/('solutions' if solution else '')/(S+'.ipynb');dest.parent.mkdir(exist_ok=True);nb.write(n,dest)
(E/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,nbformat=nb.__version__),indent=2)+'\n')
print('Built B13 lesson, reference, four figures and portable notebooks; source packet bytes',len(packet))
