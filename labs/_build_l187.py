"""Build readable HTML and standalone, inline-code student/solution notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l187';S='0187-ethics-privacy-reg'
report=json.loads((E/'report.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
def defs(path):
    text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
functions=defs(P/'relkit/privacy_l187.py')
status='**Author evidence:** complete `L187-F1-ENTITY-PRIVACY` course experiment. All 857 drivers, 2,571 histogram neighbors and 270 simulated releases checked. Published private-GNN reproduction **NOT_RUN**; production DP and complete erasure **NOT_ESTABLISHED**. Learner **PENDING_WRITTEN_DEFENSE**.'
results='The full nine-table snapshot contains **70,876 declared owned records** and **175,933 edges incident to those owned sets**. Driver ID 3 has the largest contribution: **370 results + 353 qualifying + 372 standings + 1 driver = 1,096 records**. These IDs are snapshot identities, not names inferred by an attack.\n\n'
results+='| Cap C | ε | Kept / 26,080 | Clipping L1 | Noise MAE ± SD | Raw MAE ± SD |\n|---:|---:|---:|---:|---:|---:|\n'
for row in report['summaries']:
    results+=f"| {row['cap']} | {row['epsilon']:g} | {row['kept']:,} | {row['clipping_l1']:,} | {row['noise_mae_mean']:.3f} ± {row['noise_mae_sd']:.3f} | {row['raw_mae_mean']:.3f} ± {row['raw_mae_sd']:.3f} |\n"
captions={'ownership':'Synthetic ownership map: each arrow bundles foreign-key edges. Standings connect to races and the driver, not constructors. Deleting the profile leaves child nodes; declared owner removal still retains shared context.',
    'noise':'Synthetic Laplace density comparison for neighboring scalar counts 0 and 2. At epsilon 1, scale 2 attains ratio exp(1); the incorrect row-calibrated scale 1 permits ratio exp(2). The full-vector proof is in the text.',
    'mechanism':'Synthetic Red/Blue worked example: cap 2 reduces [2,2] to [2,1]. Removing A changes the clipped vector by [2,0], whose L1 norm is 2. At epsilon 1 the ideal Laplace scale is 2.',
    'utility':'Measured full F1 simulation. All 270 seeds/configuration runs shown; means and sample standard deviations describe simulation variation. Different panel scales are explicit. Public data and seeds confer no production privacy guarantee.'}
def prose(portable=False):
    text=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',results)
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l187'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l187/'+name+'.svg'
        text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    placeholders={'WARMUP':('Recall: why does time filtering differ from target masking? What is a foreign key? Write the answers before continuing.','<div id="warmup"></div>'),
        'PREDICT':('Predict: will deleting only the driver graph node remove all event records? Give a reason before reading on.','<div id="predict"></div>'),
        'REMOVAL':('Try the three removal operations using the ownership functions below. Default graph-node removal leaves six child nodes in the synthetic example.','<div id="privacy-removal"></div><noscript>Deleting the profile leaves six child nodes and ten edges to shared context. Removing all declared owned rows removes seven nodes and sixteen edges.</noscript>'),
        'RELEASE':('Try C = 1, 2, 3 and epsilon = 0.5, 1, 2 on the Red/Blue example. Derive the clipped vector and noise scale first.','<div id="privacy-release"></div><noscript>At cap 2 and epsilon 1, clipped counts are [2,1] and noise scale is 2. Five fresh releases compose to epsilon at most 5.</noscript>'),
        'TEACHBACK':('Use the EXIT fields below to explain the mechanism in your own words.','<div id="teachback"></div>')}
    for name,(plain,widget) in placeholders.items():text=text.replace('[['+name+']]',plain if portable else widget)
    if portable:
        text=text.replace('](../','](https://avistian.github.io/relational/')
        text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    return text
def doc(title,body,interactive=False):
    html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
    styles=['lesson','atomic-route','checkpoint','lab-access','entity-privacy']
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','entity-privacy','l187-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 187 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+s+'.css">' for s in styles)+'</head><body class="checkpoint ep-lesson"><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav><header><p class="route-kicker">Year 5 · Quarter 3 · Lesson 187</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+s+'.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Ethics & privacy on relational entity graphs',prose(),True))
reference='''## Before releasing anything

Name the protected person/entity, all owned tables, the fixed public context, the observer's knowledge, the output, and the permitted purpose. Map derived artifacts separately. A graph node is a representation unit; a person can span many nodes.

## Three operations

Deleting one result removes one row. Deleting a driver graph node removes its incident edges but can leave child nodes. Declared owner removal deletes the driver plus directly keyed results, qualifying and standings; shared aggregates remain. SQL constraints and cascades require explicit configuration.

## Bounded histogram recipe

1. Freeze a public bin domain and stable owner/row identities.
2. Keep at most C total events per person, in a stable per-owner order.
3. Sum one-hot events into the complete histogram vector.
4. Under add/remove-person adjacency, unchanged other-owner contributions give L1 sensitivity at most C. Replacement adjacency can require 2C.
5. Ideal independent hidden Laplace noise has scale C/epsilon per bin. Basic fresh-release composition sums epsilon. Reuse of the same already released answer is post-processing.

## Things the recipe does not establish

Public seeded float64 simulation is not a production DP release. Empirical neighbor checks are not a proof over all possible inputs. A failed attack does not prove privacy. A private statistic does not privatize a GNN trained on raw data. Privacy does not establish consent, fairness or complete erasure.

## Measured course experiment

'''+results+'\n\n'+status+'''

[Lesson](../lessons/0187-ethics-privacy-reg.html) · [Student notebook](../labs/0187-ethics-privacy-reg.ipynb) · [Protocol](../labs/l187-reproduction.md) · [Dwork/Roth](https://www.cis.upenn.edu/~aaroth/privacybook.html) · [Graph-privacy critique](https://arxiv.org/html/2311.06888v2).
'''
(R/'reference/ethics-privacy-reg.html').write_text(doc('Relational privacy contract — reference',reference))
(E/'report.md').write_text(('# L187 measured course experiment\n\n'+status+'\n\n'+results).rstrip()+'\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(manifest['files']):
        info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(E/'packet'/name).read_bytes())
encoded=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
def make(solution):
    cells=[]
    def md(text):cells.append(nb.v4.new_markdown_cell(text))
    def code(text,tags=None):cells.append(nb.v4.new_code_cell(text,metadata={'tags':tags or []}))
    md('# Lesson 187 · Ethics & privacy on REG\n\nComplete course experiment, not a private-GNN benchmark. Tier B: full public F1 relational snapshot. The notebook embeds all nine tables and four figures. Python3, numpy, pandas, pyarrow and duckdb are required; install missing packages before running. No cloud service or download is used by the lab. Live Colab UI is NOT_CHECKED.\n\nThree TODOs are live: ownership, contribution bounding, noise/accounting. Author-reference results below do not represent your completed work.')
    code('# @colab-bootstrap\nfrom pathlib import Path\nimport base64,hashlib,io,json,math,zipfile\nimport numpy as np\nimport pandas as pd\nimport pyarrow,duckdb\nprint("Public seeded simulator; cloud/API $0; no production DP claim")',['colab-bootstrap'])
    md(prose(True))
    md('## PROVIDED · Authenticate the complete input packet\nThe fixed public constructor domain is taken from this authenticated snapshot and held fixed across neighbors. SHA256 verifies bytes, not correctness of the privacy claim.')
    code('PACKET='+repr(encoded)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\npacket=Path('l187-packet');packet.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(packet)\nmanifest="+repr(manifest),['data-payload'])
    code(defs(P/'_run_l187.py')['load187']);code("db=load187(packet,manifest)\nconfig=json.loads((packet/'config.json').read_text())\nprint(pd.DataFrame([{'table':k,'rows':len(v)} for k,v in db.items()]).to_string(index=False))")
    tasks=[('owned_counts','Count the declared contribution','Input is a dictionary of tables. Return one row per driver in sorted driverId order with results, qualifying and standings counts, owned_rows (including profile), driver_incident_edges and owned_incident_edges. Results and qualifying have three foreign keys; standings has two. Reject missing, duplicate driver identities or unknown child owners.'),
       ('bounded_histogram','Bound the whole owner contribution','Input events have resultId, driverId and constructorId. Keep at most cap total rows per driver using ascending immutable resultId. Return an integer vector in the fixed domain order, including zero bins. Reject nonpositive/noninteger caps, duplicate row IDs, missing keys, repeated or unknown categories. Removing an owner must not change another owner\'s retained rows.'),
       ('release_scale','Connect the mechanism to its privacy unit','Return a dictionary with scale and epsilon_total for cap, epsilon and repeats. Derive scale from the complete vector sensitivity. The repeats count describes fresh independent releases on the same people, not re-reading an answer. Reject invalid integer caps/repeats and nonpositive/nonfinite epsilon.')]
    tests={'owned_counts':"tiny={'drivers':pd.DataFrame({'driverId':[1,2]}),'results':pd.DataFrame({'driverId':[1,1]}),'qualifying':pd.DataFrame({'driverId':[1]}),'standings':pd.DataFrame({'driverId':[2]})}\nassert owned_counts(tiny).owned_rows.tolist()==[4,2]",
        'bounded_histogram':"toy=pd.DataFrame({'resultId':[1,2,3,4],'driverId':[1,1,1,2],'constructorId':[10,10,20,20]})\nassert np.array_equal(bounded_histogram(toy,[10,20,30],2),[2,1,0])\nassert np.array_equal(bounded_histogram(toy.iloc[::-1],[10,20,30],2),[2,1,0])",
        'release_scale':"assert release_scale(2,1,5)=={'scale':2.,'epsilon_total':5.}"}
    for name,title,instructions in tasks:
        md('## TODO · '+title+'\n'+instructions)
        node=ast.parse(functions[name]).body[0]
        code(functions[name] if solution else 'def '+name+'('+ast.unparse(node.args)+'):\n    raise NotImplementedError("Implement '+name+'")')
        md('### CHECK · '+title);code(tests[name]+"\nprint('Immediate CHECK passed')")
    md('## CHECK · Adversarial contracts\nUnknown owners, duplicated identities, missing bins, invalid budgets and order dependence must fail. These fixtures exercise the actual learner functions.')
    code(defs(P/'_check_l187.py')['checks']);code("assert checks(owned_counts,bounded_histogram,release_scale)=='PASS'")
    md('## PROVIDED · Execute all nine configurations × 30 seeds\nNo subsampling, training or HPO. Each bin retains a float64 noisy count; negative values are not clamped. Noise MAE uses the clipped query; raw MAE uses the original full histogram. All randomness is public and reproducible.')
    code(defs(P/'_run_l187.py')['experiment187'])
    code("report,counts,releases=experiment187(db,config,owned_counts,bounded_histogram,release_scale)\nPath('l187-report.json').write_text(json.dumps(report,indent=2))\nprint(report['status'],report['releases'],'vectors',report['released_coordinates'],'coordinates')\nprint(pd.DataFrame(report['summaries']).round(4).to_string(index=False))")
    md('## CHECK · Independent full-population route\nSQL checks ownership, scalar Python checks histograms and errors, and every actual owner-deletion neighbor is recomputed. These are implementation checks, distinct from the symbolic global-sensitivity proof.')
    code(defs(P/'_verify_l187.py')['independent187'])
    code("verification=independent187(db,report,counts,releases,bounded_histogram)\nPath('l187-verification.json').write_text(json.dumps(verification,indent=2))\nprint(verification)")
    md('## TRY · Reject plausible wrong policies\nPredict which contract each incorrect implementation violates. A cap per bin is not a cap per person. A scale based on one row is not generally a person-level scale.')
    code("def wrong_per_bin(events,domain,cap):\n    kept=events.sort_values('resultId').groupby(['driverId','constructorId']).head(cap)\n    return kept.constructorId.value_counts().reindex(domain,fill_value=0).to_numpy()\n\ndef wrong_row_scale(cap,epsilon,repeats=1):\n    return {'scale':1/epsilon,'epsilon_total':epsilon}\n\ndef wrong_profile_only(db):\n    result=owned_counts(db).copy();result['owned_rows']=1;return result\n\nfor policy,functions_under_test in [('per-bin cap',(owned_counts,wrong_per_bin,release_scale)),('row scale',(owned_counts,bounded_histogram,wrong_row_scale)),('profile only',(wrong_profile_only,bounded_histogram,release_scale))]:\n    try:checks(*functions_under_test)\n    except (AssertionError,ValueError):print('Rejected:',policy)\n    else:raise AssertionError('Accepted wrong policy: '+policy)\n\nbad=json.loads(json.dumps(manifest));bad['files']['db/results.parquet']='0'*64\ntry:load187(packet,bad)\nexcept ValueError:print('Rejected altered provenance')\nelse:raise AssertionError('Accepted altered provenance')")
    md('## TRY · Show why a public noise seed changes the promise\nThis is only public demonstration data. The observer can regenerate our exact noise and approximately recover the clipped counts. Explain why the ideal mechanism requires hidden randomness.')
    code("first=releases[0]\nrng=np.random.default_rng(np.random.SeedSequence([187,first['cap'],int(first['epsilon']*10),first['seed']]))\nknown_noise=rng.laplace(0,first['cap']/first['epsilon'],len(report['domain']))\nrecovered=np.array(first['values'])-known_noise\nassert np.allclose(recovered,report['clipped_histograms'][str(first['cap'])],atol=1e-12,rtol=0)\nprint('Known-noise recovery PASS; this is why the simulator is not a private service.')")
    md('## EXIT · Your written defense\nComplete each field. Include the neighboring database definition, a derivation over all allowed inputs, and a purpose/permission judgment. Paste the defense and CHECK output to the teacher. A numerical PASS does not award mastery.')
    code("submission={'learner':'PENDING_WRITTEN_DEFENSE','why_1096_records':'','adjacency_and_shared_context':'','global_sensitivity_derivation':'','clipping_bias_vs_noise':'','seed_disclosure_and_composition':'','ethical_use_and_recourse':'','what_private_gnn_work_remains':''}\nPath('l187-submission.json').write_text(json.dumps(submission,indent=2))")
    md('## NEXT STEP · Published private-GNN reproduction remains NOT_RUN\nNo training operator is claimed here. A future GAP/HeterPoisson benchmark needs an audited adjacency/accountant, complete source/model/trainer and source-pinned dataset, splits, hyperparameters, repeats and total cost. Reproducing an accuracy figure would not resolve a disputed privacy guarantee. The approved lesson experiment is complete without upgrading those claims.')
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,cell in enumerate(cells):cell.id=f'l187-{i:03}'
    return book
for solution in [False,True]:
    book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4);book.metadata=old.metadata
        previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
        if [c.source for c in previous]==[c.source for c in current]:
            for prior,c in zip(previous,current):
                c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
    nb.write(book,path)
print('Built lesson, reference and standalone student/solution notebooks')
