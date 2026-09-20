"""Canonical L095 builder. Student TODOs stay blank; identical solution code retains outputs."""
import ast,base64,hashlib,json,re
from pathlib import Path
from _walkthrough_delivery import snapshot, finalize
snapshot(95)
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0095-bipartite-graphs';T='Bipartite graphs: from interactions to recommendations'
result=json.loads((P/'_experiment_l095_results.json').read_text())
def result_table():
    text='| Method | Recall@10, mean ± fold SD | NDCG@10, mean ± fold SD |\n|---|---:|---:|\n'
    for name,metrics in result['summary'].items():
        values=[f"{metrics[m]['mean']:.4f} ± {metrics[m]['fold_sd']:.4f}" for m in ['recall','ndcg']]
        text+='| '+name.replace('_',' ')+' | '+' | '.join(values)+' |\n'
    text+='\n| Official fold | Eligible test users | Excluded: no held-out likes | Selected α |\n|---|---:|---:|---:|\n'
    for row in result['runs']:
        m=row['test']['walk'];text+=f"| {row['fold']} | {m['users']} | {m['excluded_no_relevant']} | {row['alpha']:g} |\n"
    return text+'\nFrozen author-reference measurements; not output from your current notebook kernel. The official folds have different eligible user populations. We retain their released assignments; fold variation includes this composition difference.'
def prose(portable=False):
    s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS_TABLE]]',result_table())
    for tag,name,caption in [('PIPELINE_FIG','pipeline','Complete course protocol. The official test stays hidden during fitting and mixture selection.'),('WALK_FIG','walk','Synthetic worked graph: u0→i1→u1→i2 has probability 1/8 before candidate masking.'),('PROJECTION_FIG','projection','Synthetic binary B and both projections. The off-diagonal shared-user/item counts differ from original interactions.'),('RESULTS_FIG','results','Fresh author measurements on five complete official folds. Points are fold-level user averages, not independent datasets.')]:
        src='data:image/png;base64,'+base64.b64encode((P/f'figures/l095/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l095/{name}.png'
        s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
    for tag,id_,fallback in [('WARMUP','warmup','**Cold retrieval:** attempt the three written questions before reading their answers.'),('BOUNDARY_WIDGET','boundary-widget','**Intervention:** the CHECK below introduces a reverse-only leak. Predict which assertion rejects it.'),('PREDICTION','prediction','**Predict before executing:** does removing u1–i1 change S[u0,i2] from 1/8 to zero? Explain the path.'),('WALK_WIDGET','walk-widget','**Portable intervention:** the later cell removes the bridge and recomputes both walks.'),('TEACHBACK','teachback','**Written defense:** submit the EXIT response to the agent; execution alone is not mastery.')]:s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id_}"></div>')
    if portable:
        s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
        s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
    return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 95 — {T}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','bipartite-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0094-hin-survey.html">Lesson 94</a> · <a href="../reference/bipartite-contract.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 095</p><h1>'+T+'</h1></header>'
footer='</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','bipartite-viz','l095-lesson'])+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose())+footer)
source=(P/'relkit/bipartite_l095.py').read_text();tree=ast.parse(source)
checks={
'build_graph':"tiny=np.array([[0,0,5,1],[0,1,4,2],[1,1,5,3],[1,2,4,4],[2,0,1,5]])\ng=build_graph(tiny,4,4)\nassert g['user'].num_nodes==4 and g['item'].num_nodes==4\nassert g['user','likes','item'].edge_index.shape==(2,4)\nassert torch.equal(g['item','rev_likes','user'].edge_index,g['user','likes','item'].edge_index.flip(0))\ntry:\n    build_graph(np.vstack([tiny,tiny[0]]),4,4)\nexcept ValueError:\n    print('PASS: repeated pairs rejected; typed IDs and isolated nodes retained')\nelse:\n    raise AssertionError('Repeated pairs need an explicit policy')",
'assert_boundary':"held=np.array([[0,2,5,6]])\nassert_boundary(g,held)\nbad=g.clone();bad['item','rev_likes','user'].edge_index[0,0]=2\ntry:\n    assert_boundary(bad,held)\nexcept AssertionError:\n    print('PASS: reverse-only corruption rejected')\nelse:\n    raise AssertionError('Forward-only checks miss this leak')",
'walk_scores':"s,p=walk_scores(g)\nnp.testing.assert_allclose(s[0],[.375,.5,.125,0],atol=1e-12)\nnp.testing.assert_allclose(s[2:],np.tile(p,(2,1)))\nnp.testing.assert_allclose(s.sum(1),1)\nassert np.isfinite(s).all()\nprint('PASS: three-step probabilities and cold-user fallback')"}
instructions={
'build_graph':'Validate integer local IDs, rating range and pair uniqueness. Use the declared node counts. Store only ratings ≥4 as forward edges, with the exact reverse. Do not merge the two namespaces.',
'assert_boundary':'Convert both edge stores to user–item pair sets. Reject duplicates, unequal forward/reverse content and any held-out pair in either view. Use assertions so the diagnostic CHECK can distinguish rejection.',
'walk_scores':'Construct the binary matrix from the fitting graph. Normalize outgoing choices at each type, compose the three-step walk, and return scores plus fitting-popularity probabilities. Zero-degree users use popularity; if the graph has no likes, use uniform popularity.'}
bootstrap="""# @colab-bootstrap
import os,sys,subprocess
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','torch==2.8.0','torch-geometric==2.6.1'])
import importlib.metadata as metadata
print({k:metadata.version(k) for k in ['numpy','torch','torch-geometric']})
"""
for solution in [False,True]:
    cells=[nbf.v4.new_markdown_cell('# '+T+'\n\nComplete Tier B MovieLens 100K lab, with Tier C synthetic mechanism checks. Full five-fold experiment runs by default; no GPU is needed. Three TODOs implement the live graph/scorer. No repository imports. Download the archive directly from GroupLens; acknowledge Harper & Konstan (2015), DOI 10.1145/2827872. Teacher results below are explicitly frozen. Learner mastery remains PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
    cells += [nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=\n## )',prose(True)) if x.strip()]
    cells.append(nbf.v4.new_markdown_cell('## Visible implementation and immediate feedback\n\nThe data audit reproduces the named release contract; the ranking experiment is course-defined. Every function below is used by the complete run. Implement each task before proceeding to its CHECK.'))
    prelude='\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))+'\ntorch.set_num_threads(1)'
    cells.append(nbf.v4.new_code_cell(prelude))
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        name=node.name;code=ast.get_source_segment(source,node)
        if name in checks:
            cells.append(nbf.v4.new_markdown_cell('### TODO · '+name+'\n\n'+instructions[name]))
            if not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Complete this task")'
            cells.append(nbf.v4.new_code_cell(code,metadata={'task':name}))
            cells.append(nbf.v4.new_markdown_cell('### CHECK · '+name))
            cells.append(nbf.v4.new_code_cell(checks[name],metadata={'check':name}))
        else:
            cells.append(nbf.v4.new_markdown_cell('### PROVIDED · '+name+'\n\n'+(ast.get_docstring(node) or '')))
            cells.append(nbf.v4.new_code_cell(code))
    cells += [nbf.v4.new_markdown_cell('## Predict → intervene → explain\n\nRemove the u1–i1 bridge, retaining all other likes and the scoring rule. Write your predicted S[u0,i2] before executing. Why does the candidate list, rather than the largest raw probability, determine the recommendation?'),nbf.v4.new_code_cell("changed=tiny[~((tiny[:,0]==1)&(tiny[:,1]==1))]\nnew_graph=build_graph(changed,4,4)\nnew_scores,_=walk_scores(new_graph)\nassert s[0,2]==.125 and new_scores[0,2]==0\nprint('u0 to i2 baseline:',s[0,2],'intervention:',new_scores[0,2])\nmask_test=np.array([[0,2,5,6]])\nm,d=rank_metrics(s,tiny,mask_test,k=1)\nassert d[0]['top_items']==[2]\nprint('Seen items excluded; top recommendation:',d[0]['top_items'])")]
    cells += [nbf.v4.new_markdown_cell('## Full released-data target and full course experiment\n\nThis is a fresh full run, not frozen evidence. It checks every released fold and scores all eligible users against the complete catalog. Only inner validation chooses the mixture. Refit uses all official base rows. The full loop uses the three functions you implemented.'),nbf.v4.new_code_cell("archive=Path(os.environ.get('L095_DATA_PATH','ml-100k.zip'))\narrays,member_hashes=read_release(archive)\nfresh_audit=audit_release(arrays)\nprint('Published release contract:',fresh_audit)\nfresh=run_experiment(arrays,Path('l095-results'))\nPath('l095-fresh.json').write_text(json.dumps({'audit':fresh_audit,'experiment':fresh},indent=2))\nfor name,row in fresh['summary'].items():\n    print(f\"{name:22} Recall@10={row['recall']['mean']:.4f} ± {row['recall']['fold_sd']:.4f}; NDCG@10={row['ndcg']['mean']:.4f} ± {row['ndcg']['fold_sd']:.4f}\")\nfor row in fresh['runs']:\n    print('Fold',row['fold'],'chosen alpha',row['alpha'],'eligible users',row['test']['walk']['users'])")]
    cells +=[nbf.v4.new_markdown_cell('## CHECK · fresh measurements against labeled author reference\n\nExact hashes identify the same data. Metrics allow 1e-12 numeric tolerance for the same deterministic protocol; close numbers alone do not establish historical experiment identity.'),nbf.v4.new_code_cell('AUTHOR_REFERENCE=json.loads('+repr(json.dumps(result['summary']))+')\nfor name in AUTHOR_REFERENCE:\n    for metric in ["recall","ndcg"]:\n        np.testing.assert_allclose(fresh["summary"][name][metric]["mean"],AUTHOR_REFERENCE[name][metric]["mean"],atol=1e-12,rtol=0)\nprint("PASS: fresh full-fold summary agrees with author reference")')]
    cells +=[nbf.v4.new_markdown_cell('## EXIT TICKET · PENDING_WRITTEN_DEFENSE\n\nPaste the run summary and your answers to the agent:\n\n1. Why is (user:0,item:0) not a self-loop?\n2. Why must the reverse relation share the forward split?\n3. Why mask low-rated fitting items even though they have no likes edge?\n4. Trace 1/8 and explain the projection information loss.\n5. Define the candidate catalog, relevance convention, eligible users, tie rule and selection boundary.\n6. What exactly did the published-data audit reproduce, and what score was not claimed?\n7. Specify the cutoff and repeated-event policy needed to turn this into a next-purchase study.\n\nTeacher execution is not evidence of your mastery. Ask follow-up questions wherever you cannot justify a step.')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in cells if c.cell_type=='code']
        if [c.source for c in before]==[c.source for c in after]:
            for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
    for i,c in enumerate(cells):c.id=f'l095-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
    nbf.validate(nb);nbf.write(nb,path)
    if solution:
        html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
print('Built L095 lesson, student, solution and prepared HTML')

finalize(95, preview='solution')
