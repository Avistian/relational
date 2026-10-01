"""Canonical lesson, reference, portable notebooks and complete offline evidence packet."""
import ast,base64,gzip,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l166';S='0166-rdb-pfn-synthetic-relational-priors'
report=json.loads((E/'report.json').read_text());mechanism=json.loads((E/'mechanism.json').read_text())
def table():
    rows=['| Model | Paper AUROC | Fresh mean ± sample SD | Result |','|---|---:|---:|---|']
    for name in ['RDBPFN','RDBPFN_single','TabICLv1.1']:
        m=report['models'][name];rows.append(f"| {name} | {m['paper']:.4f} | {m['mean']:.6f} ± {m['sample_sd']:.6f} | {m['status']} |")
    return '\n'.join(rows)
captions={'prior':'Schema → keys and latent states → cells → DFS → support/query prediction episodes. The original sampled database and tiny course SCM are separate artifacts.',
          'architecture':'Released numeric base: six bi-attention blocks, width 96, four heads, MLP width 192 and binary target-token decoder. Stage 2 also mixes single-table tasks. Pretraining is not rerun here.',
          'attention':'Blue entries are visible keys and values. All row receivers read support columns only. Feature attention separately mixes tokens within each row.'}
def prose(portable=False):
    text=(R/'lessons/content'/(S+'.md')).read_text()
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l166'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l166/'+name+'.svg'
        text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    variants={'WARMUP':('Recall without notes: what changes between gradient fine-tuning and in-context inference?','<div id="warmup"></div>'),
              'PREDICT':('Predict: can query D change query C by becoming its attention key? No: every row reads support keys only.','<div id="predict"></div>'),
              'EXPLORER':('Static baseline: dependency strength 1, four support rows, original labels. The code below reruns all 12 interventions with the actual checkpoint. Flip support labels without changing features or the attention mask.','<div id="prior-explorer"></div><noscript>Baseline: strength 1, four support rows, original labels. Every row reads keys 0–3 only. Changing support labels changes inputs, not weights. Use the notebook to inspect all twelve measured traces.</noscript><script type="application/json" id="prior-data">'+json.dumps(mechanism,separators=(',',':'))+'</script>'),
              'TEACHBACK':('Explain prior → DFS → two attention axes → prediction. State which training and historical provenance claims remain unestablished.','<div id="teachback"></div>')}
    for key,(static,html) in variants.items():text=text.replace('[['+key+']]',static if portable else html)
    text=text.replace('[[RESULTS]]',table())
    if portable:
        text=text.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
        text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    return text
def doc(title,body,interactive=False):
    html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','relational-prior','l166-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+name+'.css">' for name in ['lesson','atomic-route','checkpoint','lab-access','foundation-scope','relational-prior'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0165-kumorfm-in-context-relational-learning.html">Lesson 165</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 166</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+name+'.js"></script>' for name in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('RDB-PFN: synthetic relational priors',prose(),True))
reference='''## Prior, representation, predictor

The prior samples schema, keys/latent states and cells. DFS linearizes relationships. A pretrained transformer learns prediction from labeled support rows. The course SCM isolates one dependency; the executed original generator draw reuses a schema pool. Neither is full pretraining.

## Tensor path

X: [B,S+Q,F]. Support-only median fill; mean/population-variance scaling and ±100 clamp. Scalar linear projection → [B,S+Q,F,96]. Append target token column; query placeholders use support-label mean. Six blocks alternate feature attention within rows and row attention within columns. Every receiver reads only support keys/values. Query target tokens → MLP192 → two logits → softmax. 692,738 parameters.

## Three invariants

Foreign keys must resolve. Child permutations must preserve count/mean. Query targets and other query keys cannot enter support-only row attention. A later query never changes a historical feature cutoff.

## Selected reproduction

Table9, driver-dnf, 512 support rows, ten seeds, same DFS features and full 702-query test population; TabICLv1.1 uses 32 estimators. Pin one checkpoint before test access. Preserve complete (driverId,date) keys and pair support sets.

'''+table()+'''

All 30 evaluations complete; 21,060 predictions independently rescored. Seed SD describes repeated support draws, not database variation. Released targets complement current raw DNF labels. Fresh pretraining and full paper NOT_RUN; historical preprocessing identity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0166-rdb-pfn-synthetic-relational-priors.html) · [Contract and run commands](../labs/l166-reproduction.md) · [Defense](../labs/l166-defense-template.md) · [Primary paper v5](https://arxiv.org/html/2603.03805v5).
'''
(R/'reference/rdb-pfn.html').write_text(doc('RDB-PFN — quick reference',reference))
(E/'report.md').write_text('# L166 selected reproduction\n\n'+table()+'\n\n30 evaluations; 21,060 predictions; 702 unique test queries. All targets match at four decimals.\n\nPaired RDBPFN minus single-table: '+str(report['paired']['RDBPFN-minus-RDBPFN_single']['mean'])+' AUROC. Paired RDBPFN minus TabICLv1.1: '+str(report['paired']['RDBPFN-minus-TabICLv1.1']['mean'])+' AUROC. Ten support draws, one task.\n\nReleased checkpoint replay COMPLETE; fresh pretraining / whole paper NOT_RUN; historical identity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. See [full contract](../../l166-reproduction.md).\n')
def definitions(path):
    text=path.read_text();return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
def packed(raw):return base64.b64encode(gzip.compress(raw,mtime=0)).decode()
payload={}
for name in ['prepared.npz','input-manifest.json','predictions.json','mechanism.json','report.json']:
    raw=(E/name).read_bytes();payload['evidence/l166/'+name]=dict(data=packed(raw),sha256=hashlib.sha256(raw).hexdigest())
raw=(E/'checkpoints/RDBPFN.pt').read_bytes();payload['checkpoint.pt']=dict(data=packed(raw),sha256=hashlib.sha256(raw).hexdigest())
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted((P/'sources/l166/upstream/model_pretrain').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            info=zipfile.ZipInfo(str(p.relative_to(P/'sources/l166/upstream')));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
payload['upstream.zip']=dict(data=packed(buf.getvalue()),sha256=hashlib.sha256(buf.getvalue()).hexdigest())
tasks={
 'relational_prior':('Generate paired two-table draws.','Use default_rng(seed): parent standard normals, child integer parent indices, then normal noise with SD0.2. Return exactly parent_ids,latent,child_parent,noise,values. Values=strength*latent[child_parent]+noise. Same seed must hold keys/noise fixed when strength changes. Require positive integer sizes and finite strength. This is the course SCM, not the original generator.',"p=relational_prior(7,4,12,1.);q=relational_prior(7,4,12,0.)\nnp.testing.assert_allclose(p['values']-q['values'],p['latent'][p['child_parent']])\nprint('CHECK: intervention holds keys and noise fixed')"),
 'dfs_summary':('Aggregate children without depending on row order.','Inputs: unique hashable parent IDs, aligned child foreign keys and finite numeric values. Return one [count,mean] row per supplied parent, preserving parent order. Empty group->[0,0]. Reject unknown keys, duplicate parents, invalid shapes or nonfinite values with ValueError.',"np.testing.assert_allclose(dfs_summary([0,1,2],[1,0,1],[2.,10.,6.]),[[1,10],[2,4],[0,0]])\nprint('CHECK: count, mean, empty group')"),
 'context_mask':('Specify which rows can provide keys and values.','rows/support must be integers excluding bool,0<support<=rows. Return a Boolean rows×rows array. Entry[i,j] is True exactly when j<support; receiver role does not change visibility. Reject invalid boundaries with ValueError.',"np.testing.assert_array_equal(context_mask(3,2),[[True,True,False]]*3)\nprint('CHECK: query keys remain hidden')")}
for solution in [False,True]:
    cells=[nb.v4.new_markdown_cell('# Lesson 166 · RDB-PFN synthetic relational priors\n\nStandalone lesson, real checkpoint, three live learner mechanisms, and complete saved-evidence replay. Requires NumPy and PyTorch; no cloud/API calls during the default run. Portable diagrams and all default-run inputs are embedded. Public course links become available only after publication.'),nb.v4.new_markdown_cell(prose(True)),
           nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport ast,base64,gzip,hashlib,io,json,shutil,sys,urllib.request,zipfile\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\ntorch.set_num_threads(1)\nP=Path("l166-portable");P.mkdir(exist_ok=True)')]
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Authenticated portable input packet\nContains the actual RDB-PFN checkpoint, complete prepared benchmark arrays, all thirty saved prediction sets, and pinned original inference source. This is data storage, not hidden learner logic.'))
    c=nb.v4.new_code_cell('payload='+repr(payload)+'\nfor name,item in payload.items():\n    raw=gzip.decompress(base64.b64decode(item["data"]))\n    assert hashlib.sha256(raw).hexdigest()==item["sha256"]\n    dest=P/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)\nwith zipfile.ZipFile(P/"upstream.zip") as archive:\n    for name in archive.namelist():\n        assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    archive.extractall(P/"source")\nprint("Authenticated complete portable packet")');c.metadata['tags']=['data-payload'];cells.append(c)
    for name,code in definitions(P/'relkit/rdbpfn_l166.py'):
        if name in tasks:
            goal,contract,check=tasks[name]
            cells+=[nb.v4.new_markdown_cell('## TODO · '+name+'\n\n'+goal+'\n\n**Contract:** '+contract),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)]
        else:
            cells+=[nb.v4.new_markdown_cell('### PROVIDED · '+name+'\nVisible checkpoint-compatible computation. Trace the support boundary through this code.'),nb.v4.new_code_cell(code)]
    cells.append(nb.v4.new_code_cell('state=torch.load(P/"checkpoint.pt",map_location="cpu",weights_only=True)\nstate=state.get("model_state_dict",state)\nmodel=RDBPFN().eval()\nmodel.load_state_dict({k.removeprefix("module."):v for k,v in state.items()},strict=True)\nassert sum(p.numel() for p in model.parameters())==692738\nprint("Loaded real checkpoint; no training")'))
    for name,code in definitions(P/'_check_l166.py')+definitions(P/'_mechanism_l166.py'):
        cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('assert check166(relational_prior,dfs_summary,context_mask)=="PASS"\ntraces=mechanism166(relational_prior,dfs_summary,context_mask,model)\nreference=json.loads((P/"evidence/l166/mechanism.json").read_text())\nfor a,b in zip(traces["traces"],reference["traces"]):\n    np.testing.assert_allclose(a["query_probability"],b["query_probability"],atol=2e-5,rtol=2e-5)\n    print(a["strength"],a["support"],a["flip"],"row5 probability",round(a["query_probability"][-1],4))'))
    cells.append(nb.v4.new_markdown_cell('## Complete saved-evidence replay\nAll 30 runs and 21,060 predictions are checked below. No new benchmark inference occurs here. AUROC is independently recomputed using sorted negative scores and half-credit ties; the author additionally checked sklearn.'))
    for name,code in definitions(P/'_report_l166.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('packet=json.loads((P/"evidence/l166/predictions.json").read_text())\nreport=summarize166(packet)\nassert report==json.loads((P/"evidence/l166/report.json").read_text())\nPath("l166-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nfor arm,row in report["models"].items():print(arm,round(row["mean"],6),"SD",round(row["sample_sd"],6),row["status"])\nprint(report["predictions"],"predictions;",report["unique_test_queries"],"unique queries")'))
    cells.append(nb.v4.new_markdown_cell('## Optional · Fresh full selected inference\nThe complete runner and fetcher are visible below. Set RUN_FULL_REPRO=True only in an environment with torch 2.5.1, numpy 1.26.4, pandas 2.2.3, sklearn 1.6.1, pydantic 1.10.26, PyYAML 6.0.2 and tabicl 0.1.3. It downloads pinned model weights and evaluates all ten seeds for each arm. Default execution leaves it disabled, so no background paid work is launched. A cloud session may incur your provider’s charges; the author used the separately guarded $10 Modal lane. This reruns released-checkpoint inference, not pretraining.'))
    for name,code in definitions(P/'_fetch_l166.py')+definitions(P/'_run_l166.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('RUN_FULL_REPRO=False\nif RUN_FULL_REPRO:\n    import random,time\n    inputs=fetch166(P/"input")\n    fresh=run166(inputs,P/"source",P/"fresh-results",list(range(10)))\n    print("Fresh complete selected inference:",len(fresh["records"]),"runs")\nelse:\n    print("Fresh benchmark inference NOT_RUN in this notebook; complete author results replayed above.")'))
    cells.append(nb.v4.new_markdown_cell('## EXIT · Written defense\nWrite 400–600 words and four claim/evidence/limit rows. Explain prior, DFS and both attention axes; compare paired results; identify the label orientation and provenance limits. Five criteria scored 0–2: mechanism, fair inputs, uncertainty, provenance, bounded conclusion. Target ≥8/10, no zero,teacher review required. The sample below is a short illustration, not a completed defense. Ask the agent to review your own explanation.'))
    example='A synthetic dependency can survive child aggregation and become a DFS feature. Support-only row attention lets a frozen model use labeled examples without reading other query keys. The selected checkpoint replay matches all three rounded paper means, but ten support draws on one task do not establish broad superiority. Released labels have the opposite orientation from the current raw DNF rule. Full pretraining and historical preprocessing identity remain unestablished.' if solution else ''
    cells.append(nb.v4.new_code_cell('defense='+repr(example)+'\nPath("l166-submission.json").write_text(json.dumps(dict(defense=defense,review=None,learner="PENDING_WRITTEN_DEFENSE"),indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE")'))
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(cells):c.id=f'l166-{i:03d}'
    path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4)
        if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in book.cells]:
            book.metadata=old.metadata
            for c,prior in zip(book.cells,old.cells):
                if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
    nb.write(book,path)
print('Built L166 lesson, reference and portable notebooks')
