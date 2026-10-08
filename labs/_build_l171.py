"""Generate a self-contained lesson, reference and portable corpus audit notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0171-corpus-of-databases';E=P/'evidence/l171'
r=json.loads((E/'report.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
def definitions(path):
    s=path.read_text();return [(n.name,ast.get_source_segment(s,n)) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)]
def inventory_table():
    lines=['| Database | Declared tables | FK columns | Row evidence |','|---|---:|---:|---|']
    for x in r['inventory']:lines.append(f"| {x['database']} | {len(x['tables'])} | {sum(len(t['fkey_col_to_pkey_table']) for t in x['tables'].values())} | {'Complete F1 snapshot' if x['database']=='rel-f1' else 'Source inventory only'} |")
    return '\n'.join(lines)
def f1_table():
    lines=['| Table | Rows | PK nulls / duplicate excess | Time column |','|---|---:|---:|---|']
    for n,t in r['full_snapshot']['tables'].items():lines.append(f"| {n} | {t['rows']:,} | {t['pk_nulls']} / {t['pk_duplicate_excess']} | {t['time_col'] or 'None'} |")
    return '\n'.join(lines)+'\n\nAcross 227,716 non-null FK references: **0 dangling, 0 null FK values**. This is measured archive integrity; historical availability remains unestablished.'
captions={
'holdout':'Illustrative source-lineage chain. Starting with held-out F1, exclude its identical copy, then the bridge, then the tail. Six original declared families remain candidates. Synthetic additions are not actual RelBench contamination.',
'foreign-keys':'Measured full F1 snapshot. Each filled cell counts non-null references from the row table to the column table. All 13 declared FK columns resolve; 227,716 references are not 227,716 distinct entities.',
'time-windows':'Measured full F1 snapshot. Bars split each time-bearing table at 2005-01-01 and 2010-01-01. Labels show row counts. Three other tables lack time columns; event dates do not establish historical availability.'}
def prose(portable=False):
    s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[INVENTORY]]',inventory_table()).replace('[[F1RESULTS]]',f1_table())
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l171'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l171/'+name+'.svg'
        s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0" style="overflow-x:auto"><img style="min-width:760px;width:100%" src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on a narrow screen.</figcaption></figure>')
    widgets={
'WARMUP':('Recall without looking: why is a support draw not a new database? Which timestamp constrains a historical query?','<div id="warmup"></div>'),
'PREDICT':('Before reading on: should the tail corpus be excluded even though its hash differs from the held-out original? Write your prediction.','<div id="predict"></div>'),
'EXPLORER':('In the notebook, change the held-out database and compare the clean inventory with the synthetic copy/bridge/tail intervention. Six original candidates remain; three synthetic relatives are quarantined.','<div id="corpus-explorer"></div><noscript>Clean inventory: each of seven declared database holdouts leaves six candidates. The synthetic copy/bridge/tail intervention adds three quarantined relatives. The portable notebook computes this without JavaScript.</noscript>'),
'TEACHBACK':('Write your defense before reading a reference outline. Explain why clean keys cannot certify historical availability. Ask the teacher to review it.','<div id="teachback"></div>')}
    for key,(plain,html) in widgets.items():s=s.replace('[['+key+']]',plain if portable else html)
    s=s.replace('[[CODE]]','The live TODO below implements the connected-component exclusion rule.' if portable else '''```python
# Worked integrity example; full implementation is visible in the notebook.
parents = {10, 20}
children = [10, 10, None, 99]
missing = sum(x is None for x in children)
dangling = sum(x is not None and x not in parents for x in children)
assert (missing, dangling) == (1, 1)
```''')
    if portable:
        s=s.replace('](../','](https://avistian.github.io/relational/')
        s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
    return s

def doc(title,s,interactive=False):
    body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(s)).replace('</table>','</table></div>')
    css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','corpus-holdout']
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','corpus-holdout','l171-evidence','l171-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0170-fm-design-checkpoint.html">Lesson 170</a></nav><header><p class="route-kicker">Year 5 · Quarter 2 · Lesson 171</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Corpus of databases',prose(),True))
(R/'assets/l171-evidence.js').write_text('window.L171Inventory='+json.dumps([{k:x[k] for k in ['database','source_families','archive_sha256']} for x in r['inventory']],separators=(',',':'))+';\n')
reference='''**One skill:** freeze a corpus and defend its database-level exclusions before pretraining.

| Term | Operational meaning |
|---|---|
| Snapshot | One recorded database version |
| Manifest | Versioned inventory, hashes, source declarations and use rules |
| Source family | Declared shared ancestry; names alone do not establish independence |
| Holdout | Database reserved for evaluation |
| Quarantine | Source relatives excluded from training pending provenance review |
| Dangling FK | Non-null reference absent from the target primary key |
| Availability time | When a fact could be used, not merely when its event occurred |

**Exclusion rule:** connect records with overlapping source families OR identical archive hashes. Exclude the entire connected component of the held-out database. Keep evaluation identity separate from quarantined relatives. Unknown lineage remains unknown.

**Integrity rule:** count all primary-key nulls and duplicate excess; count null foreign keys separately from non-null orphans. Require every declared target table and column to exist. Do not drop failing rows to make an audit pass.

**Time rule:** describe full-snapshot windows; do not relabel them task splits. Missing time columns and unobserved arrival histories cannot establish point-in-time validity.

'''+inventory_table()+'''

**Executed scope:** all seven pinned RelBench1.1.0 source definitions (50 tables,62 FK columns); complete F1 snapshot (97,606 rows,9 tables,13 FK columns,227,716 non-null references); seven clean holdout splits plus synthetic contamination checks. Full F1 keys pass. Other six row audits and fresh pretraining NOT_RUN; availability, unknown lineage and transfer NOT_ESTABLISHED; data rights REVIEW_REQUIRED. No model benchmark is reproduced.

**Corpus card:** name the holdout; list admitted candidates and quarantined relatives; attach byte/source hashes; state observed vs declared evidence; record rights and availability gaps; name the next audit that could change inclusion.

[Lesson](../lessons/0171-corpus-of-databases.html) · [Notebook](../labs/0171-corpus-of-databases.ipynb) · [Protocol](../labs/l171-reproduction.md) · [Manifest](../labs/evidence/l171/corpus-manifest.json) · [RelBench paper](https://arxiv.org/html/2407.20060v1).
'''
(R/'reference/corpus-of-databases.html').write_text(doc('Corpus contracts — quick reference',reference))
(E/'report.md').write_text('# L171 RelBench Corpus and Holdout Audit\n\n'+inventory_table()+'\n\n'+f1_table()+'\n\nComplete declared audit. Other six row audits and fresh pretraining NOT_RUN; availability/transfer NOT_ESTABLISHED; source-family exclusion covers declared lineage only.\n')
# Fixed zip metadata keeps notebook bytes deterministic.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(manifest['files']):
        info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
contracts={
'validate_corpus':('Freeze valid records',
'''Accept a nonempty list of record dictionaries. Require unique nonempty database strings, a nonempty list of distinct nonempty source-family strings, and a lowercase 64-character hexadecimal archive SHA256. Raise ValueError on violations. Return an independent deep copy sorted by database, with each source-family list sorted; retain additional fields unchanged. Do not modify the caller's records.''',
'''demo=[dict(database="z",source_families=["origin-z"],archive_sha256="a"*64),dict(database="a",source_families=["origin-a"],archive_sha256="b"*64)]
assert [r["database"] for r in validate_corpus(demo)]==["a","z"]
assert demo[0]["database"]=="z"
print("CHECK: deterministic inventory, original untouched")'''),
'split_corpus':('Exclude the connected source family',
'''Call validate_corpus. Reject an unknown heldout name with ValueError. Link records when any source-family identifier matches OR the archive hashes match. Follow links transitively. Return train (sorted unconnected names), heldout (the single requested name), quarantine (sorted connected names except the requested one). This is a conservative declared-lineage rule, not a claim that all connected records contain the same rows.''',
'''example=[dict(database=n,source_families=f,archive_sha256=h*64) for n,f,h in [("a",["x"],"a"),("b",["x","y"],"b"),("c",["y"],"c"),("d",["z"],"d")]]
assert split_corpus(example,"a")==dict(train=["d"],heldout=["a"],quarantine=["b","c"])
print("CHECK: transitive bridge excluded")'''),
'audit_tables':('Audit every declared row relationship',
'''Input maps table names to df (pandas DataFrame), pkey_col, time_col and fkey_col_to_pkey_table. Also accept ordered finite validation and test timestamp strings. Reject empty tables, reversed/invalid boundaries and missing declared columns or FK targets with ValueError.

Return tables (sorted name to stats), foreign_keys (sorted table/column order), rows (sum), foreign_key_columns (count), integrity (PASS unless PK nulls/duplicates or dangling FKs), availability (always NOT_ESTABLISHED). Per-table stats: rows, columns, pkey, pk_nulls, pk_duplicate_excess (repetitions beyond the first non-null key), time_col. With no PK both PK counts are None. With no clock set time_status=NO_TIME_COLUMN, time_windows=None. Otherwise use time_status=OBSERVED_EVENT_TIME; time_min/time_max are string timestamps or None; time_windows has before_val, val_to_test, at_or_after_test, nulls. Use <val, [val,test), >=test. Each FK entry: table,column,target,rows,nulls,dangling,nonnull. Null FKs alone do not fail integrity. Do not remove any rows.''',
'''import pandas as pd
example={"p":dict(df=pd.DataFrame({"id":[10,20]}),pkey_col="id",time_col=None,fkey_col_to_pkey_table={}),"c":dict(df=pd.DataFrame({"parent":[10,10,None,99]}),pkey_col=None,time_col=None,fkey_col_to_pkey_table={"parent":"p"})}
observed=audit_tables(example,"2005-01-01","2010-01-01")
assert observed["foreign_keys"][0]["nulls"]==1
assert observed["foreign_keys"][0]["dangling"]==1
assert observed["integrity"]=="FAIL" and observed["availability"]=="NOT_ESTABLISHED"
print("CHECK: null references are not dangling references")''')}
for solution in [False,True]:
    cells=[nb.v4.new_markdown_cell('# Lesson 171 · Corpus of databases\n\n**Skill:** produce an auditable corpus manifest and defend its holdout boundary. Tier B real relational data; synthetic examples only test failure modes. This portable notebook embeds the full small F1 archive and seven source definitions. It needs Python3, pandas and pyarrow; no RelBench installation, network or cloud execution. PROVIDED is scaffolding, TODO is your implementation, CHECK is feedback, EXIT is your written defense.'),nb.v4.new_markdown_cell(prose(True)),
      nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport base64,hashlib,io,json,zipfile\nfrom pathlib import Path\ntry:\n    import pandas as pd\n    import pyarrow.parquet as pq\nexcept ImportError as exc:\n    raise RuntimeError("Install pandas==3.0.3 and pyarrow==24.0.0, then rerun this cell") from exc\nP=Path("l171-portable");P.mkdir(exist_ok=True)')]
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Authenticate exact evidence bytes\nThe payload holds the nine complete F1 tables, their archive and pinned source definitions. The input manifest records each file hash. This does not authenticate historical paper identity or hidden lineage. Data payload is hidden only in prepared HTML for readability; it remains in this notebook.'))
    c=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as archive:\n    for name in archive.namelist():\n        assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    archive.extractall(P)\nmanifest='+repr(manifest)+'\nprint("Exact source and complete F1 archive packet authenticated")');c.metadata['tags']=['data-payload'];cells.append(c)
    for name,code in definitions(P/'relkit/corpus_l171.py'):
        title,contract,check=contracts[name]
        cells += [nb.v4.new_markdown_cell('## TODO · '+title+'\n\n**Why:** this operation controls the final real-data audit. The CHECK is deliberately small; a broader suite follows.\n\n**Contract:** '+contract),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)]
    cells.append(nb.v4.new_markdown_cell('## CHECK · Challenge the live functions\nThese checks reject invalid manifests, renamed copies, transitive overlap, orphan references, duplicate and missing primary keys, absent targets and reversed time bounds. They call the functions you just wrote.'))
    for name,code in definitions(P/'_check_l171.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('assert check171(validate_corpus,split_corpus,audit_tables)=="PASS"\nprint("Live corpus contracts PASS")'))
    labels={'source_inventory':'Read source declarations without executing source code','load_f1':'Read every actual F1 table and its key metadata','replay171':'Authenticate, join the declared schema to actual rows, and run the complete audit'}
    for name,code in definitions(P/'_audit_l171.py'):
        cells += [nb.v4.new_markdown_cell('## PROVIDED · '+labels[name]+'\nThis visible scaffolding calls your validation, split and audit functions. It retains source-only evidence separately from observed row checks.'),nb.v4.new_code_cell(code)]
    cells.append(nb.v4.new_code_cell('report=replay171(P,manifest,validate_corpus,split_corpus,audit_tables)\nassert report["inventory_databases"]==7 and report["full_snapshot"]["rows"]==97606\nPath("l171-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nprint(report["status"],"| F1 integrity:",report["full_snapshot"]["integrity"])\ndisplay(pd.DataFrame([{ "database":x["database"], "declared_tables":len(x["tables"]), "row_audit":x["row_audit"]} for x in report["inventory"]]))\ndisplay(pd.DataFrame(report["full_snapshot"]["tables"]).T)\nprint("Deliberate synthetic intervention:",report["contamination_demo"])\nprint("Historical availability:",report["full_snapshot"]["availability"],"| fresh pretraining:",report["fresh_pretraining"])'))
    cells += [nb.v4.new_markdown_cell('## Intervention · Change a declared source relationship\nPredict which split changes before executing. This intentionally false family declaration is a teaching intervention; it does not amend the real corpus or report. Explain why the result is conservative.'),nb.v4.new_code_cell('import copy\nvariant=copy.deepcopy(report["inventory"])\na=next(x for x in variant if x["database"]=="rel-amazon")\nf=next(x for x in variant if x["database"]=="rel-f1")\na["source_families"].extend(f["source_families"])\nprint("Original:",split_corpus(report["inventory"],"rel-f1"))\nprint("Hypothetical shared source:",split_corpus(variant,"rel-f1"))\nassert split_corpus(variant,"rel-f1")["quarantine"]==["rel-amazon"]'),
    nb.v4.new_markdown_cell('## EXIT · Your corpus card\nWrite 200–400 words. Name the holdout, list training candidates and quarantined relatives, and distinguish declared schema, observed row integrity and unobserved availability. Explain why a different hash cannot prove independence. Identify one concrete new audit that could change inclusion. Code checks plus teacher-reviewed reasoning are required; no mastery is inferred from author execution.'),
    nb.v4.new_code_cell('corpus_card={name:"" for name in ["holdout","candidates","quarantine","identity_evidence","availability_gap","next_audit"]}\nsubmission=dict(corpus_card=corpus_card,report_sha256=hashlib.sha256(Path("l171-report.json").read_bytes()).hexdigest(),teacher_review=None,learner="PENDING_WRITTEN_DEFENSE")\nPath("l171-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Submit the corpus card and report to the teacher. Learner PENDING_WRITTEN_DEFENSE.")'),
    nb.v4.new_markdown_cell('## Reproduction boundary\nThis is the complete approved corpus-engineering audit. Other six database row audits, model pretraining and paper performance reproduction remain NOT_RUN. The executable operator is this notebook or `_budget_l171.py ... _audit_l171.py`; see the linked protocol. A training script would answer a later lesson’s question and is not silently substituted here. No cloud dispatch is enabled. Live Colab and deployment remain NOT_CHECKED.')]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(cells):c.id=f'l171-{i:03d}'
    path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4);book.metadata=old.metadata
        previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
        if [c.source for c in previous]==[c.source for c in current]:
            for prior,c in zip(previous,current):
                c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
    nb.write(book,path)
print('Built lesson, reference, report and both portable notebooks')
