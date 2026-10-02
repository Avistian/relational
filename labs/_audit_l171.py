"""Visible preparation and replay helpers; no upstream source is executed."""
def source_inventory(root, validate):
    import ast
    import hashlib
    import json
    import re
    from pathlib import Path
    root=Path(root); base=root/'sources/l171/relbench/datasets'
    hashes=json.loads((base/'hashes.json').read_text())
    records=[]
    for short in ['amazon','avito','event','f1','hm','stack','trial']:
        path=base/(short+'.py'); text=path.read_text();tree=ast.parse(text)
        schema={};boundaries={}
        def table_call(node):
            return isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Table'
        def add(name,call):
            if name in schema:raise ValueError('Ambiguous table definition')
            fields={k.arg:ast.literal_eval(k.value) for k in call.keywords if k.arg in ['pkey_col','time_col','fkey_col_to_pkey_table']}
            schema[name]=dict(pkey_col=fields.get('pkey_col'),time_col=fields.get('time_col'),fkey_col_to_pkey_table=fields.get('fkey_col_to_pkey_table',{}))
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in ['val_timestamp','test_timestamp']:
                        boundaries[target.id]=ast.literal_eval(node.value.args[0])
                    if isinstance(target,ast.Subscript) and table_call(node.value):
                        add(ast.literal_eval(target.slice),node.value)
            if isinstance(node,ast.Dict):
                for key,value in zip(node.keys,node.values):
                    if table_call(value):add(ast.literal_eval(key),value)
        calls=sum(table_call(n) for n in ast.walk(tree))
        if not schema or len(schema)!=calls or len(boundaries)!=2:
            raise ValueError('Incomplete static extraction: '+short)
        for spec in schema.values():
            if not set(spec['fkey_col_to_pkey_table'].values())<=set(schema):
                raise ValueError('Declared FK table absent: '+short)
        database='rel-'+short
        records.append(dict(database=database,source_families=['declared-original:'+short],
            family_evidence='CURATOR_DECLARATION_NOT_EXHAUSTIVE_LINEAGE',
            archive_sha256=hashes[database+'/db.zip'],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_path=str(path.relative_to(root)),
            upstream_urls=sorted(set(re.findall(r'https?://[^\s\"\'<>]+',text))),
            tables=dict(sorted(schema.items())),**boundaries,
            row_audit='FULL_SNAPSHOT' if short=='f1' else 'NOT_RUN',
            rights='REVIEW_REQUIRED',historical_availability='NOT_ESTABLISHED'))
    return validate(records)


def load_f1(root):
    import json
    from pathlib import Path
    import pyarrow.parquet as pq
    result={}
    for path in sorted((Path(root)/'evidence/l171/db').glob('*.parquet')):
        table=pq.read_table(path)
        spec={name:json.loads(table.schema.metadata[name.encode()]) for name in ['pkey_col','time_col','fkey_col_to_pkey_table']}
        result[path.stem]=dict(df=table.to_pandas(),**spec)
    return result


def replay171(root, manifest, validate, split, audit):
    import copy
    import hashlib
    import json
    import zipfile
    from pathlib import Path
    root=Path(root)
    for name,digest in manifest['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe manifest path')
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch: '+name)
    inventory=source_inventory(root,validate)
    f1=next(r for r in inventory if r['database']=='rel-f1')
    archive=root/'evidence/l171/rel-f1-db.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=f1['archive_sha256']:raise ValueError('Archive/registry mismatch')
    tables=load_f1(root)
    if set(tables)!=set(f1['tables']):raise ValueError('Incomplete F1 table set')
    with zipfile.ZipFile(archive) as z:
        names={n for n in z.namelist() if n.endswith('.parquet')}
        if names!={'db/'+n+'.parquet' for n in tables}:raise ValueError('Archive population mismatch')
        for name,spec in tables.items():
            if z.read('db/'+name+'.parquet')!=(root/'evidence/l171/db'/(name+'.parquet')).read_bytes():raise ValueError('Archive member mismatch')
            if {k:spec[k] for k in f1['tables'][name]}!=f1['tables'][name]:raise ValueError('Source/parquet schema mismatch')
    row_audit=audit(tables,f1['val_timestamp'],f1['test_timestamp'])
    splits={row['database']:split(inventory,row['database']) for row in inventory}
    # Deliberately synthetic aliases and a two-hop bridge, never real source claims.
    contaminated=copy.deepcopy(inventory)
    alias=copy.deepcopy(f1);alias.update(database='synthetic-f1-copy',source_families=['synthetic-copy'])
    bridge=copy.deepcopy(inventory[0]);bridge.update(database='synthetic-bridge',source_families=['synthetic-copy','synthetic-tail'],archive_sha256='1'*64)
    tail=copy.deepcopy(inventory[0]);tail.update(database='synthetic-tail',source_families=['synthetic-tail'],archive_sha256='2'*64)
    contaminated.extend([alias,bridge,tail])
    return dict(experiment=manifest['experiment'],status='COMPLETE_DECLARED_CORPUS_AUDIT',
        inventory=inventory,splits=splits,contamination_demo=split(contaminated,'rel-f1'),
        full_snapshot=row_audit,inventory_databases=len(inventory),
        declared_tables=sum(len(r['tables']) for r in inventory),
        declared_foreign_key_columns=sum(len(s['fkey_col_to_pkey_table']) for r in inventory for s in r['tables'].values()),
        inventory_validation='PASS',split_validation='PASS',
        source_family_exclusion='DECLARED_LINEAGE_ONLY',other_six_row_audits='NOT_RUN',
        deployment_rights='REVIEW_REQUIRED',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',
        transfer='NOT_ESTABLISHED',historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.corpus_l171 import validate_corpus,split_corpus,audit_tables
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'evidence/l171/input-manifest.json').read_text())
    report=replay171(root,manifest,validate_corpus,split_corpus,audit_tables)
    (root/'evidence/l171/report.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'evidence/l171/corpus-manifest.json').write_text(json.dumps(report['inventory'],indent=2)+'\n')
    print(report['status'],report['inventory_databases'],'databases;',report['declared_tables'],'declared tables;',report['full_snapshot']['rows'],'F1 rows;',report['full_snapshot']['integrity'])
