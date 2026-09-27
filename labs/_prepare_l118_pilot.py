"""Bounded real Home Credit sample for resource calibration, NOT paper evaluation.
Scan every raw table; reconstruct source Cypher properties for selected applications.
Only the author-prepared full dataset is accepted by the named five-fold runner.
"""
import csv,hashlib,json,pickle,re,time,zipfile
from pathlib import Path
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent;SOURCE=P/'sources/l118/data/homecreditdefaultrisk';ROOT=P/'data/l118';OUT=ROOT/'pilot';OUT.mkdir(parents=True,exist_ok=True)
info=json.loads((SOURCE/'homecreditdefaultrisk.db_info.json').read_text())
archive=ROOT/'raw/home-credit-default-risk.zip'
# Fixed 1024-member sample of fold-0 training identities, selected without targets.
from relkit.cvitkovic_l118 import released_folds
ids=list(released_folds(info['train_dp_ids']))[0][0][:1024];wanted=set(map(int,ids))
cypher=re.sub(r'//[^\n]*','',(SOURCE/'homecreditdefaultrisk_neo4j_loader.cypher').read_text())
properties={}
for table,body in re.findall(r'CREATE \(\w+:(\w+)\s*\{(.*?)\}\)',cypher,re.S):
    parts=[];start=0;depth=0
    for i,c in enumerate(body):
        if c=='(':depth+=1
        if c==')':depth-=1
        if c==',' and depth==0:parts.append(body[start:i]);start=i+1
    parts.append(body[start:]);properties[table]={k.strip():re.sub(r'\s+','',v) for k,v in (p.split(':',1) for p in parts)}

def interpret(expr,row):
    if expr.startswith('-1*'):return None if interpret(expr[3:],row) is None else -interpret(expr[3:],row)
    if expr.startswith('row.'):return row.get(expr[4:],'')
    if expr=='null':return None
    if expr[0:1]=="'":return expr[1:-1]
    name,args=expr.split('(',1);args=args[:-1];parts=[];depth=0;start=0;quote=False
    for i,c in enumerate(args):
        if c=="'":quote=not quote
        if not quote:
            if c=='(':depth+=1
            if c==')':depth-=1
            if c==',' and depth==0:parts.append(args[start:i]);start=i+1
    parts.append(args[start:]);values=[interpret(p,row) for p in parts]
    if name=='replace':return values[0].replace(values[1],values[2])
    if name=='toBoolean':return {'true':True,'false':False}.get(str(values[0]).lower())
    if name in ['toFloat','toInteger']:
        try:return float(values[0]) if name=='toFloat' else int(float(values[0]))
        except (ValueError,TypeError):return None
    raise ValueError(expr)

files={'Application':'application_train.csv','Bureau':'bureau.csv','PreviousApplication':'previous_application.csv','CashBalance':'POS_CASH_balance.csv','CreditBalance':'credit_card_balance.csv','InstallmentPayment':'installments_payments.csv','BureauBalance':'bureau_balance.csv'}
rows={};census={};start=time.monotonic()
with zipfile.ZipFile(archive) as z:
    for table,filename in files.items():
        collected=[];count=0
        key='SK_ID_BUREAU' if table=='BureauBalance' else 'SK_ID_CURR'
        keys=set(int(r['SK_ID_BUREAU']) for r in rows['Bureau']) if table=='BureauBalance' else wanted
        with z.open(filename) as stream:
            for frame in pd.read_csv(stream,dtype=str,keep_default_na=False,chunksize=100000):
                count+=len(frame);mask=pd.to_numeric(frame[key],errors='coerce').isin(keys)
                collected.extend(frame[mask].to_dict('records'))
        rows[table]=collected;census[table]={'raw_rows_scanned':count,'selected_rows':len(collected)}
        print(table,census[table],flush=True)
# Assemble direct membership and all nine original relationship types.
records={i:[] for i in wanted};bureau_owner={int(r['SK_ID_BUREAU']):int(r['SK_ID_CURR']) for r in rows['Bureau']}
for table,rs in rows.items():
    for row in rs:
        owner=bureau_owner[int(row['SK_ID_BUREAU'])] if table=='BureauBalance' else int(row['SK_ID_CURR'])
        records[owner].append((table,row))
node_counts=[];edge_counts=[];mismatches=0
for identity,group in records.items():
    # Source requires feature order to follow node order within each type.
    mapping={};features={t:{f:[] for f in fs if t+'.'+f!=info['label_feature']} for t,fs in info['node_types_and_features'].items()};types=[];edges=[];etypes=[];label=None
    for j,(table,row) in enumerate(group):
        types.append(info['node_type_to_int'][table])
        pk={'Application':'SK_ID_CURR','Bureau':'SK_ID_BUREAU','PreviousApplication':'SK_ID_PREV'}.get(table)
        if pk:mapping[(table,int(row[pk]))]=j
        for f in features[table]:features[table][f].append(interpret(properties[table][f],row))
        if table=='Application':label=int(row['TARGET'])
    relationships={'Bureau':[('Application','SK_ID_CURR','BUREAU_TO_APPLICATION')],'BureauBalance':[('Bureau','SK_ID_BUREAU','BUREAUBALANCE_TO_BUREAU')],'PreviousApplication':[('Application','SK_ID_CURR','PREVIOUSAPPLICATION_TO_APPLICATION')],'CashBalance':[('Application','SK_ID_CURR','CASHBALANCE_TO_APPLICATION'),('PreviousApplication','SK_ID_PREV','CASHBALANCE_TO_PREVIOUSAPPLICATION')],'CreditBalance':[('Application','SK_ID_CURR','CREDITBALANCE_TO_APPLICATION'),('PreviousApplication','SK_ID_PREV','CREDITBALANCE_TO_PREVIOUSAPPLICATION')],'InstallmentPayment':[('Application','SK_ID_CURR','INSTALLMENTPAYMENT_TO_APPLICATION'),('PreviousApplication','SK_ID_PREV','INSTALLMENTPAYMENT_TO_PREVIOUSAPPLICATION')]}
    for j,(table,row) in enumerate(group):
        for dest,key,relation in relationships.get(table,[]):
            endpoint=mapping.get((dest,int(row[key])))
            if endpoint is not None:edges.append((j,endpoint));etypes.append(info['edge_type_to_int'][relation])
            elif dest=='PreviousApplication':mismatches+=1
    with (OUT/str(identity)).open('wb') as f:pickle.dump((edges,types,etypes,features,label),f)
    node_counts.append(len(types));edge_counts.append(len(edges))
report={'status':'PILOT_ONLY','n_graphs':len(ids),'selection':'first 1024 train IDs from released fold 0; target-blind','archive_sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),'census':census,'node_count_mean':float(np.mean(node_counts)),'node_count_max':max(node_counts),'stored_edge_count_mean':float(np.mean(edge_counts)),'missing_previous_references':mismatches,'seconds':time.monotonic()-start,'source_query_equivalence':'NOT_CHECKED against Neo4j; structural and property reconstruction only'}
(OUT/'ids.json').write_text(json.dumps(ids.tolist()));(P/'_data_l118_results.json').write_text(json.dumps(report,indent=2));print(report)
