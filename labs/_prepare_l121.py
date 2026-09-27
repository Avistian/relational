"""Filter target-blind real applicant audit fixture; scan full previous-key universe.
No licensed records are published. Reuses L118's trusted local pilot identities.
"""
import csv,hashlib,json,time,zipfile
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P/'data/l121';OUT=ROOT/'neo4j-input';OUT.mkdir(parents=True,exist_ok=True)
ids=json.loads((P/'data/l118/pilot/ids.json').read_text())[:32];wanted=set(ids)
archive=P/'data/l118/raw/home-credit-default-risk.zip'
files=['application_train.csv','application_test.csv','bureau.csv','previous_application.csv','POS_CASH_balance.csv','credit_card_balance.csv','installments_payments.csv','bureau_balance.csv']
selected={};census={};previous={};start=time.monotonic()
with zipfile.ZipFile(archive) as z:
 for filename in files:
  count=0;parts=[]
  with z.open(filename) as stream:
   for df in pd.read_csv(stream,dtype=str,keep_default_na=False,chunksize=100000):
    count+=len(df)
    if filename=='previous_application.csv':previous.update(zip(df.SK_ID_PREV.astype(int),df.SK_ID_CURR.astype(int)))
    key='SK_ID_BUREAU' if filename=='bureau_balance.csv' else 'SK_ID_CURR'
    keys=set(selected['bureau.csv'].SK_ID_BUREAU.astype(int)) if key=='SK_ID_BUREAU' else wanted
    parts.append(df[pd.to_numeric(df[key],errors='coerce').isin(keys)])
  result=pd.concat(parts);selected[filename]=result;result.to_csv(OUT/filename,index=False)
  census[filename]={'raw_rows':count,'selected_rows':len(result)};print(filename,census[filename],flush=True)
missing=0;cross_owner=0;references=0
for f in ['POS_CASH_balance.csv','credit_card_balance.csv','installments_payments.csv']:
 for r in selected[f].itertuples():
  references+=1;owner=previous.get(int(r.SK_ID_PREV));missing+=owner is None;cross_owner+=owner is not None and owner!=int(r.SK_ID_CURR)
assert cross_owner==0,'Subset lacks cross-owner neighbors: extend closure before comparison'
source=P/'sources/l118/data/homecreditdefaultrisk/homecreditdefaultrisk_neo4j_loader.cypher'
# Exact released loader, apart from URI relocation into isolated container import directory.
loader=source.read_text().replace('file:///data/','file:///')
(ROOT/'loader.cypher').write_text(loader)
report={'status':'PASS_RAW_KEY_AUDIT','sample_graphs':len(ids),'selection':'first 32 L118 target-blind fold-0 train identities','full_previous_key_count':len(previous),'payment_previous_references':references,'globally_missing_previous_references':missing,'cross_owner_references':cross_owner,'census':census,'raw_sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),'loader_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seconds':time.monotonic()-start,'boundary':'32 applicants, full previous-key universe; not full-population graph equivalence'}
(ROOT/'ids.json').write_text(json.dumps(ids));(P/'_data_l121_results.json').write_text(json.dumps(report,indent=2));print(report)
