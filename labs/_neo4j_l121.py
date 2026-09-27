"""Execute original Neo4j 3.5 loader and compare 32 real graph multisets.
Run once on a fresh isolated database; queries contain no record content in logs.
"""
import argparse,collections,copy,hashlib,json,pickle,re,time,urllib.request
import torch
from relkit.cvitkovic_l118 import encode_features
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P/'data/l121'
URL='http://127.0.0.1:17474/db/data/transaction/commit'
def query(statement,graph=False):
 payload={'statements':[{'statement':statement,'resultDataContents':['graph'] if graph else ['row']}]}
 req=urllib.request.Request(URL,json.dumps(payload).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as response:data=json.load(response)
 if data['errors']:raise RuntimeError(data['errors'])
 return data['results'][0]['data']
def signature(table,props,info):
 fields=info['node_types_and_features'][table]
 return json.dumps([table,{f:props.get(f) for f in fields if table+'.'+f!=info['label_feature']}],sort_keys=True,separators=(',',':'))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--compare-only',action='store_true');args=parser.parse_args()
 start=time.monotonic();loader=(ROOT/'loader.cypher').read_text();loader=re.sub(r'^\s*//[^\n]*','',loader,flags=re.M)
 for i,statement in enumerate(loader.split(';')):
  if statement.strip() and not args.compare_only:query(statement);print('Loaded statement',i,flush=True)
 info=json.loads((P/'sources/l118/data/homecreditdefaultrisk/homecreditdefaultrisk.db_info.json').read_text());ids=json.loads((ROOT/'ids.json').read_text());records=[];raw_mismatch_graphs=0;empty_cells=0
 verified=ROOT/'verified';verified.mkdir(exist_ok=True)
 reverse={v:k for k,v in info['node_type_to_int'].items()};et={v:k for k,v in info['edge_type_to_int'].items()}
 for identity in ids:
  data=query(f'MATCH r = (a:Application)-[*0..2]-(n) WHERE a.SK_ID_CURR = {int(identity)} RETURN a,r,n',True)
  nodes={};rels={}
  for row in data:
   for node in row['graph']['nodes']:nodes[node['id']]=node
   for rel in row['graph']['relationships']:rels[rel['id']]=rel
  sigs={key:signature(n['labels'][0],n['properties'],info) for key,n in nodes.items()}
  ns=collections.Counter(sigs.values());es=collections.Counter((sigs[e['startNode']],sigs[e['endNode']],e['type']) for e in rels.values())
  with (P/'data/l118/pilot'/str(identity)).open('rb') as f:edges,types,ets,features,label=pickle.load(f)
  old=copy.deepcopy(features)
  raw_node_sigs=[];pos=collections.Counter()
  for typ in types:
   table=reverse[typ];idx=pos[table];pos[table]+=1;raw_node_sigs.append(signature(table,{k:v[idx] for k,v in features[table].items()},info))
  raw_mismatch_graphs+=ns!=collections.Counter(raw_node_sigs)
  # Neo4j LOAD CSV parses empty cells as null; pandas keep_default_na=False did not.
  for table,fields in features.items():
   for name,values in fields.items():
    empty_cells+=sum(v=='' for v in values);features[table][name]=[None if v=='' else v for v in values]
  before,after=encode_features(old,info),encode_features(features,info)
  for table in before:
   for a,b in zip(before[table],after[table]):assert torch.equal(a,b),'Missing-value correction changed encoded input'
  pos=collections.Counter();ps=[]
  for typ in types:
   table=reverse[typ];idx=pos[table];pos[table]+=1;ps.append(signature(table,{k:v[idx] for k,v in features[table].items()},info))
  pn=collections.Counter(ps);pe=collections.Counter((ps[u],ps[v],et[t]) for (u,v),t in zip(edges,ets))
  assert ns==pn,('node feature multiset mismatch',identity,len(ns),len(pn))
  assert es==pe,('edge multiset mismatch',identity,len(es),len(pe))
  assert [n['properties'].get('TARGET') for n in nodes.values() if n['labels']==['Application']]==[label]
  with (verified/str(identity)).open('wb') as f:pickle.dump((edges,types,ets,features,label),f)
  records.append({'nodes':len(types),'edges':len(edges)})
 report={'status':'PASS','engine':'Neo4j 3.5.4 original jars on native Temurin Java8 aarch64','image_digest':'sha256:1eb754a81e7e48431be4f846dfc5ab820e7b7c841d99e7604ef62c3420a2a886','graphs':len(records),'raw_reconstruction_mismatch_graphs':raw_mismatch_graphs,'empty_to_null_cells_corrected':empty_cells,'correction':'CSV empty string becomes Neo4j null; encoded tensors unchanged exactly','native_java_image':'eclipse-temurin:8-jre@sha256:66a7b358772dfdcc1486919a5ede66d6c7275f203a5fefddb39565c2a151a485','nodes':sum(r['nodes'] for r in records),'edges':sum(r['edges'] for r in records),'comparison':'exact type-feature node multisets and directed typed edge multisets, labels','loader_change':'file URI relocation only','seconds':time.monotonic()-start,'boundary':'32 target-blind applicants in referentially audited subset; duplicate identical rows compared as multisets, not global graph isomorphism; full-population equivalence NOT_CHECKED'}
 (P/'_neo4j_l121_results.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
