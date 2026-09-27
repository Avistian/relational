"""Pin primary code and probe original archives. No training or cloud spending."""
import concurrent.futures,hashlib,json,subprocess,urllib.request,urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent
specs=[('relbench','/tmp/l125-relbench-upstream','b20c72d',['LICENSE','relbench/__init__.py','relbench/datasets/stackex.py','relbench/tasks/stackex.py','relbench/data/dataset.py','relbench/data/task_node.py','relbench/external/graph.py','relbench/external/nn.py','examples/model.py','examples/gnn_node.py','examples/inferred_stypes.py','examples/text_embedder.py']),('frame','/tmp/l125-frame-upstream','d998aae368db6a4e36139ccc56bd54579a70874b',['LICENSE','torch_frame/nn/models/resnet.py','torch_frame/nn/encoder/stype_encoder.py','torch_frame/nn/encoder/stypewise_encoder.py','torch_frame/data/dataset.py','torch_frame/data/mapper.py','torch_frame/data/stats.py','torch_frame/_stype.py'])]
manifest={'paper':'https://arxiv.org/html/2404.00776v2','selection':'Table 2 rel-stackex-engage; ROC-AUC .854','files':{},'revisions':{}}
for name,repo,rev,files in specs:
 rev=subprocess.check_output(['git','rev-parse',rev],cwd=repo,text=True).strip();manifest['revisions'][name]=rev
 for file in files:
  raw=subprocess.check_output(['git','show',rev+':'+file],cwd=repo)
  dest=P/'sources/l125'/name/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
  gh='snap-stanford/relbench' if name=='relbench' else 'pyg-team/pytorch-frame'
  manifest['files'][str(dest.relative_to(P))]={'sha256':hashlib.sha256(raw).hexdigest(),'url':f'https://raw.githubusercontent.com/{gh}/{rev}/{file}'}
manifest['source_boundary']='Contemporaneous RelBench snapshot; not an author-attested Table 2 commit. Teaching Frame pinned to 0.3.0; not historical binary identity.'
(P/'_sources_l125.json').write_text(json.dumps(manifest,indent=2)+'\n')
urls=['https://relbench.stanford.edu/staging_data/rel-stackex/db.zip','https://relbench.stanford.edu/staging_data/rel-stackex/tasks/rel-stackex-engage.zip','https://relbench.stanford.edu/data/relbench-forum-raw.zip','https://relbench.stanford.edu/data/rel-stackex/db.zip','https://relbench.stanford.edu/download/rel-stackex/db.zip','https://relbench.stanford.edu/download/rel-stackex/tasks/rel-stackex-engage.zip']
def probe(url):
 try:
  with urllib.request.urlopen(url,timeout=30) as r:
   return {'url':url,'status':r.status,'final_url':r.url,'content_type':r.headers.get('Content-Type'),'prefix_hex':r.read(8).hex()}
 except urllib.error.HTTPError as e:return {'url':url,'status':e.code,'final_url':e.url}
 except Exception as e:return {'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:probes=list(ex.map(probe,urls))
audit={'date':'2026-09-27','status':'AUDITED','archive_probes':probes,'expected_sha256':{'db.zip':'deb00ccdf825e569b34935834444429cd1c0074b50226b12d616aab22d36242d','engage.zip':'9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18','raw.zip':'ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f'},'paper_result':'NOT_RUN','historical_identity':'NOT_ESTABLISHED','protocol_gaps':['No author-attested Table 2 source commit or seed ensemble','Contemporaneous example regenerates tasks: archived labels not guaranteed identical','Historical package environment and text-model revision not pinned by paper','Original archives unavailable at probed URLs; no checksum-matching mirror recovered'],'spend_usd':0}
(P/'_paper_audit_l125_results.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
