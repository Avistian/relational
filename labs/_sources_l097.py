"""Archive primary sources with byte identities and precise claim scope."""
import hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'sources/l097';D.mkdir(exist_ok=True)
specs=[('bpr.pdf','https://arxiv.org/pdf/1205.2618','BPR §4.1–4.3 and §5.1: pairwise objective and MF scoring; historical experiment NOT_RUN'),('sampled-metrics.pdf','https://www.ijcai.org/proceedings/2021/0651.pdf','Krichene and Rendle: sampled metrics can fail to preserve model comparisons'),('pyg-negative.py','https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.6.1/torch_geometric/utils/_negative_sampling.py','Tuple size selects bipartite universe; supplied positives excluded; global negative sampler'),('movielens-readme.txt','https://files.grouplens.org/datasets/movielens/ml-100k-README.txt','SUMMARY and u1.base through u5.test descriptions')]
rows=[]
for name,url,scope in specs:
 path=D/name
 if not path.exists():urllib.request.urlretrieve(url,path)
 rows.append({'url':url,'archived':str(path.relative_to(P)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':scope,'retrieved':'2026-09-20'})
result=json.loads((P/'_experiment_l097_results.json').read_text())
report={'sources':rows,'data_url':'https://files.grouplens.org/datasets/movielens/ml-100k.zip','archive_sha256':result['archive_sha256'],'member_sha256':result['member_sha256'],'published_experiment':'No model-paper target assigned in roadmap; full release reconstruction plus declared course ablation','paper_parity':'NOT_ESTABLISHED','BPR_paper_experiments':'NOT_RUN'}
(P/'_sources_l097.json').write_text(json.dumps(report,indent=2)+'\n');print('Archived and hashed',len(rows),'primary sources')
