"""Full ML-100K release audit and five-fold course ranking baseline."""
import argparse,hashlib,importlib.metadata,json,os,platform,time,zipfile
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from relkit.bipartite_l095 import read_release,audit_release,run_experiment,sha256,ARCHIVE_URL,ARCHIVE_SHA256
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=P/'data/l095/ml-100k.zip');args=ap.parse_args();start=time.time()
arrays,hashes=read_release(args.data);audit=audit_release(arrays)
result=run_experiment(arrays,P/'results/l095')
result.update(release_audit=audit,archive_sha256=ARCHIVE_SHA256,file_sha256=hashes,source_sha256=sha256(P/'relkit/bipartite_l095.py'),seconds=time.time()-start,environment={'python':platform.python_version(),'machine':platform.machine(),**{k:importlib.metadata.version(k) for k in ['numpy','torch','torch-geometric']},'threads':1})
(P/'_experiment_l095_results.json').write_text(json.dumps(result,indent=2)+'\n')
manifest={'archive':{'url':ARCHIVE_URL,'sha256':ARCHIVE_SHA256,'redistribution':'Download from GroupLens; archive remains untracked'},'members':hashes,'primary_sources':[{'url':'https://files.grouplens.org/datasets/movielens/ml-100k-README.txt','locator':'SUMMARY and DETAILED DESCRIPTIONS: u.data, u1.base through u5.test'},{'url':'https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html','locator':'typed node and edge stores'},{'url':'https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.transforms.RandomLinkSplit.html','locator':'rev_edge_types; is_undirected ignored for bipartite edge types'}],'retrieved':'2026-09-20','implementation':'Original course walk and evaluator; no external model source is claimed'}
(P/'_sources_l095.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'audit':audit,'summary':result['summary'],'seconds':result['seconds']},indent=2))
