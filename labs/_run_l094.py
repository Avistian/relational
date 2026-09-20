"""Fresh full released-graph audit; no sampled counts or model retraining."""
import argparse,json,time,platform,sys,urllib.request,os,resource
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hin_l094 import file_sha256,count_graph,table_arithmetic
from oag_read_l094 import load_oag
P=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--memory-gib',type=int,default=3);ap.add_argument('--dataset',choices=['NN','CS','OAG'],default='NN');ap.add_argument('--data',type=Path);ap.add_argument('--sha256');ap.add_argument('--download',action='store_true');ap.add_argument('--output',type=Path,default=P/'_paper_l094_results.json');a=ap.parse_args()
    if a.memory_gib<1:ap.error('Memory cap must be positive')
    resource.setrlimit(resource.RLIMIT_AS,(a.memory_gib*1024**3,a.memory_gib*1024**3))
    manifest=json.loads((P/'_sources_l094.json').read_text());table_path=P/'sources/hin-l094/table1.json'
    assert file_sha256(table_path)==manifest['paper']['table_sha256']
    assert file_sha256(P/'sources/hin-l094/survey.pdf')==manifest['paper']['sha256']
    table=json.loads(table_path.read_text());data=a.data or P/'data/l093/graph_NN.pk'
    if a.dataset!='NN' and (not a.data or not a.sha256):ap.error('Other rows need explicit full graph bytes and a verified hash; NN is not a substitute')
    digest=a.sha256 or manifest['nn_data']['sha256']
    if a.dataset=='NN' and digest!=manifest['nn_data']['sha256']:ap.error('NN requires the pinned complete released snapshot')
    if a.download:
        if a.dataset!='NN':ap.error('Only the verified NN download is automated')
        data.parent.mkdir(parents=True,exist_ok=True)
        if not data.exists():
            tmp=data.with_suffix('.part');urllib.request.urlretrieve(manifest['nn_data']['url'],tmp)
            if file_sha256(tmp)!=digest:raise ValueError('Downloaded bytes changed')
            tmp.replace(data)
    start=time.time();graph=load_oag(data,digest);stats=count_graph(graph)
    expected=table[a.dataset];actual=stats['table_columns']
    comparison={k:{'paper':expected[k],'released':actual[k],'delta':actual[k]-expected[k],'status':'MATCH' if actual[k]==expected[k] else 'MISMATCH'} for k in expected}
    result={'status':'FULL_RELEASE_GRAPH_AUDITED','dataset':a.dataset,'data_sha256':digest,'data_bytes':data.stat().st_size,'seconds':time.time()-start,'python':platform.python_version(),'implementation_sha256':file_sha256(P/'relkit/hin_l094.py'),'loader_sha256':file_sha256(P/'relkit/oag_read_l094.py'),'paper_sha256':manifest['paper']['sha256'],'table_sha256':manifest['paper']['table_sha256'],'statistics':stats,'comparison':comparison,'printed_table_arithmetic':{k:table_arithmetic(v) for k,v in table.items()},'paper_result_parity':'NOT_ESTABLISHED','historical_comparison':'INCOMPARABLE','scope':'Every node and stored adjacency of the named release; arithmetic only for other printed rows; no predictive training','fresh_graph_coverage':[a.dataset],'not_run':[k for k in table if k!=a.dataset]}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='statistics'},indent=2))
if __name__=='__main__':main()
