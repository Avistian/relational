"""Fresh full F1 replay; inspect every temporal batch without changing the trainer."""
import json,time,hashlib
from pathlib import Path
import relkit.rdl_l117 as model_module
import _run_l117 as original_runner
from relkit.scale_l134 import inspect_batch,frontier_bound

def run(seed,epochs,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    counts={};rows=[];BaseLoader=model_module.NeighborLoader
    class AuditedLoader(BaseLoader):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.graph=args[0];self.entity=kwargs['input_nodes'][0]
            self.key=str(len(counts));counts[self.key]=dict(batches=0,queries=0,dated_node_occurrences=0,edge_occurrences=0)
        def __iter__(self):
            iterator=super().__iter__()
            while True:
                start=time.perf_counter()
                try:batch=next(iterator)
                except StopIteration:return
                sample=time.perf_counter()-start;start=time.perf_counter()
                report=inspect_batch(batch,self.graph,self.entity)
                for k,v in report.items():counts[self.key][k]+=v
                bound=frontier_bound(self.graph.edge_types,self.entity,report['queries'],[128,64])
                assert sum(batch.num_nodes_dict.values())<=bound['node_occurrences']
                rows.append(dict(split=self.key,queries=report['queries'],nodes=sum(batch.num_nodes_dict.values()),edges=report['edge_occurrences'],sample_s=sample,audit_s=time.perf_counter()-start,bound=bound['node_occurrences']))
                yield batch
    model_module.NeighborLoader=AuditedLoader
    try:result=original_runner.run(seed,epochs,output)
    finally:model_module.NeighborLoader=BaseLoader
    for key,n in [('0',epochs*7453),('1',(epochs+1)*499),('2',760)]:assert counts[key]['queries']==n
    (output/'temporal-audit.json').write_text(json.dumps(dict(status='PASS',splits=dict(zip(['train','val','test'],counts.values())),audit_sha256=hashlib.sha256(Path(__file__).with_name('relkit').joinpath('batch_audit_l123.py').read_bytes()).hexdigest()),indent=2))
    (output/'batch-profile.json').write_text(json.dumps(rows))
    return result
