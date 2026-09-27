"""Run unchanged L117 training with transparent audits on every loader yield."""
import json,hashlib
from pathlib import Path
import torch
import relkit.rdl_l117 as model_module
import _run_l117 as original_runner
from relkit.batch_audit_l123 import audit_batch


def run(seed,epochs,output):
    output=Path(output);counts={};BaseLoader=model_module.NeighborLoader
    class AuditedLoader(BaseLoader):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.original_graph=args[0];self.entity=kwargs['input_nodes'][0]
            self.audit_key=str(len(counts));counts[self.audit_key]=dict(batches=0,queries=0,dated_node_occurrences=0,edge_occurrences=0)
        def __iter__(self):
            for batch in super().__iter__():
                report=audit_batch(batch,self.original_graph,self.entity)
                for name,value in report.items():counts[self.audit_key][name]+=value
                yield batch
    model_module.NeighborLoader=AuditedLoader
    try:
        result=original_runner.run(seed,epochs,output)
    finally:
        model_module.NeighborLoader=BaseLoader
    assert counts['0']['queries']==epochs*7453
    assert counts['1']['queries']==(epochs+1)*499
    assert counts['2']['queries']==760
    report=dict(status='PASS',splits=dict(zip(['train','val','test'],counts.values())),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(Path(__file__).with_name('relkit').joinpath('batch_audit_l123.py').read_bytes()).hexdigest(),
        contract='Every batch: original node times, root cutoff <=, original edge identity and query isolation',
        unavailable='Ingestion/availability histories and mutable features; static table creation times')
    (output/'temporal-audit.json').write_text(json.dumps(report,indent=2))
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=10);p.add_argument('--output',default='labs/results/l124/local');a=p.parse_args();run(a.seed,a.epochs,a.output)
