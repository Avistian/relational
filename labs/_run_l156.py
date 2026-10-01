"""Full released F1 run with passive root identity, temporal and availability audits.

The optional fit_horizon lane changes preprocessing only; all static-table arrival
histories remain unknown. It is a course correction for a declared fit-time policy.
"""
import copy,hashlib,json
from pathlib import Path
import torch
import relkit.rdl_l117 as model_module
import _run_l117 as original_runner
from _run_l152 import run as audited_run
from relkit.temporal_audit_l156 import audit_observations


def fit_horizon_graph(db,types,text_cfg):
    """Fit dated feature processors by2005-01-01; transform the full graph rows."""
    import pandas as pd
    from torch_frame.data import Dataset
    data,stats=BASE_GRAPH(db,copy.deepcopy(types),text_cfg)
    report={}
    for name,table in db.table_dict.items():
        cols=copy.deepcopy(types[name]);model_module.remove_pkey_fkey(cols,table)
        if not cols:
            report[name]=dict(status='CONSTANT_NO_FIT');continue
        if table.time_col is None:
            report[name]=dict(status='STATIC_ARRIVAL_NOT_ESTABLISHED',fit_rows=len(table.df));continue
        mask=table.df[table.time_col]<=pd.Timestamp('2005-01-01')
        train=table.df.loc[mask].copy();assert len(train)>0
        dataset=Dataset(train,col_to_stype=cols,col_to_text_embedder_cfg=text_cfg).materialize()
        data[name].tf=dataset.convert_to_tensor_frame(table.df)
        stats[name]=dataset.col_stats
        report[name]=dict(status='PASS',fit_rows=len(train),transformed_rows=len(table.df),latest_fit_event=str(train[table.time_col].max()))
    fit_horizon_graph.report=report
    return data,stats

BASE_GRAPH=original_runner.make_pkey_fkey_graph

def run(seed,epochs,output,lane='released'):
    if lane not in ['released','fit_horizon']:raise ValueError(lane)
    output=Path(output);base=model_module.NeighborLoader;counts={};mutants=[]
    class RootAuditedLoader(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.root_nodes=kwargs['input_nodes'][1].clone();self.root_times=kwargs['input_time'].clone()
            self.entity=kwargs['input_nodes'][0];self.original_graph=args[0]
            self.target=getattr(kwargs.get('transform'),'target',None)
            self.key=str(len(counts));counts[self.key]=dict(batches=0,queries=0,equality_nodes=0,owner_table_checks=0,availability_unknown_groups=0)
        def check_roots(self,batch):
            root=batch[self.entity];n=root.batch_size;ids=root.input_id.cpu()
            if not torch.equal(root.n_id[:n].cpu(),self.root_nodes[ids]):raise ValueError('Root entity differs from task query')
            if not torch.equal(root.seed_time.cpu(),self.root_times[ids]):raise ValueError('Root cutoff differs from task query')
            if self.target is not None and not torch.equal(root.y.cpu(),self.target[ids]):raise ValueError('Root label differs from task query')
        def __iter__(self):
            for batch in super().__iter__():
                self.check_roots(batch);c=counts[self.key];c['batches']+=1;c['queries']+=batch[self.entity].batch_size
                if c['batches']==1:
                    from relkit.batch_audit_l123 import audit_batch
                    bad=batch.clone();bad[self.entity].seed_time[0]+=1
                    try:self.check_roots(bad)
                    except ValueError:mutants.append(self.key+':wrong-root-cutoff-rejected')
                    else:raise AssertionError('Mutated root cutoff accepted')
                    for kind in batch.node_types:
                        if 'time' in batch[kind] and batch[kind].time.numel():
                            bad=batch.clone();bad[kind].time[0]+=1
                            try:audit_batch(bad,self.original_graph,self.entity)
                            except AssertionError:mutants.append(self.key+':forged-node-time-rejected')
                            else:raise AssertionError('Forged timestamp accepted')
                            break
                roots=batch[self.entity].seed_time
                records=[]
                for kind in batch.node_types:
                    s=batch[kind]
                    if not s.num_nodes:continue
                    if 'time' in s:
                        # Max is taken PER OWNER, never over the batch. Original
                        # identity/cutoff checks still inspect every sampled row.
                        maxima=torch.full((len(roots),),torch.iinfo(torch.int64).min,dtype=torch.int64)
                        maxima.scatter_reduce_(0,s.batch.cpu(),s.time.cpu(),reduce='amax',include_self=True)
                        present=maxima!=torch.iinfo(torch.int64).min
                        c['equality_nodes']+=int((s.time==roots[s.batch]).sum())
                        records.extend(dict(owner=i,cutoff=int(roots[i]),event=int(maxima[i]),available=None,rule='inclusive',kind='event') for i in present.nonzero().flatten().tolist())
                    else:
                        records.extend(dict(owner=i,cutoff=int(roots[i]),event=None,available=None,rule='inclusive',kind='static') for i in torch.unique(s.batch).tolist())
                verdict=audit_observations(records)
                if verdict['counts']['FAIL']:raise ValueError('Temporal dependency failure')
                c['owner_table_checks']+=len(records);c['availability_unknown_groups']+=verdict['counts']['NOT_ESTABLISHED']
                yield batch
    old_types=original_runner.get_stype_proposal
    if lane=='fit_horizon':
        def train_types(db):
            import pandas as pd
            from relbench.base import Database,Table
            subset={}
            for name,t in db.table_dict.items():
                df=t.df if t.time_col is None else t.df[t.df[t.time_col]<=pd.Timestamp('2005-01-01')]
                subset[name]=Table(df.copy(),fkey_col_to_pkey_table=t.fkey_col_to_pkey_table,pkey_col=t.pkey_col,time_col=t.time_col)
            return old_types(Database(subset))
        original_runner.get_stype_proposal=train_types
        original_runner.make_pkey_fkey_graph=fit_horizon_graph
    model_module.NeighborLoader=RootAuditedLoader
    try:result=audited_run(seed,epochs,output)
    finally:
        model_module.NeighborLoader=base;original_runner.get_stype_proposal=old_types;original_runner.make_pkey_fkey_graph=BASE_GRAPH
    report=dict(status='PASS',lane=lane,root_query_identity='PASS',mutants_rejected=mutants,splits=dict(zip(['train','val','test'],counts.values())),availability='NOT_ESTABLISHED',coverage='All yielded batches, every root and every sampled dated node/edge through L123 checker; maximum timestamp per owner/table through live L156 checker',preprocessing=fit_horizon_graph.report if lane=='fit_horizon' else 'Released test-cutoff materialization',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'audit-l156.json').write_text(json.dumps(report,indent=2));return result
