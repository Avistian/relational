"""Authenticate and reconstruct every displayed value in four frozen tables."""
import hashlib,json,math,re
from html.parser import HTMLParser
from pathlib import Path
from relkit.tracking_l191 import signed_gap,eligible,compare_pool,average_ranks

def parse_tables(html):
    class Cells(HTMLParser):
        def __init__(self):
            super().__init__(); self.table=None; self.cell=None; self.row=[]; self.tables={}; self.annotation=False
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='figure' and a.get('id') in ('S4.T3','S4.T4','S4.T7','S4.T8'):
                self.table=a['id']; self.tables[self.table]=[]
            if self.table and tag=='tr':self.row=[]
            if self.table and tag in ('td','th'):self.cell=[]
            if tag=='annotation':self.annotation=True
        def handle_data(self,data):
            if self.cell is not None and not self.annotation:self.cell.append(data)
        def handle_endtag(self,tag):
            if tag=='annotation':self.annotation=False
            if self.table and tag in ('td','th') and self.cell is not None:
                self.row.append(' '.join(''.join(self.cell).split())); self.cell=None
            if self.table and tag=='tr':self.tables[self.table].append(self.row)
            if tag=='figure':self.table=None
    parser=Cells();parser.feed(html)
    config={3:('RelBenchV1','AUROC',12,18),4:('RelBenchV2','AUROC',5,4),7:('RelBenchV1','MAE',9,17),8:('RelBenchV2','MAE',2,6)}
    # Ordered database labels are explicit because HTML group headings span columns.
    dbs={3:['f1']*2+['avito']*2+['event']*2+['trial']+['amazon']*2+['stack']*2+['hm'],4:['mimic']+['ratebeer']*3+['arxiv'],7:['f1','avito','event','trial','trial','amazon','amazon','stack','hm'],8:['ratebeer','arxiv']}
    result=[]
    for number,(suite,metric,n,m) in config.items():
        raw=parser.tables[f'S4.T{number}']; headers=raw[1][-(n+2):-2]
        rows=[]
        for cells in raw[2:]:
            if len(cells) not in (n+3,n+4):raise ValueError('Unexpected table row width')
            name=cells[-(n+3)];name=name.replace('∗','').strip()
            if name.startswith('HGT') and name!='HGT':name='HGT_PE'
            if name.startswith('RT'):name='RT_zero'
            if name.startswith('LLM'):name=name.replace(' ','')
            values=cells[-(n+2):-2]
            if any(not re.fullmatch(r'\d+\.\d+',s) for s in values+cells[-2:]):raise ValueError('Unexpected/missing numeric source cell')
            rows.append(dict(method=name,author_evaluated='∗' in cells[-(n+3)],displayed=values,values=list(map(float,values)),published_aggregate=float(cells[-2]),published_rank=float(cells[-1])))
        if len(rows)!=m or len(set(x['method'] for x in rows))!=m:raise ValueError('Missing/duplicate method')
        result.append(dict(number=number,suite=suite,metric=metric,tasks=[a+'/'+b for a,b in zip(dbs[number],headers)],rows=rows))
    return result

def reconstruct(tables,metadata):
    summaries=[];count=0
    for table in tables:
        rows=table['rows'];n=len(table['tasks']);metric=table['metric'];lookup={r['method']:r for r in rows};baseline=lookup['LightGBM']['values'];target=lookup['KumoRFM-2']['values']
        ranks=[average_ranks([r['values'][i] for r in rows],metric=='AUROC') for i in range(n)]
        audit=[]
        for j,row in enumerate(rows):
            vals=row['values'];aggregate=sum(vals)/n if metric=='AUROC' else sum(a/b for a,b in zip(vals,baseline))/n
            # Independent rounding intervals, not uncertainty intervals from repeated runs.
            half=[.5*10**(-len(v.split('.')[1])) for v in row['displayed']]
            if metric=='AUROC':lo=sum(a-h for a,h in zip(vals,half))/n;hi=sum(a+h for a,h in zip(vals,half))/n;summary_half=.005
            elif row['method']=='LightGBM':lo=hi=1.;summary_half=.0005
            else:
                bh=[.5*10**(-len(v.split('.')[1])) for v in lookup['LightGBM']['displayed']]
                lo=sum((a-h)/(b+k) for a,h,b,k in zip(vals,half,baseline,bh))/n
                hi=sum((a+h)/(b-k) for a,h,b,k in zip(vals,half,baseline,bh))/n;summary_half=.0005
            published=row['published_aggregate'];rank=sum(v[j] for v in ranks)/n
            audit.append(dict(method=row['method'],recomputed_aggregate=aggregate,published_aggregate=published,rounding_interval=[lo,hi],aggregate_status='ROUNDING_COMPATIBLE' if lo-summary_half-1e-12<=published<=hi+summary_half+1e-12 else 'OUTSIDE_ROUNDING_BOUND',displayed_mean_rank=rank,published_rank=row['published_rank'],rank_status='MATCH_DISPLAYED' if abs(rank-row['published_rank'])<=.005+1e-12 else 'DIFFERS_DISPLAYED',per_task_ranks=[v[j] for v in ranks]))
        pools={pool:compare_pool(target,{r['method']:r['values'] for r in rows if eligible(metadata.get(r['method'],{}),pool)},metric,baseline) for pool in ['foundation','supervised','all']}
        count+=len(rows)*n
        summaries.append(dict(number=table['number'],suite=table['suite'],metric=metric,tasks=table['tasks'],method_count=len(rows),audit=audit,pools=pools))
    return dict(experiment='L191 RelBench Published-Table Reconstruction',reconstruction='COMPLETE_SELECTED_TABLES',tables=summaries,task_cells=count,method_rows=sum(len(t['rows']) for t in tables),fresh_inference='NOT_RUN',historical_model_reproduction='NOT_ESTABLISHED',current_global_sota='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

def replay191(packet,manifest):
    actual={str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file()}
    if actual!=set(manifest['files']):raise ValueError('Packet file set mismatch')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source hash mismatch: '+name)
    tables=parse_tables((packet/'paper-v1.html').read_text())
    expected=json.loads((packet/'tables.json').read_text())
    if tables!=expected:raise ValueError('Extraction differs from frozen table packet')
    return reconstruct(tables,json.loads((packet/'method-policy.json').read_text()))

if __name__=='__main__':
    root=Path(__file__).resolve().parent/'evidence/l191'
    report=replay191(root/'packet',json.loads((root/'input-manifest.json').read_text()))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['reconstruction'],report['task_cells'],'task cells',report['method_rows'],'method rows')
    for t in report['tables']:
        print('Table',t['number'],'summary mismatches:',[(x['method'],x['aggregate_status'],x['rank_status']) for x in t['audit'] if x['aggregate_status']!='ROUNDING_COMPATIBLE' or x['rank_status']!='MATCH_DISPLAYED'])
