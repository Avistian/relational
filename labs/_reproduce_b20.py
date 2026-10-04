"""Complete printed-table arithmetic, with a fail-closed historical training gate."""
import argparse,hashlib,json,re
from pathlib import Path
from decimal import Decimal
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent;S=P/'sources/b20';E=P/'evidence/b20'

def replay(packet):
    out=[]
    for row in packet['rows']:
        mean=sum(Decimal(x) for x in row['scores'])/Decimal(23)
        reported=Decimal(row['reported'])
        # Each input and displayed average can differ by up to .0005 due to rounding.
        out.append(dict(label=row['label'],mean=float(mean),printed_average=float(reported),difference=float(mean-reported),matches_3_decimals=abs(mean-reported)<Decimal('.0005'),compatible_with_input_rounding=abs(mean-reported)<=Decimal('.001')))
    by={x['label']:x for x in out};a=by['TF17'];b=by['Family B all-at-once']
    av=next(x for x in packet['rows'] if x['label']=='TF17')['scores'];bv=next(x for x in packet['rows'] if x['label']=='Family B all-at-once')['scores']
    table1=[]
    for label,count,printed in packet['table1'][1:]:
        if 'All-at-once' in label:key='Family B all-at-once'
        elif 'RDB-PFN' in label:key='Paper (single-tbl)'
        else:key=label.split()[-1]
        table1.append(dict(label=label,printed=printed,reconstructed=by[key]['mean'],matches_3_decimals=abs(Decimal(str(by[key]['mean']))-Decimal(printed))<Decimal('.0005')))
    appendix_targets={'TF12':'0.7150','TF17':'0.7026','Family B all-at-once':'0.5407','Paper (single-tbl)':'0.7997'}
    appendix=[]
    for label,value in appendix_targets.items():
        assert value in packet['appendix_b']
        mean=Decimal(str(by[label]['mean']));diff=abs(mean-Decimal(value))
        appendix.append(dict(label=label,claimed_exact=value,reconstructed=float(mean),matches_4_decimals=diff<Decimal('.00005'),compatible_with_rounded_inputs=diff<=Decimal('.00055')))
    return dict(table1=table1,appendix_b=appendix,status='COMPLETE_PRINTED_TABLE_AUDIT',historical_reproduction='INCOMPLETE_SOURCE_PROTOCOL_GATE',fresh_training='NOT_RUN',tasks=23,configurations=len(out),score_cells=23*len(out),rows=out,selected=dict(a_mean=a['mean'],b_mean=b['mean'],gap=a['mean']-b['mean'],tasks_a_better=sum(Decimal(x)>Decimal(y) for x,y in zip(av,bv)),tasks_b_better=sum(Decimal(y)>Decimal(x) for x,y in zip(av,bv)),ties=sum(x==y for x,y in zip(av,bv))),claim='Printed-table arithmetic only; no raw predictions or historical weights')

def extract():
    for item in json.loads((S/'manifest.json').read_text()):
        if hashlib.sha256((S/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('SOURCE_HASH_MISMATCH '+item['file'])
    soup=BeautifulSoup((S/'paper.html').read_text(),'html.parser')
    def table(id):return [[c.get_text(' ',strip=True) for c in tr.select('td,th')] for tr in soup.find(id=id).select('tr')]
    left=table('A1.T5');right=table('A1.T6');rows=[]
    assert len(left)==len(right)==15
    for a,b in zip(left[1:],right[1:]):
        assert a[0]==b[0] and len(a)==13 and len(b)==13
        label=re.sub(r'^(?:\+\s*)+','',a[0]);scores=a[1:]+b[1:-1];assert len(scores)==23
        assert all(0<=Decimal(x)<=1 for x in scores)
        rows.append(dict(label=label,scores=scores,reported=b[-1]))
    assert len(rows)==14
    packet=dict(tasks=left[0][1:]+right[0][1:-1],rows=rows,source_sha256=hashlib.sha256((S/'paper.html').read_bytes()).hexdigest(),table1=table('S4.T1'),appendix_b=soup.find(id='A2').get_text(' ',strip=True))
    return packet

def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true');args=p.parse_args()
    packet=extract()
    if args.fresh:raise SystemExit('NOT_RUN: historical corpus, schedule, architecture, seeds, splits, checkpoint selection and support identities unresolved. See b20-reproduction.md. No benchmark trainer dispatched.')
    report=replay(packet);E.mkdir(exist_ok=True,parents=True)
    (E/'paper-packet.json').write_text(json.dumps(packet,indent=2)+'\n');(E/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
