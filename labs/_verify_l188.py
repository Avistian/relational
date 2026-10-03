"""Independent DOM audit, contract interventions, corrupt-packet rejection."""
import copy,hashlib,json,random,shutil,tempfile
from pathlib import Path
from xml.dom import minidom
from relkit.literature_l188 import canonical_id,coverage,triage,replay,review_log,parse_atom
P=Path(__file__).resolve().parent;E=P/'evidence/l188'

def verify():
    packet=E/'packet';report=replay(packet);receipt=json.loads((packet/'collection.json').read_text())
    unique=set();count=0;per_query={}
    for a in receipt['attempts']:
        raw=(packet/a['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==a['sha256']
        if not a['accepted']:continue
        dom=minidom.parseString(raw)
        def field(node,name):
            nodes=node.getElementsByTagNameNS('*',name)
            return ''.join(x.data for x in nodes[0].childNodes if x.nodeType==x.TEXT_NODE)
        total=int(field(dom,'totalResults'));entries=dom.getElementsByTagNameNS('*','entry')
        ids=[]
        for e in entries:
            uri=field(e,'id');base=uri.split('/abs/',1)[1].rsplit('v',1)[0]
            assert canonical_id(uri)==base
            assert '2026-07-01T00:00:00Z'<=field(e,'published')<'2026-10-01T00:00:00Z'
            unique.add(base);ids.append(base);count+=1
        assert len(ids)==len(set(ids));per_query[a['kind']]=(total,len(entries))
    assert report['unique_papers']==len(unique)==30 and count==31
    assert per_query=={'relbench':(9,9),'tabarena':(22,22)}
    assert report['collection_status']=='INCOMPLETE'
    reviews=json.loads((E/'screening-review.json').read_text());rows=review_log(report,reviews)
    assert sum(r['decision']=='INCLUDE' for r in rows)==7
    assert sum(r['decision']=='DEFER' for r in rows)==23
    rng=random.Random(188)
    for _ in range(100):
        n=rng.randint(2,250);size=rng.randint(1,50)
        pages=[dict(start=i,total=n,ids=[f'2607.{j:05d}' for j in range(i,min(i+size,n))]) for i in range(0,n,size)]
        rng.shuffle(pages);assert coverage(pages)=='COMPLETE'
        bad=copy.deepcopy(pages);bad.pop();assert coverage(bad)=='INCOMPLETE'
        bad=copy.deepcopy(pages);bad[0]['total']+=1;assert coverage(bad)=='INCOMPLETE'
    rejected=0
    for name in ['config.json',receipt['attempts'][0]['file'],'screening.json','collection.json']:
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'packet';shutil.copytree(packet,dest);(dest/name).write_bytes(b'corrupted')
            try:replay(dest)
            except ValueError:rejected+=1
            else:raise AssertionError('Corrupt packet accepted')
    # Wrong student functions must affect downstream results, not an unused CHECK.
    assert replay(packet,identity_fn=lambda _: 'same')['unique_papers']!=30
    assert replay(packet,coverage_fn=lambda _: 'COMPLETE')['collection_status']!='INCOMPLETE'
    assert sum(r['decision']=='INCLUDE' for r in review_log(report,reviews,lambda _: 'DEFER'))!=7
    sample=b'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/"><o:totalResults>0</o:totalResults><o:startIndex>0</o:startIndex></feed>'
    assert coverage([parse_atom(sample)])=='COMPLETE'
    for bad in [b'<html/>',b'not xml']:
        try:parse_atom(bad)
        except Exception:pass
        else:raise AssertionError('Invalid response accepted')
    result=dict(status='PASS',raw_records=count,unique_papers=len(unique),included_for_reading=7,deferred=23,random_pagination_cases=100,corrupt_packets_rejected=rejected,wrong_learner_functions_rejected=3,collection='INCOMPLETE',paper_result_reproduction='NOT_RUN')
    (P/'_verify_l188_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':verify()
