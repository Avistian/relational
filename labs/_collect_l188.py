"""Collect a fresh immutable packet. Never overwrites an earlier collection.
Usage: python labs/_collect_l188.py --output /new/packet
"""
import argparse,hashlib,json,time,urllib.request,urllib.error
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
from relkit.literature_l188 import parse_atom,coverage

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir()
    terms=[('relational','"relational foundation model"'),('tabular','"tabular foundation model"'),('relbench','RelBench'),('tabarena','TabArena')]
    config=dict(experiment='L188 Q3-2026 Literature Tracking Replay',queries=[dict(name=n,search_query=f'all:{q} AND submittedDate:[202607010000 TO 202609302359]') for n,q in terms],page_size=100,sort_by='submittedDate',sort_order='ascending',max_pages=100,max_attempts=2,timeout=30,delay_seconds=3)
    def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
    save('config.json',config);receipt=dict(attempts=[]);save('screening.json',{})
    last=0
    def fetch(url,kind,start=0):
        nonlocal last
        time.sleep(max(0,3-(time.monotonic()-last)))
        a=dict(url=url,kind=kind,start=start,retrieved_at=datetime.now(timezone.utc).isoformat(),ok=False,accepted=False)
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'RelationalCourse-L188/1.0 (educational reproducibility audit)'})
            with urllib.request.urlopen(req,timeout=30) as r:
                raw=r.read();a.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type',''),ok=r.status==200)
        except urllib.error.HTTPError as exc:raw=exc.read();a.update(status=exc.code,error=str(exc))
        except Exception as exc:raw=b'';a.update(status=None,error=repr(exc))
        last=time.monotonic();name=f'raw/{len(receipt["attempts"]):04d}-{kind}.bin'
        (out/name).write_bytes(raw);a.update(file=name,sha256=hashlib.sha256(raw).hexdigest())
        receipt['attempts'].append(a);save('collection.json',receipt)
        return a,raw
    for query in config['queries']:
        pages=[];offset=0
        for _ in range(config['max_pages']):
            accepted=False
            for attempt in range(config['max_attempts']):
                url='https://export.arxiv.org/api/query?'+urlencode(dict(search_query=query['search_query'],start=offset,max_results=100,sortBy='submittedDate',sortOrder='ascending'))
                a,raw=fetch(url,query['name'],offset)
                if a['ok']:
                    try:
                        p=parse_atom(raw)
                        if p['start']!=offset:raise ValueError('Wrong response offset')
                        pages.append(p);a['accepted']=True;accepted=True
                    except Exception as exc:a.update(ok=False,error='Parse: '+repr(exc))
                save('collection.json',receipt)
                if accepted:break
            if not accepted:break
            if coverage(pages)=='COMPLETE':break
            if not p['ids'] or len({x['total'] for x in pages})!=1:break
            offset+=len(p['ids'])
        print(query['name'],coverage(pages),sum(len(p['ids']) for p in pages),flush=True)
    sources=[('api-manual','https://info.arxiv.org/help/api/user-manual.html'),('api-terms','https://info.arxiv.org/help/api/tou.html'),('rss-guide','https://info.arxiv.org/help/rss.html'),('relbench-board','https://star-project.stanford.edu/relbench/leaderboard/'),('relbench-home','https://star-project.stanford.edu/relbench/'),('tabarena-board','https://huggingface.co/spaces/TabArena/leaderboard/raw/main/README.md')]
    for name,url in sources:
        for _ in range(2):
            a,raw=fetch(url,name)
            if a['ok']:break
    save('manifest.json',dict(files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}))
    print('Packet:',out,flush=True)
if __name__=='__main__':main()
