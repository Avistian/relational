"""Freeze primary-source bytes and inherited executable evidence without paid calls."""
import hashlib,json,shutil,urllib.request,zipfile,io
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b13';E=P/'evidence/b13'
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'B13-source-audit'}),timeout=60).read()
def main():
    S.mkdir(parents=True,exist_ok=True);ledger=[]
    for name,url in [('plurel-v1.html','https://arxiv.org/html/2602.04029v1'),('rdbpfn-v1.html','https://arxiv.org/html/2603.03805v1'),('rdbpfn-v5.html','https://arxiv.org/html/2603.03805v5')]:
        p=S/name
        if not p.exists():p.write_bytes(fetch(url))
        ledger.append(dict(file=name,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pin=S/'plurel-release.json'
    if not pin.exists():
        d=json.loads(fetch('https://api.github.com/repos/stanford-star/plurel/commits/main'))
        pin.write_text(json.dumps(dict(repository='https://github.com/stanford-star/plurel',commit=d['sha'],date=d['commit']['committer']['date']),indent=2)+'\n')
    revision=json.loads(pin.read_text())['commit'];archive=S/'plurel-source.zip'
    if not archive.exists():archive.write_bytes(fetch('https://codeload.github.com/stanford-star/plurel/zip/'+revision))
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            rel=Path(*Path(name).parts[1:])
            if not name.endswith('/') and rel.suffix in ['.py','.md','.toml','.yaml','.yml','.json'] or rel.name=='LICENSE':
                dest=S/'plurel'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    ledger.append(dict(file=archive.name,url='https://github.com/stanford-star/plurel/tree/'+revision,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),identity='Current release; v1 training identity NOT_ESTABLISHED'))
    for src,dest in [(P/'evidence/l200/reproducer.zip',E/'l200-reproducer.zip'),(P/'relkit/rdbpfn_l166.py',S/'rdbpfn_visible.py'),(P/'l166-reproduction.md',S/'l166-reproduction.md'),(P/'l200-reproduction.md',S/'l200-reproduction.md'),(P/'_fetch_l166.py',S/'_fetch_l166.py'),(P/'_run_l166.py',S/'_run_l166.py'),(P/'l166-requirements.txt',S/'l166-requirements.txt')]:
        shutil.copyfile(src,dest);ledger.append(dict(file=str(dest.relative_to(P)),inherited=str(src.relative_to(P)),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    (S/'source-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print('Frozen',len(ledger),'source entries; PluRel',revision)
if __name__=='__main__':main()
