"""Recover the paper-label task from verified current archive and pinned old SQL.

Historical zip bytes/order cannot be established. Never bypass the old checksum.
"""
import hashlib,io,json,zipfile,urllib.request
from pathlib import Path
from relbench.base import Table
from relbench.tasks import get_task

def historical_task(output):
    task=get_task('rel-f1','driver-dnf',download=False)
    url='https://relbench.stanford.edu/download/rel-f1/tasks/driver-dnf.zip'
    raw=urllib.request.urlopen(url,timeout=60).read()
    digest=hashlib.sha256(raw).hexdigest()
    assert digest=='bd562529a3c0016363d5cae247979fd3712d77ba948574f135dab88c036d3e2d'
    folder=Path(output)/'historical-task';folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for split,n,positives in [('train',11411,1365),('val',566,125),('test',702,207)]:
            path=folder/(split+'.parquet');path.write_bytes(archive.read('driver-dnf/'+split+'.parquet'))
            table=Table.load(path);assert len(table)==n
            table.df['did_not_finish']=1-table.df['did_not_finish']
            assert int(table.df.did_not_finish.sum())==positives
            table.save(path)
    task.cache_dir=str(folder)
    return task,dict(current_archive_sha256=digest,historical_archive_sha256='UNAVAILABLE',
        reconstruction='1-current_label; exact pre-flip CASE expression; paper Table13 counts',
        historical_source_commit='e416c3d208cb8a6d8bd31a68c46ca9090ecb334a',
        row_order='Current archive order; historical ordering NOT_ESTABLISHED')
