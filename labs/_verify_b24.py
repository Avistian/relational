"""Independent expected values and hostile mutations, including rehashed semantics."""
import copy,hashlib,json,shutil,tempfile
from pathlib import Path
import numpy as np
from _audit_b24 import unpack,scan,rank_auc,replay
from _test_b24 import checks

def verify(p):
    e=p/'evidence/b24';identity=json.loads((e/'source-identity.json').read_text());packet=p/'evidence/b23/portable-packet.zip'
    assert checks()=='PASS'
    assert rank_auc([1,0,1,0],[.8,.5,.5,.2])==.875
    r=replay(packet,identity)
    assert r['runs']==40 and r['predictions']==28080
    assert [c['positive'] for c in r['comparisons']]==[10,6,10]
    assert abs(r['comparisons'][1]['mean']-.004366369004050163)<1e-12
    rejected=[]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);badzip=td/'bad.zip';badzip.write_bytes(packet.read_bytes()+b'bad')
        try:unpack(badzip,identity,td/'bad')
        except ValueError:rejected.append('archive_bytes')
        else:raise AssertionError('Changed archive admitted')
        root=unpack(packet,identity,td/'original')
        for mode in ['missing','duplicate','keys','labels','supports','nan','range','metric','baseline_coef']:
            out=td/mode;shutil.copytree(root,out);phase='baseline-1' if mode=='baseline_coef' else 'pilot-1';rp=out/phase/'receipt.json';receipt=json.loads(rp.read_text());row=receipt['records'][0]
            if mode=='missing':receipt['records'].pop()
            elif mode=='duplicate':receipt['records'].append(copy.deepcopy(row))
            elif mode=='metric':row['auc']+=.1
            else:
                path=out/phase/f"{row['arm']}-{row['seed']}.npz"
                with np.load(path) as d:a={k:d[k].copy() for k in d.files}
                if mode=='keys':a['keys'][0]=a['keys'][1]
                if mode=='labels':a['label'][0]=1-a['label'][0]
                if mode=='supports':a['support_keys'][0]=a['support_keys'][1]
                if mode=='nan':a['probability'][0]=np.nan
                if mode=='range':a['probability'][0]=1.1
                if mode=='baseline_coef':a['coef'][0,0]+=1
                np.savez_compressed(path,**a);row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            rp.write_text(json.dumps(receipt))
            try:scan(out)
            except (ValueError,AssertionError):rejected.append(mode)
            else:raise AssertionError('Semantic corruption admitted '+mode)
    assert r==json.loads((e/'report.json').read_text())
    out=dict(status='PASS',grid_runs=40,predictions=28080,independent_scoring=True,gate_cases=1944,corruptions_rejected=rejected,report_parity=True)
    (p/'_verify_b24_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':verify(Path(__file__).resolve().parent)
