"""Verify extracted arrays against both pinned archives and the recorded data manifest."""
import hashlib,json,tarfile
from pathlib import Path
from _fetch_l055 import HASHES,ROOT,fetch


def verify_extracted(root):
    root=Path(root);manifest=json.loads((ROOT/'_data_l055.json').read_text())
    checked={}
    for task,archive_sha in HASHES.items():
        archive=root/(task+'.tabred')
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==archive_sha
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                if not member.isfile() or not (member.name.endswith('.npy') or member.name.endswith('/info.json')):continue
                raw=stream.extractfile(member).read()
                relative=task+'/'+member.name
                actual=(root/relative).read_bytes()
                sha=hashlib.sha256(raw).hexdigest()
                assert hashlib.sha256(actual).hexdigest()==sha,(relative,'extracted cache differs from pinned archive')
                if relative in manifest['files']:
                    assert sha==manifest['files'][relative]['sha256'],(relative,'manifest differs')
                checked[relative]=sha
    return dict(status='PASS',files=checked,archives=HASHES,manifest_sha256=hashlib.sha256((ROOT/'_data_l055.json').read_bytes()).hexdigest())


if __name__=='__main__':
    result=verify_extracted(fetch())
    result['measured_evidence_sha256']=hashlib.sha256((ROOT/'_verify_l055_v2_results.json').read_bytes()).hexdigest()
    result['scope']='Post-run cache verification against pinned archive contents; binds measured artifact, does not claim a pre-run check occurred'
    (ROOT/'_release_check_l055_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS:',len(result['files']),'extracted arrays/metadata match pinned archive bytes and recorded hashes')
