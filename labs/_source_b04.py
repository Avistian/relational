"""Reconstruct the bounded source search and fail closed on missing Figure 3 identities."""
import hashlib,json,tarfile,re
from pathlib import Path

def source_gate(root):
    p=Path(root);manifest=json.loads((p/'manifest.json').read_text())
    for item in manifest['files']:
        data=(p/item['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==item['sha256'],item['file']
        assert len(data)==item['bytes']
    # Inspect actual immutable archive rather than trusting a prose absence claim.
    with tarfile.open(p/'upstream.tar.gz') as t:
        files={str(Path(*Path(x.name).parts[1:])):t.extractfile(x).read() for x in t.getmembers() if x.isfile()}
    matches=[name for name,data in files.items() if name.endswith(('.py','.sh','.md','.ipynb')) and re.search(rb'needle|haystack',data,re.I)]
    assert not matches, 'New candidate Figure 3 source needs review'
    hf=json.loads((p/'hf-v2.json').read_text());weights=[x['rfilename'] for x in hf['siblings'] if x['rfilename'].endswith('.ckpt')]
    assert weights==['tabicl-classifier-v1-20250208.ckpt','tabicl-classifier-v1.1-20250506.ckpt','tabicl-classifier-v2-20260212.ckpt','tabicl-regressor-v2-20260212.ckpt']
    assert b'use_cautious_wd' in files['README.md']
    missing=['exact generator parameters and seeds','ordered complete context grid and input rows','No-SSMax and SSMax matched Figure-3 checkpoint identities','Figure-3 QASSMax checkpoint identity','raw probabilities/entropy reference and exact aggregation implementation']
    return dict(experiment='B04-TABICLV2-FIG3-ATTENTION-FADING',status='INCOMPLETE_SOURCE_PROTOCOL',benchmark_runs=0,commit=manifest['commit'],authenticated_files=len(manifest['files']),archive_files=len(files),figure_source_matches=matches,released_weights=weights,missing=missing,scope='Pinned complete upstream archive, paper v1 and Figure 3 assets, author HF repositories, first two GitHub issue pages (165 entries at retrieval); not proof of global unavailability',pretraining='NOT_RUN',full_benchmark='NOT_RUN')

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=source_gate(p/'sources/b04')
    (p/'evidence/b04/source-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
