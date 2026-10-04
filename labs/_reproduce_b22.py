"""Authenticate the primary source; refuse historical inference without its artifacts."""
import hashlib,json,sys,tarfile
from pathlib import Path
from bs4 import BeautifulSoup


def source_audit(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    for row in manifest['files']:
        if hashlib.sha256((folder/row['file']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('SOURCE_HASH_MISMATCH: '+row['file'])
    soup=BeautifulSoup((folder/'paper.html').read_text(),'html.parser');plain=' '.join(soup.get_text(' ',strip=True).split())
    for fragment in ['805843','0.0510','0.0417','0.0608','1.44','Trained weights, benchmark data, and the real-informed continuation manifests are not included']:
        assert fragment in plain,fragment
    with tarfile.open(folder/'source.tar.gz') as archive:
        names=archive.getnames()
    executable=[n for n in names if n.endswith(('.py','.ipynb','.ckpt','.pt','.safetensors'))]
    assert len(names)==35 and not executable
    inventory=json.loads((folder/'archive-inventory.json').read_text());assert [r['name'] for r in inventory]==names
    return dict(status='PASS_SOURCE_AUDIT',historical_status='INCOMPLETE_SOURCE_PROTOCOL_GATE',fresh_historical_inference='NOT_RUN',full_paper='NOT_RUN',target='2609.27679v1 Appendix E.2 support-write intervention',paper_reference=dict(episodes=72,checkpoint='L24 15K',block=12,features=8,support=1024,queries=256,class_counts=[2,4,8],rbf_components=[4,16,64],replicates=8,generator_seed=805843,delta_ce=.0510,episode_ci95=[.0417,.0608],positive_episodes=72,delta_accuracy_pp=-1.44),archive_members=len(names),executable_members=executable,missing=['Authenticated L24 15K checkpoint','Executable model and exact probe driver from the described supplement','Exact episode identities or fully reproducible generator','Bootstrap implementation, RNG and full-precision paired records'],scope='arXiv v1 responses inspected; no claim that a separate external release cannot exist')

if __name__=='__main__':
 P=Path(__file__).resolve().parent;report=source_audit(P/'sources/b22')
 if '--fresh' in sys.argv:raise SystemExit('NOT_RUN: INCOMPLETE_SOURCE_PROTOCOL_GATE — '+ '; '.join(report['missing']))
 (P/'evidence/b22/reproduction.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
