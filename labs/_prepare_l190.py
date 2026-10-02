"""Freeze the approved input set once; never refresh a completed packet in place."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l190';Q=E/'packet'
if Q.exists():raise SystemExit('Frozen packet exists; refusing overwrite')
Q.mkdir(parents=True)
origins={}
def cp(src,dst):
 src=P/src;dst=Q/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 origins[str(dst.relative_to(Q))]=dict(origin=str(src.relative_to(P.parent)),sha256=hashlib.sha256(src.read_bytes()).hexdigest())
for n in range(181,189):cp(f'evidence/l{n}/report.json',f'reports/l{n}.json')
cp('evidence/l166/prepared.npz','model/prepared.npz')
for folder in ['pilot-1','full-1']:
 cp(f'evidence/l182/{folder}/receipt.json',f'model/{folder}/receipt.json')
 records=json.loads((P/f'evidence/l182/{folder}/receipt.json').read_text())['records']
 for row in records:
  name=f"{row['arm']}-{row['seed']}.npz";cp(f'evidence/l182/{folder}/{name}',f'model/{folder}/{name}')
for path in sorted((P/'evidence/l188/packet').rglob('*')):
 if path.is_file():cp(str(path.relative_to(P)),'literature/'+str(path.relative_to(P/'evidence/l188/packet')))
cp('evidence/l188/screening-review.json','literature-review.json')
cp('sources/l166/paper.html','sources/rdbpfn-v5.html')
cp('sources/l182/source-ledger.json','sources/l182-ledger.json')
cp('evidence/l189/packet/sources/relgnn.html','sources/relgnn-v2.html')
# These sources are inherited pinned bytes, not a new systematic literature search.
sources=[dict(file='sources/rdbpfn-v5.html',url='https://arxiv.org/html/2603.03805v5',scope='Released DFS and relational-prior/ICL method; selected Table 9 targets'),dict(file='sources/relgnn-v2.html',url='https://arxiv.org/html/2502.06784v2',scope='Composite message passing exists; proposed foundation-model interaction remains untested')]
(Q/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
cases=[
 dict(id='availability',title='Availability-aware autocomplete',impact=4,feasibility=[5,4,5],reason='Existing query keys and audits make a paired information-access test easy to specify. True arrival histories remain missing.',hypothesis='The apparent benefit of relational context shrinks when context must be available by the query cutoff.',baseline='Same fixed predictor, preprocessing and queries; event-time-only visibility versus measured availability visibility.',falsifier='The predeclared paired deterioration does not exceed the useful-effect threshold on held-out queries.',novelty='NOT_ESTABLISHED',cost='NOT_ESTABLISHED'),
 dict(id='composite',title='Composite structure in a foundation learner',impact=5,feasibility=[3,2,4],reason='High potential relevance, but inserting routes changes the model and invalidates direct checkpoint reuse.',hypothesis='Adding composite routes interacts positively with relational pretraining under matched compute.',baseline='Four matched arms: conventional/composite encoder crossed with scratch/pretrained initialization.',falsifier='Interaction confidence interval includes zero or gains disappear against the matched conventional control.',novelty='NOT_ESTABLISHED',cost='NOT_ESTABLISHED'),
 dict(id='constraints',title='Accuracy under serving and privacy constraints',impact=4,feasibility=[3,3,2],reason='Requires real latency, availability histories, a privacy unit, and a bounded release/training mechanism; existing simulations do not measure their joint utility.',hypothesis='A declared relational pipeline retains useful accuracy subject to fixed freshness, latency and entity-level privacy constraints.',baseline='Same workload, task and privacy accounting for a strong flattened baseline and relational model.',falsifier='Any mandatory constraint fails or the accuracy difference is not useful.',novelty='NOT_ESTABLISHED',cost='NOT_ESTABLISHED')]
(Q/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(Q/'origins.json').write_text(json.dumps(origins,indent=2)+'\n')
files={str(p.relative_to(Q)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Q.rglob('*')) if p.is_file()}
(E/'input-manifest.json').write_text(json.dumps(dict(experiment='L190 Q3 Research-Gap Evidence Replay',files=files),indent=2)+'\n')
print('Frozen',len(files),'files;',sum(p.stat().st_size for p in Q.rglob('*') if p.is_file()),'bytes')
