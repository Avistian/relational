"""Freeze all course inputs and settings before dispatch; no score-based selection."""
import hashlib,json,tarfile,io,platform,sys
from pathlib import Path
import numpy as np
import contextlib,os,torch,sklearn
P=Path(__file__).resolve().parent;S=P/'sources/b04a';E=P/'evidence/b04a'
if (E/'protocol.json').exists():raise SystemExit('Refuse overwrite of frozen protocol')
manifest=json.loads((S/'manifest.json').read_text());full=manifest['files'].pop('upstream.tar.gz')
manifest['full_archive']=full
# Compact, complete executable source/config/license archive: notebooks excluded, full tree retained.
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for p in sorted((S/'upstream').rglob('*')):
  if p.is_file() and (p.suffix in ['.py','.yaml','.yml','.cfg','.toml'] or p.name in ['README.md','LICENSE.txt','NOTICE.txt','setup.py']):
   b=p.read_bytes();m=tarfile.TarInfo(str(p.relative_to(S/'upstream')));m.size=len(b);m.mtime=0;t.addfile(m,io.BytesIO(b))
b=buf.getvalue();(S/'code.tar.gz').write_bytes(b)
manifest['files']['code.tar.gz']={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'url':'Derived from full archive; complete Python/config/license subset, notebooks excluded'}
for name in ['tabflex','wide','beta']:
 b=(S/(name+'.txt')).read_bytes();manifest['files'][name+'.txt']={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'url':'Text extracted from archived HTML'}
(S/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Paired latent tables: seed identifies data; all widths share the same rows and targets.
tables={}
for seed in [0,1,2]:
 rng=np.random.default_rng(seed);x=rng.normal(size=(1088,8));noise=rng.normal(size=(1088,248))
 y=(x[:,0]+.7*x[:,1]-.4*x[:,2]+.25*rng.normal(size=1088)>0).astype(int)
 tables[str(seed)]={'latent':x.tolist(),'noise':noise.tolist(),'y':y.tolist(),'support_pool':list(range(1024)),'query_ids':list(range(1024,1088))}
inputs={'generator':'NumPy default_rng(seed); normal latent1088x8, normal noise1088x248, binary noisy linear target; no split/seed selection','tables':tables}
b=(json.dumps(inputs,separators=(',',':'))+'\n').encode();(E/'inputs.json').write_bytes(b)
configs=[]
for seed in [0,1,2]:
 for n in [64,256,1024]:
  for f in [8,64,256]:
   for widening in ['noise','copies']:
    for kernel in ['linear','softmax']:
     configs.append(dict(name=f's{seed}-n{n}-f{f}-{widening}-{kernel}',seed=seed,support=n,features=f,widening=widening,kernel=kernel))
protocol={'experiment':'B04A-COURSE-KERNEL-STRESS','status':'FROZEN_BEFORE_EXECUTION','inputs_sha256':hashlib.sha256(b).hexdigest(),'configs':configs,'classes':[0,1],'capacity':10,'dtype':'float64','ensemble_size':1,'warm_repeats':3,'eps':1e-6,'threads':1,'feature_rule':'latent8 unchanged; append independent noise or repeated latent columns in order, then fit support mean/std; project to16 dimensions with default_rng(4000+F), normal(F,16)/sqrt(F)','softmax':'exp(q k^T/sqrt(16)), stable row normalization','linear':'ELU+1, associative normalized readout, epsilon1e-6, normalize final class row sums','scope':'Fixed random projection + kernel label averaging; no pretrained checkpoint, no optimization, no tuned hyperparameters','paired_control':'8-feature noise/copies inputs identical, all support prefixes nested, query labels isolated from predictor','class_boundary':[2,10,11],'forecast':{'configurations':108,'predictions':6912,'local_seconds_reserved':900,'local_cap_seconds':3600,'cloud_usd':0,'pilot':'No separate pilot; conservative per-process9s bound includes imports and three warm repeats. Stop at900s; preserve incomplete cells.'},'paper_lane':'INCOMPLETE_SOURCE_PROTOCOL','current_checkpoint_baseline':'NOT_RUN'}
(E/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
from _source_b04a import source_gate
(E/'source-gate.json').write_text(json.dumps(source_gate(S),indent=2)+'\n')
stream=io.StringIO()
with contextlib.redirect_stdout(stream):np.__config__.show()
(E/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),torch=torch.__version__,sklearn=sklearn.__version__,machine=platform.machine(),logical_cpus=os.cpu_count(),worker_environment={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},numpy_build=stream.getvalue()),indent=2)+'\n')
print('Frozen',len(configs),'configurations, 6912 predictions; source gate closed')
