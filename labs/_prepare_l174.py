"""Authenticate upstream evidence and pin a portable adaptation packet."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from relkit.multitask_l173 import load_packet
from relkit.finetune_l174 import adaptation_split
P=Path(__file__).resolve().parent;E=P/'evidence/l174';old=P/'evidence/l173'
a,t,m=load_packet(old)
seal=json.loads((old/'artifact-manifest.json').read_text())
inputs={}
for n,h in m['inputs'].items():
    assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n;inputs[n]=h
for n,h in seal['files'].items():
    assert hashlib.sha256((old/n).read_bytes()).hexdigest()==h,n
for n in ['population.npz','tasks.json']:
    shutil.copyfile(old/n,E/n);inputs['evidence/l173/'+n]=hashlib.sha256((old/n).read_bytes()).hexdigest()
lineage=[]
for seed in range(3):
    result=json.loads((old/f'cell-{seed}/result.json').read_text())
    chosen=int(np.argmin([v['validation']['macro_loss'] for v in result['epochs']]))+1
    assert chosen==result['selected_epoch']
    name=f'cell-{seed}/epoch-{chosen}.pt';shutil.copyfile(old/name,E/f'source-{seed}.pt')
    lineage.append(dict(seed=seed,arm='cell',selected_epoch=chosen,original='evidence/l173/'+name,sha256=seal['files'][name],selection='2005 validation macro loss only'))
    for n in [name,f'cell-{seed}/result.json']:inputs['evidence/l173/'+n]=seal['files'][n]
split=adaptation_split(a['dates'][a['cell_ids']]);counts=[int((split==i).sum()) for i in range(3)]
assert counts==[6429,6074,106757]
config=dict(name='L174 F1 Temporal Adaptation',arms=['freeze','full','adapter','scratch'],seeds=[0,1,2],epochs=10,batch_size=1024,optimizer='Adam',learning_rate=.001,weight_decay=0,loss_weighting='equal_task_from_2006_counts',adapter=[32,8,32],adapter_initialization='zero_up_random_down',head_initialization='retained_except_scratch',preprocessing='L173 pre2005 frozen',adaptation_year=2006,validation_year=2007,test_from=2008,split_counts=counts,selection='earliest minimum validation macro loss',test_exposure='already evaluated in L173; retrospective descriptive experiment',lineage=lineage)
(E/'config.json').write_text(json.dumps(config,indent=2)+'\n')
(E/'coverage.json').write_text(json.dumps(dict(excluded_pre2006=int((split<0).sum()),tasks=[dict(name=s['name'],counts=[int(((a['task']==i)&(split==j)).sum()) for j in range(3)]) for i,s in enumerate(t)]),indent=2)+'\n')
files={n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['population.npz','tasks.json','config.json','coverage.json']+[f'source-{s}.pt' for s in range(3)]}
sources={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['relkit/multitask_l173.py','relkit/finetune_l174.py','_prepare_l174.py','_run_l174.py']}
(E/'manifest.json').write_text(json.dumps(dict(files=files,inputs=inputs,source_files=sources),indent=2)+'\n')
print('Authenticated full packet:',counts)
