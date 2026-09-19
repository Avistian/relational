"""Fresh download plus full training intervention on held-out labels."""
from pathlib import Path
import tempfile,json,shutil
from relkit.gcn_l082 import load_cora,train_cora
import torch
LAB=Path(__file__).resolve().parent
torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='l082-download-') as tmp:
    root=Path(tmp);shutil.copy(LAB/'_sources_l078.json',root/'_sources_l078.json')
    data=load_cora(root)
    assert data[0].shape==(2708,1433)
    original=train_cora(data,82)
    xx,ss,yy,tr,va,te=data;changed=yy.clone();changed[te]=(changed[te]+1)%7
    intervention=train_cora((xx,ss,changed,tr,va,te),82)
    assert original['epochs']==intervention['epochs']
    assert original['validation_loss']==intervention['validation_loss']
    report={'status':'PASS','fresh_download_files':len(list((root/'data/l078').iterdir())),'training_test_label_intervention':'Identical validation trace and stopping epoch','original_test_accuracy':original['test_accuracy'],'changed_label_test_accuracy':intervention['test_accuracy']}
    (LAB/'_isolation_l082_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
