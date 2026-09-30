"""One documented feature-type reconstruction for the released checkpoint.

Fresh inference classifies qualifying.position as categorical. Saved checkpoint
weights require two numerical columns. Their mean/std identify number,position.
This is compatibility evidence, not recovery of the historical stypes cache.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from relbench.datasets import get_dataset
from _full_l141 import sha,full_run

def prepare_checkpoint_compatibility(prepared_root,output_root):
    root=Path(prepared_root).resolve();out=Path(output_root);out.mkdir(parents=True,exist_ok=False)
    data,stats=torch.load(root/'graph.pt',weights_only=False)
    cp=torch.load(root/'released.pth',map_location='cpu',weights_only=False)
    dataset=get_dataset('rel-f1');dataset.cache_dir=str(root/'unpacked');dataset.get_db.cache_clear();df=dataset.get_db().table_dict['qualifying'].df
    columns={col:typ for typ,cols in data['qualifying'].tf.col_names_dict.items() for col in cols}
    assert columns['position']==stype.categorical and columns['number']==stype.numerical
    old={k:str(v) for k,v in columns.items()};columns['position']=stype.numerical
    frame=Dataset(df,col_to_stype=columns).materialize()
    assert frame.tensor_frame.col_names_dict[stype.numerical]==['number','position']
    prefix='encoder.encoders.qualifying.encoder.encoder_dict.numerical.'
    checks={}
    for kind,values in [('mean',df[['number','position']].mean().to_numpy()),('std',df[['number','position']].std(ddof=0).to_numpy())]:
        actual=cp[prefix+kind].numpy();np.testing.assert_allclose(values,actual,rtol=1e-6,atol=1e-6)
        checks[kind]={'table':values.tolist(),'checkpoint':actual.tolist()}
    data['qualifying'].tf=frame.tensor_frame;stats['qualifying']=frame.col_stats
    torch.save((data,stats),out/'graph.pt')
    for name in ['cache','unpacked','released.pth']:(out/name).symlink_to(root/name,target_is_directory=name!='released.pth')
    info=json.loads((root/'prepared.json').read_text());info['parent_graph_sha256']=info['graph_sha256'];info['graph_sha256']=sha(out/'graph.pt')
    info['preprocessing']='CHECKPOINT_COMPATIBILITY_RECONSTRUCTION; qualifying.position categorical->numerical'
    info['compatibility']={'before':old,'after':{k:str(v) for k,v in columns.items()},'evidence':checks,'historical_stypes_cache':'NOT_ESTABLISHED','unchanged_other_tables':True}
    (out/'prepared.json').write_text(json.dumps(info,indent=2));return info
