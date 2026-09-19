"""Execute source identity, API contracts, leakage intervention and real rows."""
import hashlib, importlib.metadata, json, platform
from pathlib import Path
import pandas as pd
import torch
import torch_frame
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from _check_l075 import check
from relkit.frame_l075 import fixture, schema, fit_materializer, real_table_demo
ROOT=Path(__file__).resolve().parent

def run():
    torch.set_num_threads(1)
    provenance=json.loads((ROOT/'_sources_l075.json').read_text())
    source_check={}
    for name, record in provenance['files'].items():
        data=(ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==record['sha256']
        if '/torch_frame/' in name:
            local=Path(torch_frame.__file__).parent/name.split('/torch_frame/',1)[1]
            assert local.read_bytes()==data, f'Installed source differs: {name}'
            source_check[name]='BYTE_MATCH'
    result=check()
    train, query=fixture(); ds, held=fit_materializer(train,query,schema())
    # Same training values [10,20,30,NaN]; add query [1000,NaN].
    # Passing split_col still computes statistics on the entire DataFrame.
    combined=pd.concat([train.assign(split=0),query.assign(split=2)],ignore_index=True)
    leaky=Dataset(combined[['amount','split']],col_to_stype={'amount':torch_frame.numerical},split_col='split').materialize()
    mean=leaky.col_stats['amount'][StatType.MEAN]
    assert mean==265.0
    result['leakage']={'train_only_mean':20.,'materialize_all_with_split_col_mean':mean,
                       'same_training_rows':True,'training_mean_shift':mean-20.}
    result['train_std']=float(ds.col_stats['amount'][StatType.STD])
    result['materialized']={s.value:{'columns':names,'shape':list(ds.tensor_frame.feat_dict[s].shape)} for s,names in ds.tensor_frame.col_names_dict.items()}
    result['real_table']=real_table_demo()
    result['source_identity']=source_check
    result['versions']={k:importlib.metadata.version(k) for k in ['pytorch-frame','torch','numpy','pandas','scikit-learn']}
    result['python']=platform.python_version()
    result['implementation_sha256']=hashlib.sha256((ROOT/'relkit/frame_l075.py').read_bytes()).hexdigest()
    result['data_sha256']=hashlib.sha256((ROOT/'data/cache/credit_g.parquet').read_bytes()).hexdigest()
    result['scope']={'api_demonstration':'PASS','paper_benchmark':'NOT_RUN','paper_result_parity':'INCOMPARABLE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
    (ROOT/'_verify_l075_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ['real_table','source_identity']})
    return result
if __name__=='__main__':run()
