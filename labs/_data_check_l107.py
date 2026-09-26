"""Full released input transform audit using original functions and independent raw parsing."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P/'relkit'),str(P/'sources/l107/original')]
import sbm_l107 as s,taskers_utils as tu
if __name__=='__main__':
 torch.set_num_threads(1);d=s.load_sbm(P/'data/l107');raw=pd.read_csv(P/'data/l107/sbm.csv').to_numpy(dtype=np.int64);times=raw[:,3]-raw[:,3].min()
 data={'idx':torch.tensor(np.column_stack([raw[:,:2],times])),'vals':torch.ones(len(raw),dtype=torch.long)}
 max_error=0
 for t in range(50):
  a=tu.get_sp_adj(data,t,True,1);np.testing.assert_array_equal(a['idx'].numpy(),d['pairs'][t])
  f=tu.get_1_hot_deg_feats(a,d['f'],d['n']);dense=torch.sparse_coo_tensor(f['idx'].T,f['vals'],(d['n'],d['f'])).to_dense()
  torch.testing.assert_close(dense.float(),d['features'][t])
  normal=tu.normalize_adj(a,d['n']);dense=torch.sparse_coo_tensor(normal['idx'].T,normal['vals'],(d['n'],d['n'])).to_dense();err=float((dense-d['adj'][t].to_dense()).abs().max());max_error=max(max_error,err);assert err<1e-6
 # Authenticate TGN's cached arrays against a fresh raw parse, including every feature.
 raw=pd.read_csv(P/'data/l102/wikipedia.csv',header=None,skiprows=1);z=np.load(P/'data/l102/processed.npz');u=raw.iloc[:,0].to_numpy(dtype=np.int64)+1;v=raw.iloc[:,1].to_numpy(dtype=np.int64)+u.max()+1
 for key,a in [('u',u),('v',v),('t',raw.iloc[:,2].to_numpy()),('x',raw.iloc[:,4:].to_numpy(dtype=np.float32))]:np.testing.assert_array_equal(z[key],a)
 result={'status':'PASS','SBM_raw_rows':d['rows'],'snapshots':50,'source_adjacency_and_features':'EXACT','source_normalization_max_error':max_error,'Wikipedia_raw_rows':len(u),'all_cached_fields_reparsed':'EXACT','processed_sha256':hashlib.file_digest((P/'data/l102/processed.npz').open('rb'),'sha256').hexdigest()}
 (P/'_data_check_l107_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
