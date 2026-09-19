"""Complete QM9 reconstruction; historical IDs and tuning configurations are unavailable."""
import copy,csv,hashlib,json,random,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data,Batch
from rdkit import Chem,RDConfig
from rdkit.Chem import ChemicalFeatures
from relkit.mpnn_l081 import SparseGGNN

DATA_URL='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
EXCLUDE_URL='https://ndownloader.figshare.com/files/3195404'
EXPECTED={'qm9.zip':'f11d3f8ecc3097a72656a35c4847767852e6f346bf99a54619a1a3b6389ddc02','uncharacterized.txt':'3aa5115d540b356de94791d4a74c3bf1ed91c469ecf52a4f5d7cc0506fe02e24'}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_qm9(root,limit=None):
    """13 Table-1-style features, implicit H, public exclusions, recorded failures.

    RDKit BaseFeatures donor/acceptor definitions replace the missing historical reader.
    """
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for name,url in [('qm9.zip',DATA_URL),('uncharacterized.txt',EXCLUDE_URL)]:
        path=root/name
        if not path.exists():
            tmp=path.with_suffix('.part');urllib.request.urlretrieve(url,tmp);tmp.replace(path)
        if sha256(path)!=EXPECTED[name]:raise ValueError(f'Input checksum mismatch: {name}')
    with zipfile.ZipFile(root/'qm9.zip') as z:
        for name in ('gdb9.sdf','gdb9.sdf.csv'):
            if not (root/name).exists():
                import shutil
                with z.open(name) as src,open(root/name,'wb') as dst:shutil.copyfileobj(src,dst)
            # Validate cached extraction against the already SHA256-pinned archive.
            import zlib
            crc=0
            with open(root/name,'rb') as f:
                for chunk in iter(lambda:f.read(1<<20),b''):crc=zlib.crc32(chunk,crc)
            if crc!=z.getinfo(name).CRC:raise ValueError(f'Cached extraction differs from archive: {name}')
    with open(root/'gdb9.sdf.csv') as f:targets={r['mol_id']:float(r['mu']) for r in csv.DictReader(f)}
    excluded=set()
    for line in (root/'uncharacterized.txt').read_text().splitlines()[9:-2]:
        words=line.split()
        if words and words[0].isdigit():excluded.add(int(words[0]))
    factory=ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'))
    bond_types={Chem.BondType.SINGLE:0,Chem.BondType.DOUBLE:1,Chem.BondType.TRIPLE:2,Chem.BondType.AROMATIC:3}
    atom_types=['H','C','N','O','F'];hybrids=[Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3]
    graphs=[];failed=[]
    for i,mol in enumerate(Chem.SDMolSupplier(str(root/'gdb9.sdf'),removeHs=False,sanitize=False)):
        if i+1 in excluded:continue
        if mol is None:failed.append(i+1);continue
        name=mol.GetProp('_Name')
        try:
            Chem.SanitizeMol(mol);mol=Chem.RemoveHs(mol);flags={'Donor':set(),'Acceptor':set()}
            for feature in factory.GetFeaturesForMol(mol):
                if feature.GetFamily() in flags:flags[feature.GetFamily()].update(feature.GetAtomIds())
            features=[]
            for a in mol.GetAtoms():
                j=a.GetIdx();features.append([float(a.GetSymbol()==t) for t in atom_types]+[a.GetAtomicNum(),float(j in flags['Acceptor']),float(j in flags['Donor']),float(a.GetIsAromatic())]+[float(a.GetHybridization()==t) for t in hybrids]+[a.GetTotalNumHs()])
            edges=[];kinds=[]
            for b in mol.GetBonds():
                u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx();edges.extend([(u,v),(v,u)]);kinds.extend([bond_types[b.GetBondType()]]*2)
            graphs.append(Data(x=torch.tensor(features,dtype=torch.float32),edge_index=torch.tensor(edges,dtype=torch.long).reshape(-1,2).t().contiguous(),edge_type=torch.tensor(kinds,dtype=torch.long),y=torch.tensor([targets[name]]),molecule_id=torch.tensor([i+1])))
        except (ValueError,RuntimeError,KeyError):failed.append(i+1);continue
        if limit is not None and len(graphs)>=limit:break
    info={'urls':[DATA_URL,EXCLUDE_URL],'input_sha256':{n:sha256(root/n) for n in ['qm9.zip','uncharacterized.txt']},'molecules_loaded':len(graphs),'failed_ids':failed,'limit':limit,'features':13,'explicit_hydrogens':False,'historical_population':130462,'population_match':len(graphs)==130462 if limit is None else False}
    return graphs,info


def split_ids(n,seed=81,full=False):
    """Fresh label-blind IDs, not the unreleased historical split."""
    order=np.random.default_rng(seed).permutation(n);nv=10000 if full else max(1,n//10)
    if n<=2*nv:raise ValueError('Insufficient molecules for split')
    return {'train':order[2*nv:].tolist(),'validation':order[:nv].tolist(),'test':order[nv:2*nv].tolist()}


def pack(graphs,ids,device):
    return Batch.from_data_list([graphs[int(i)] for i in ids]).to(device)


def predict(model,graphs,ids,mean,std,device):
    model.eval();pred=[]
    with torch.no_grad():
        for start in range(0,len(ids),128):
            b=pack(graphs,ids[start:start+128],device)
            pred.extend((model(b.x,b.edge_index,b.edge_type,b.batch)*std+mean).cpu().tolist())
    return np.asarray(pred)


def fit_trial(graphs,split,config,seed,device='cpu'):
    """Normalized MSE, Adam, linear LR decay, validation-MAE checkpoint selection."""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=SparseGGNN(width=config['width'],steps=config['rounds']).to(device)
    y=np.array([float(graphs[i].y) for i in split['train']]);mean=float(y.mean());std=float(y.std())
    if not std>0:raise ValueError('Zero training-target variance')
    opt=torch.optim.Adam(model.parameters(),lr=config['lr']);rng=np.random.default_rng(seed)
    best=float('inf');history=[];order=rng.permutation(split['train']);cursor=0
    val_y=np.array([float(graphs[i].y) for i in split['validation']])
    for step in range(config['updates']):
        if cursor>=len(order):order=rng.permutation(split['train']);cursor=0
        ids=order[cursor:cursor+20];cursor+=20;b=pack(graphs,ids,device)
        fraction=step/max(1,config['updates']-1);decay=max(0,(fraction-config['decay_start'])/(1-config['decay_start']))
        for group in opt.param_groups:group['lr']=config['lr']*(1-decay*(1-config['final_factor']))
        model.train();opt.zero_grad();loss=((model(b.x,b.edge_index,b.edge_type,b.batch)-(b.y-mean)/std)**2).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),4.0);opt.step()
        if (step+1)%config['validate_every']==0 or step+1==config['updates']:
            mae=float(np.abs(predict(model,graphs,split['validation'],mean,std,device)-val_y).mean());history.append({'step':step+1,'validation_mae':mae})
            if mae<best:best=mae;best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()});best_step=step+1
    model.load_state_dict(best_state)
    return model,{'seed':seed,'config':config,'mean':mean,'std':std,'validation_mae':best,'best_step':best_step,'history':history}


def run_reconstruction(root,output,preset='smoke',seed=81,device='cpu'):
    """Named-target attempt; test labels used only after validation selects one trial."""
    import platform,rdkit,torch_geometric
    start=time.time();torch.set_num_threads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise FileExistsError('Use a new output directory to preserve prior evidence')
    limit,updates,trials={'smoke':(512,40,1),'closer':(12000,10000,3),'paper-budget':(None,3000000,50)}[preset]
    graphs,data=load_qm9(root,limit);split=split_ids(len(graphs),seed,full=limit is None)
    molecule_ids={k:[int(graphs[i].molecule_id) for i in v] for k,v in split.items()}
    (output/'split.json').write_text(json.dumps(molecule_ids,indent=2)+'\n')
    rng=np.random.default_rng(seed);records=[];best=float('inf')
    for trial in range(trials):
        config={'width':50,'rounds':int(rng.integers(3,9)) if trials>1 else 6,'lr':float(rng.uniform(1e-5,5e-4)) if trials>1 else .00013,'decay_start':float(rng.uniform(.1,.9)),'final_factor':float(rng.uniform(.01,1)),'updates':updates,'validate_every':20 if preset=='smoke' else 5000}
        model,record=fit_trial(graphs,split,config,seed+trial,device);records.append(record)
        if record['validation_mae']<best:
            best=record['validation_mae'];winner=trial;torch.save({'state_dict':model.state_dict(),'record':record},output/'selected.pt')
        (output/'trials.json').write_text(json.dumps(records,indent=2)+'\n');print(f'trial {trial+1}/{trials}: validation MAE {record["validation_mae"]:.5f} D',flush=True)
    selected=torch.load(output/'selected.pt',map_location=device,weights_only=False);record=selected['record']
    model=SparseGGNN(width=record['config']['width'],steps=record['config']['rounds']).to(device);model.load_state_dict(selected['state_dict'])
    predictions=predict(model,graphs,split['test'],record['mean'],record['std'],device);test_y=np.array([float(graphs[i].y) for i in split['test']]);mae=float(np.abs(predictions-test_y).mean())
    np.savez(output/'predictions.npz',molecule_id=molecule_ids['test'],prediction=predictions,target=test_y)
    result={'preset':preset,'status':'INCOMPARABLE','target':'Gilmer 2017 supplement Table 3 GG-NN mu','paper_mae_debye':.394,'test_mae_debye':mae,'test_error_ratio':mae/.1,'selected_trial':winner,'trials':records,'data':data,'split_seed':seed,'split_sha256':sha256(output/'split.json'),'prediction_sha256':sha256(output/'predictions.npz'),'seconds':time.time()-start,'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'rdkit':rdkit.__version__,'torch_geometric':torch_geometric.__version__},'deviations':['fresh split IDs','modern sanitized QM9 population and donor/acceptor definitions','fixed width 50; historical trial configurations unavailable','PyTorch initialization/Adam; no TensorFlow runtime parity','train-only target scaling; historical fit scope unstated','fixed validation interval; historical cadence unavailable'],'full_historical_reproduction':'NOT_RUN'}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
