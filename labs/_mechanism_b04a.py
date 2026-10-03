"""Execute unmodified archived operator/preprocess ASTs with narrow dependencies."""
import ast,hashlib,io,json,tarfile,types,contextlib
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from relkit.scaling_b04a import linear_readout,class_values
P=Path(__file__).resolve().parent;E=P/'evidence/b04a';S=P/'sources/b04a'
with tarfile.open(S/'code.tar.gz') as t:
 source=t.extractfile('ticl/models/linear_attention.py').read().decode();wrapper=t.extractfile('ticl/prediction/tabpfn.py').read().decode()
names=['FeatureMap','ActivationFunctionFeatureMap','LinearAttention']
ns={'torch':torch,'Module':torch.nn.Module}
for name in names:
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name)
 if name=='LinearAttention':ns['elu_feature_map']=ns['ActivationFunctionFeatureMap'].factory(lambda x:torch.nn.functional.elu(x)+1)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<archived:'+name+'>','exec'),ns)
rng=np.random.default_rng(304);errors=[]
for n in [1,7,64,1024]:
 q=rng.normal(size=(11,16));k=rng.normal(size=(n,16));v=rng.normal(size=(n,3))
 out=ns['LinearAttention'](16).double()(torch.tensor(q)[None,:,None,:],torch.tensor(k)[None,:,None,:],torch.tensor(v)[None,:,None,:],None,None,None).detach().numpy()[0,:,0,:]
 delta=float(np.max(np.abs(out-linear_readout(q,k,v))));assert delta<1e-12;errors.append({'support':n,'max_abs':delta})
# Source preprocess method only: no checkpoint loading, training or model inference.
cls=next(n for n in ast.parse(wrapper).body if isinstance(n,ast.ClassDef) and n.name=='TabPFNClassifier');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess')
w={'np':np,'torch':torch,'LabelEncoder':LabelEncoder};exec(compile(ast.Module(body=[method],type_ignores=[]),'<archived:preprocess>','exec'),w)
boundaries=[]
for count in [2,10,11]:
 obj=types.SimpleNamespace(max_num_features=100,max_num_classes=10,max_num_train_samples=1024);y=np.tile(np.arange(count),3);original=y.copy()
 with contextlib.redirect_stdout(io.StringIO()):w['preprocess'](obj,np.zeros((len(y),8)),y,True)
 try:class_values(original,list(range(count)),10);course='ACCEPT'
 except ValueError:course='REJECT_CAPACITY'
 boundaries.append({'input_classes':count,'configured_capacity':10,'source_output_classes':len(np.unique(obj.y_)),'source_mode':obj.classes_mode,'course_action':course,'original_labels':original.tolist(),'source_labels':obj.y_.tolist()})
assert boundaries[-1]['source_output_classes']==10 and boundaries[-1]['course_action']=='REJECT_CAPACITY'
r={'status':'PASS','operator_parity':errors,'tolerance':1e-12,'class_boundary':boundaries,'scope':'Unmodified archived operator and preprocessing method only; random inputs, CPU float64, no pretrained model','torch':torch.__version__}
(E/'mechanism.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS upstream operator parity and 2/10/11 class boundary')
