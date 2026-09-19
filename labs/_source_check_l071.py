"""Download only pinned public source; execute just its NumPy corruption function."""
import ast,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
import torch
from relkit.vime_l071 import corrupt
ROOT=Path(__file__).parent
COMMIT='996c58cf4c570061b30c38ecf2a754a9af85aafd'
files=['vime_self.py','vime_utils.py','vime_semi.py','data_loader.py','main_vime.py']
rows=[];sources={}
for name in files:
    url=f'https://raw.githubusercontent.com/jsyoon0823/VIME/{COMMIT}/{name}'
    raw=urllib.request.urlopen(url,timeout=30).read()
    sources[name]=raw.decode()
    rows.append(dict(file=name,url=url,sha256=hashlib.sha256(raw).hexdigest()))
node=next(n for n in ast.parse(sources['vime_utils.py']).body if isinstance(n,ast.FunctionDef) and n.name=='pretext_generator')
ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned pretext_generator>','exec'),ns)
x=np.array([[0.,1.,.5],[1.,1.,0.],[0.,0.,.5]],dtype=np.float32)
m=np.array([[1.,1.,0.],[0.,1.,1.],[1.,0.,1.]],dtype=np.float32)
np.random.seed(71);expected=ns['pretext_generator'](m,x)
np.random.seed(71);donors=np.stack([np.random.permutation(len(x)) for _ in range(x.shape[1])],axis=1)
actual=corrupt(torch.from_numpy(x),torch.from_numpy(m),torch.from_numpy(donors))
for a,b in zip(actual,expected):np.testing.assert_array_equal(a.numpy(),b)
assert "m_new = 1 * (x != x_tilde)" in sources['vime_utils.py']
assert "tf.nn.moments(yv_hat_logit, axes = 0)[1]" in sources['vime_semi.py']
assert "loss_weights={'mask':1, 'feature':alpha}" in sources['vime_self.py']
out=dict(commit=COMMIT,files=rows,corruption_parity='PASS: exact values and actual-change targets on collision fixture',
         source_observations=['Fixed corruption generated before self-supervised fit','All-coordinate MSE',
                              'Frozen encoder.predict in downstream stage','Population logit variance across views'],
         limitations=['No historical TensorFlow training replay','PyTorch optimizer/default initialization differ'])
(ROOT/'_sources_l071.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['corruption_parity'])
