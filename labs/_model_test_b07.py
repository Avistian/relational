"""Information-boundary and pretrained graph behavior tests."""
import numpy as np
import pandas as pd
import torch
from relkit.semantic_b07 import encode_table
from relkit.carte_b07 import load_encoder,make_graph,embed
from pathlib import Path
p=Path(__file__).resolve().parent
v=np.load(p/'data/b07/vectors.npz');vectors=dict(zip(v['names'].tolist(),v['vectors']))
model=load_encoder(p/'data/b07/selected_checkpoint.pt',True,0)
frame=pd.DataFrame({'6':[np.nan,1.,2.]})
x=encode_table(model,frame,vectors,make_graph,embed)
assert x.shape==(3,300) and np.array_equal(x[0],np.zeros(300))
np.testing.assert_array_equal(x[1:],embed(model,[make_graph({'6':1.},vectors),make_graph({'6':2.},vectors)]))
y=encode_table(model,frame.iloc[[2,0,1]],vectors,make_graph,embed)
np.testing.assert_array_equal(y,x[[2,0,1]])
print('PASS: empty-row fallback preserves rows; nonempty rows and row permutation exact')
