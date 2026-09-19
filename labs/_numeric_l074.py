"""Regression fixture for constant-column power-transform overflow."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import PowerTransformer
from relkit.carte_l074 import normalize_numeric
ROOT=Path(__file__).resolve().parent
x=pd.read_parquet(ROOT/'data/l074/wine_pl.parquet').drop(columns='price');tr=np.random.default_rng(76).permutation(len(x))[:64]
old=PowerTransformer().fit(x.iloc[tr][['volume']]);bad=old.transform(x[['volume']])
assert np.nanmax(abs(bad))>np.finfo(np.float32).max
new,policy=normalize_numeric(x,tr)
assert np.nanmax(abs(new['volume']))<1e6
assert policy['volume']=='standard_constant'
for d in ['wine_pl','wine_dot_com_prices','wine_vivino_price']:
 frame=pd.read_parquet(ROOT/'data/l074'/f'{d}.parquet')
 for seed in range(3):
  ids=np.random.default_rng(seed+74).permutation(len(frame))[:64]
  out,_=normalize_numeric(frame,ids)
  for col in out.select_dtypes(include='number'):
   assert np.isfinite(out[col].dropna().astype('float32')).all()
# Changing evaluation rows must not alter transformed training rows.
x2=x.copy();x2.iloc[[i for i in range(len(x)) if i not in tr],x.columns.get_loc('volume')]=1e7
other,_=normalize_numeric(x2,tr)
assert np.allclose(new.iloc[tr]['volume'],other.iloc[tr]['volume'],equal_nan=True)
r={'status':'PASS','dataset':'wine_pl','seed':2,'train_observed_volume_unique':x.iloc[tr]['volume'].dropna().unique().tolist(),'original_power_lambda':old.lambdas_.tolist(),'original_max_abs':float(np.nanmax(abs(bad))),'fixed_policy':policy,'all_default_numeric_splits_finite':True,'evaluation_invariance':True}
(ROOT/'_numeric_l074_results.json').write_text(json.dumps(r,indent=2));print(r)
