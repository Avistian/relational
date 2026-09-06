"""Cache three numeric tasks from the authors' January 2023 OpenML suite 337."""
import json,hashlib
from pathlib import Path
from sklearn.datasets import fetch_openml
IDS={'electricity':44120,'MagicTelescope':44125,'bank-marketing':44126}
if __name__=='__main__':
 out=Path(__file__).parent/'data/cache/l051';out.mkdir(parents=True,exist_ok=True);records={}
 for name,did in IDS.items():
  b=fetch_openml(data_id=did,as_frame=True,parser='auto',data_home=str(out/'openml'))
  df=b.data.copy();df['__target__']=b.target
  p=out/f'{name}.parquet';df.to_parquet(p,index=False)
  records[name]={'openml_id':did,'rows':len(df),'features':len(df.columns)-1,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'url':f'https://www.openml.org/d/{did}'}
  print(name,records[name],flush=True)
 (Path(__file__).parent/'_data_l051.json').write_text(json.dumps(records,indent=2))
