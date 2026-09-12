"""Fetch immutable original PFN sources for validation only; no installation."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COMMIT='9c20031b355923bdd456d5fcfe4e98092b016b97'
FILES=['transformer.py','bar_distribution.py','utils.py','train.py','priors/fast_gp.py','notebooks/SetupForGPFittingExperiments.ipynb']
def fetch():
 folder=ROOT/'data/cache/l061-source'/COMMIT;folder.mkdir(parents=True,exist_ok=True)
 records=[]
 for filename in FILES:
  url=f'https://raw.githubusercontent.com/automl/TransformersCanDoBayesianInference/{COMMIT}/{filename}'
  path=folder/filename;path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():urllib.request.urlretrieve(url,path)
  records.append(dict(path=filename,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 return folder,records
if __name__=='__main__':print(json.dumps(fetch()[1],indent=2))
