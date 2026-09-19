"""Fetch the pinned release's teaching-relevant source and license."""
from pathlib import Path
import urllib.request, hashlib, json
REV='d998aae368db6a4e36139ccc56bd54579a70874b'
ROOT=Path(__file__).resolve().parent
FILES=['torch_frame/_stype.py','torch_frame/data/dataset.py','torch_frame/data/stats.py','torch_frame/data/mapper.py','torch_frame/nn/encoder/stype_encoder.py','torch_frame/nn/encoder/stypewise_encoder.py','LICENSE']
manifest={'paper':'https://arxiv.org/html/2404.00776v2','release':REV,'package':'pytorch-frame==0.3.0','files':{}}
for file in FILES:
    url=f'https://raw.githubusercontent.com/pyg-team/pytorch-frame/{REV}/{file}'
    data=urllib.request.urlopen(url).read()
    path=ROOT/'sources/frame-l075'/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest['files'][str(path.relative_to(ROOT))]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
(ROOT/'_sources_l075.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Pinned source and license saved')
