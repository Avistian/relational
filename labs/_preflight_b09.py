"""Fail closed before spending benchmark time on missing/corrupt pinned inputs."""
import hashlib,json,struct,sys
from pathlib import Path
P=Path(__file__).resolve().parent

def verify_files(root,entries):
 for relative,digest in entries.items():
  path=Path(root)/relative
  if not path.is_file():raise ValueError('Missing pinned file: '+str(path))
  if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=digest:raise ValueError('Changed pinned file: '+str(path))

def main():
 s=P/'sources/b09';weights=json.loads((s/'weights.json').read_text())
 verify_files('/tmp/b09-weights',{x['model']+'/'+x['file']:x['sha256'] for x in weights})
 for entry in json.loads((s/'provenance.json').read_text()):
  if 'commit' in entry:verify_files(s,{entry['name']+'.tar.gz':entry['sha256']})
 verify_files('/tmp/b09-src',json.loads((s/'extracted-source-hashes.json').read_text()))
 seal=P/'evidence/b09/artifact-manifest.json'
 if seal.exists():verify_files(P.parent,json.loads(seal.read_text())['files'])
 print('Pinned weights, archives and any sealed package verified')
if __name__=='__main__':main()
