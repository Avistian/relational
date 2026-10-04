"""Source and portable-package regression checks; never print matched credentials."""
import ast,base64,hashlib,io,json,re,shutil,tempfile,unittest,zipfile
from pathlib import Path
from _source_b05 import source_gate
P=Path(__file__).resolve().parent
KEY=re.compile(rb'AIza[0-9A-Za-z_-]{35}')
class SourceSecretTests(unittest.TestCase):
 def test_saved_page_has_no_google_api_key(self):
  self.assertFalse(bool(KEY.search((P/'sources/b05/tabzilla-drive.html').read_bytes())),'Google API key pattern in saved Drive page')
 def test_reproducer_archive_has_no_google_api_key(self):
  with zipfile.ZipFile(P/'evidence/b05/reproducer.zip') as z:
   for name in z.namelist():
    if name.endswith(('.html','.json','.py','.md','.txt')):
     self.assertFalse(bool(KEY.search(z.read(name))),'Google API key pattern in archive member '+name)
 def test_notebooks_embed_only_the_sanitized_archive(self):
  expected=(P/'evidence/b05/reproducer.zip').read_bytes()
  for path in [P/'b05-tabdpt-real-data-retrieval.ipynb',P/'solutions/b05-tabdpt-real-data-retrieval.ipynb']:
   book=json.loads(path.read_text());payloads=[]
   for cell in book['cells']:
    if cell['cell_type']!='code':continue
    source=''.join(cell['source'])
    if not source.startswith('payload = '):continue
    tree=ast.parse(source)
    payloads.extend(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
   self.assertEqual(len(payloads),1,'Expected one source packet')
   self.assertTrue(base64.b64decode(payloads[0])==expected,'Notebook packet differs from sanitized archive: '+path.name)
 def test_valid_checksum_cannot_authorize_a_key(self):
  with tempfile.TemporaryDirectory(prefix='b05-key-regression-') as td:
   root=Path(td)/'source';shutil.copytree(P/'sources/b05',root)
   path=root/'tabzilla-drive.html';path.write_bytes(b'<script>var key="'+b'AI'+b'za'+b'0'*35+b'"</script>')
   mpath=root/'manifest.json';m=json.loads(mpath.read_text());m['files']['tabzilla-drive.html']=hashlib.sha256(path.read_bytes()).hexdigest();mpath.write_text(json.dumps(m))
   with self.assertRaisesRegex(AssertionError,'Google API key'):source_gate(root)
if __name__=='__main__':unittest.main()
