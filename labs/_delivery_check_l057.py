"""L057 actual artifact, notebook, row-boundary and copied-Pages checks."""
import base64,hashlib,json,re,shutil,tempfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit,unquote
import numpy as np
import nbformat
from PIL import Image
from bs4 import BeautifulSoup
from _build_l057 import ROOT,SLUG,build
site=ROOT.parent;r=json.loads((ROOT/'_verify_l057_v2_results.json').read_text())
old=json.loads((ROOT/'_verify_l057_results.json').read_text())
for f,sha in old['source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha,f
assert hashlib.sha256((ROOT/'_verify_l057_results.json').read_bytes()).hexdigest()==r['historical_result_sha256']
for f,sha in r['source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha,f
sources=json.loads((ROOT/'_sources_l057.json').read_text())
for f,item in sources['sources'].items():assert hashlib.sha256((ROOT/'sources/l057'/f).read_bytes()).hexdigest()==item['sha256']
for f,item in sources['audit_2026_09_10']['source_files'].items():assert hashlib.sha256((ROOT/'sources/l057'/f).read_bytes()).hexdigest()==item['sha256']
data=json.loads((ROOT/'_data_l057_v2.json').read_text())
for item in data['predictions']:
 p=ROOT/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 foldpath=p.with_name(p.stem+'-folds.json');assert hashlib.sha256(foldpath.read_bytes()).hexdigest()==item['folds_sha256']
 folds=json.loads(foldpath.read_text());q=np.load(p);n=len(q['y'])
 assert q['families'].tolist()==r['families'] and q['classes'].tolist()==[0,1]
 archive=ROOT/item['archive_path'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['archive_sha256']
 with np.load(archive) as prior:
  for key in ['oof','test']:np.testing.assert_array_equal(q[key][:,[0,2]],prior[key][:,[0,2]])
  for key in ['y','yt']:np.testing.assert_array_equal(q[key],prior[key])
 assert q['oof'].shape==(n,3) and q['test'].shape==(len(q['yt']),3)
 assert sorted(i for f in folds for i in f['held_rows'])==list(range(n))
 for f in folds:assert not set(f['held_rows'])&set(f['fit_rows']) and sorted(f['held_rows']+f['fit_rows'])==list(range(n))
 d=r['datasets'][item['dataset']];assert not set(d['dev_ids'])&set(d['test_ids'])
student=nbformat.read(ROOT/(SLUG+'.ipynb'),as_version=4);solution=nbformat.read(ROOT/'solutions'/(SLUG+'.ipynb'),as_version=4)
for nb,sol in [(student,False),(solution,True)]:
 assert [(c.cell_type,c.source) for c in nb.cells]==[(c.cell_type,c.source) for c in build(sol).cells],('builder drift',sol)
assert sum(c.cell_type=='code' and 'raise NotImplementedError' in c.source for c in student.cells)==5
assert all(c.execution_count is None and not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert '@colab-bootstrap' in next(c.source for c in student.cells if c.cell_type=='code')
assert not any('from relkit.cross_' in c.source or 'from relkit.tabm import' in c.source for c in student.cells)
payloads=[]
for c in student.cells:
 assert 'attachment:' not in c.source
 for token in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',c.source):
  b=base64.b64decode(token);Image.open(BytesIO(b)).verify();payloads.append(hashlib.sha256(b).hexdigest())
assert len(payloads)==6
expected_images={hashlib.sha256((ROOT/'figures/l057'/n).read_bytes()).hexdigest() for n in ['diversity.png','selection.png','results_v2.png','uncertainty_v2.png','portfolio.png']}
expected_images.update(hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'figures/architecture-revision').glob('0057-*.png'))
assert set(payloads)==expected_images
teacher=json.loads((ROOT/'data/cache/l057-student-audit.json').read_text());assert teacher==r['summary']
assert json.loads((ROOT/'data/cache/l057-student-ablation.json').read_text())==r['ablations']
manifest=json.loads((site/'lessons/manifest.json').read_text());e=next(x for x in manifest['lessons'] if x['id']==57)
assert e['labPath']=='labs/'+SLUG+'.ipynb' and e['published']
assert len({x['id'] for x in manifest['lessons']})==len(manifest['lessons'])
for name in ['index.html','notebooks.html']:assert BeautifulSoup((site/name).read_text(),'html.parser').find('meta',attrs={'name':'rdl-manifest-version'})['content']==str(manifest['version'])
stage=Path(tempfile.mkdtemp(prefix='l057-pages-'))
for folder in ['assets','lessons','reference']:shutil.copytree(site/folder,stage/folder)
(stage/'labs/html').mkdir(parents=True);(stage/'modal').mkdir()
shutil.copytree(ROOT/'figures/l057',stage/'labs/figures/l057')
shutil.copytree(ROOT/'figures/architecture-revision',stage/'labs/figures/architecture-revision')
shutil.copytree(ROOT/'html/architecture-review',stage/'labs/html/architecture-review')
for rel in ['html/'+SLUG+'.html',SLUG+'.ipynb']:shutil.copy2(ROOT/rel,stage/'labs'/rel)
for n in ['index.html','notebooks.html','flashcards.html']:shutil.copy2(site/n,stage/n)
for n in ['l057-reproduction.md','_verify_l057_results.json','_sources_l057.json','_data_l057.json','_source_check_l057_results.json','_verify_l057_v2_results.json','_data_l057_v2.json','_source_check_l057_v2_results.json']:
 assert n in (site/'.github/workflows/pages.yml').read_text();shutil.copy2(ROOT/n,stage/'labs'/n)
shutil.copy2(site/'modal/l057_paper_repro.py',stage/'modal/l057_paper_repro.py')
links=0
for rel in ['lessons/'+SLUG+'.html','reference/cross-family-ensembling.html','labs/html/'+SLUG+'.html']:
 p=stage/rel;soup=BeautifulSoup(p.read_text(),'html.parser')
 for el in soup.find_all(['a','img','script','link']):
  url=el.get('href') or el.get('src')
  if not url:continue
  u=urlsplit(url)
  if u.scheme or u.netloc:continue
  target=(p.parent/unquote(u.path)).resolve() if u.path else p
  assert target.is_relative_to(stage) and target.exists(),(rel,url)
  if u.fragment and target.suffix=='.html':assert BeautifulSoup(target.read_text(),'html.parser').find(id=unquote(u.fragment)),(rel,url)
  links+=1
mounts=re.findall(r"getElementById\('([^']+)'\)",(site/'assets/cross-ensemble-lesson.js').read_text());lesson=BeautifulSoup((site/'lessons'/(SLUG+'.html')).read_text(),'html.parser')
assert all(lesson.find(id=m) for m in mounts)
checks=dict(solution_code_cells=sum(c.cell_type=='code' for c in solution.cells),student_todos=5,portable_pngs=6,
 archived_three_family_runs=9,new_tabm_fold_fits=27,archived_other_fit_contexts=54,family_removal_comparisons=27,live_student_audit='MATCH',oof_coverage_and_disjointness='PASS',
 copied_pages_links=links,staging_path=str(stage),model_source_visible=True,selector_parity='Synthetic 5/5 MATCH; actual upstream 4/9 MATCH; 5 rounding differences; modified unrounded reference 9/9 MATCH',
 browser='Parent integration: reviews/lesson-quality-audit-047-070/057-browser.json',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',
 larger_run='NOT_RUN',modal_run='NOT_RUN',paper_figure6='INCOMPARABLE',figure_review='Synthetic arithmetic checked; portfolio PNG inspected; parent browser report covers final desktop/mobile delivery')
(ROOT/'_delivery_l057_results.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
