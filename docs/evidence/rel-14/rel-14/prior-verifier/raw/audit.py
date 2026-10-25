import pathlib, subprocess, hashlib, json, re, tempfile, shutil, sys
R=pathlib.Path('/tmp/lead-wt/v-rel14-final-20261025')
P=pathlib.Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(R),*a])
def run(label,args,cwd=R,expected=0):
 r=subprocess.run(args,cwd=cwd,text=True,capture_output=True)
 print(json.dumps(dict(label=label,argv=args,cwd=str(cwd),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)))
 assert r.returncode==expected,(label,r.returncode)
 return r
print('reviewer=verifier:rel-14-final; producer of browser images=external lead, not this verifier')
assert git('rev-parse','HEAD').decode().strip()=='67bf8c68d34e7cb33684570633915935cce5cb4e'
assert not git('status','--porcelain=v1')
base=git('show','edebddae:index.html').decode(); now=(R/'index.html').read_text()
old_data=re.search(r'<script type="application/json" id="lesson-search-data">(.*?)</script>',base,re.S).group(1)
js=(R/'assets/search-index.js').read_text(); data=json.loads(js[len('var LessonSearchIndex = '):-2])
assert json.loads(old_data)==data
assert re.search(r'<style>(.*?)</style>',base,re.S).group(1)==re.search(r'<style>(.*?)</style>',now,re.S).group(1)
expected=re.sub(r'<script type="application/json" id="lesson-search-data">.*?</script>','<script src="assets/search-index.js"></script>',base,flags=re.S)
expected=expected.replace('var index = null, failed = false;\n  try { index = JSON.parse(document.getElementById("lesson-search-data").textContent); } catch (e) { failed = true; }','var index = window.LessonSearchIndex;\n  var failed = !Array.isArray(index);')
assert expected==now
assert git('show','edebddae:assets/search.js')==(R/'assets/search.js').read_bytes()
assert '<a href="progress.html">Course progress</a>' in now and (R/'progress.html').is_file()
print(json.dumps(dict(index_bytes=len((R/'index.html').read_bytes()),entries=len(data),old_new_data_equal=True,css_byte_equal=True,search_module_byte_equal=True,rest_of_home_equal=True,rel13_link_exists=True)))
for rev in ['575b49b9','341875eb','67bf8c68']:
 b=git('show',rev+':tests/search/test_external_asset.py');print('frozen_test',rev,sha(b))
assert not git('diff','575b49b9','341875eb','--','tests')
assert not git('diff','341875eb','67bf8c68','--',':(exclude)docs/evidence/rel-14/rel-14')
ready=json.loads((R/'docs/evidence/rel-14/rel-14/READY.json').read_text())
for name,h in ready['evidence_sha256'].items():
 actual=sha((R/name).read_bytes()); assert actual==h,(name,actual,h)
 print('builder_hash_ok',name,h)
receipt=json.loads((R/'docs/evidence/rel-14/rel-14/canonical-tdd-receipt.json').read_text())
assert receipt['report_sha256']==sha((R/'docs/evidence/rel-14/rel-14/canonical-tdd-report.json').read_bytes())
for name,h in receipt['artifacts'].items():
 local=R/'docs/evidence/rel-14/rel-14'/('canonical-'+pathlib.Path(name).name)
 assert sha(local.read_bytes())==h
print('canonical retained logs/report hashes verified; no canonical execution performed')
run('current_freshness',[sys.executable,'-B','scripts/search_index.py','--check'])
# Copy only the generator, page, and asset; symlink real read-only inputs.
with tempfile.TemporaryDirectory(prefix='rel14-audit-') as td:
 t=pathlib.Path(td);(t/'scripts').mkdir();(t/'assets').mkdir()
 for f in ['scripts/search_index.py','index.html','assets/search-index.js']:shutil.copy2(R/f,t/f)
 for d in ['lessons','reference']:(t/d).symlink_to(R/d,target_is_directory=True)
 argv=[sys.executable,'-B',str(t/'scripts/search_index.py'),'--check']
 asset=t/'assets/search-index.js';page=t/'index.html'
 before={str(p.relative_to(t)):sha(p.read_bytes()) for p in [asset,page,t/'scripts/search_index.py']}
 run('isolated_fresh_default',argv,cwd=t)
 assert before=={str(p.relative_to(t)):sha(p.read_bytes()) for p in [asset,page,t/'scripts/search_index.py']}
 asset.write_text('[]\n');b=asset.read_bytes();run('isolated_stale_default_no_write',argv,cwd=t,expected=1);assert asset.read_bytes()==b
 asset.unlink();run('isolated_missing_default_no_create',argv,cwd=t,expected=1);assert not asset.exists()
 shutil.copy2(R/'assets/search-index.js',asset)
 page.write_text(now.replace('src="assets/search-index.js"','src="assets/missing.js"'));b=page.read_bytes();a=asset.read_bytes()
 run('isolated_bad_wiring_no_write',argv,cwd=t,expected=1);assert page.read_bytes()==b and asset.read_bytes()==a
run('focused_search',[sys.executable,'-B','-m','unittest','discover','-s','tests/search','-v'])
r=subprocess.run(['/home/avist/agent-stack/lead/bin/lead-task','check-skills','rel-14','--slice','rel-14'],text=True,capture_output=True)
print(json.dumps(dict(label='historical_skill_gate',exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)))
assert not git('status','--porcelain=v1')
print('SOURCE_AUDIT_PASS; browser=PENDING; historical missing sharpen not backfilled; final HEAD='+git('rev-parse','HEAD').decode().strip())
