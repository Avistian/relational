import hashlib, importlib.util, json, os, pathlib, re, subprocess, sys, tempfile
root=pathlib.Path('/tmp/lead-wt/night-vrel21-grouped')
out=pathlib.Path(__file__).parent
sha='1a3752fbd2dbf6d25a2d75ac92c326cfd8626d7a'
baseline='503704f30fa28e9c8896302d1423f0fcca9b539c'
red='06e82129ac40322461819cde82725cc0247c495c'
proof=root/'docs/evidence/rel-21-grouped/rel-21-grouped'
def git(*args):
    return subprocess.check_output(['git',*args],cwd=root)
def save(name,obj):
    (out/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def run(name,argv):
    p=subprocess.run(argv,cwd=root,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    (out/(name+'.stdout.txt')).write_text(p.stdout)
    (out/(name+'.stderr.txt')).write_text(p.stderr)
    save(name+'.status.json',dict(argv=argv,exit_code=p.returncode,source_head=sha))
    return p
assert git('rev-parse','HEAD').decode().strip()==sha
assert not git('status','--porcelain')
focused=run('focused',[sys.executable,'-B','-m','unittest','discover','-s','tests/curriculum','-p','test_*.py','-v'])
assert focused.returncode==0 and 'Ran 8 tests' in focused.stderr and '\nOK\n' in focused.stderr
live=run('live',[sys.executable,'-B','tests/curriculum/check_curriculum.py'])
assert live.returncode==0
report=json.loads(live.stdout)
assert report==json.loads((proof/'live.stdout.json').read_text())
# Independent source census: explicitly expand only the three reviewed related groups;
# all remaining manifest pages must have exact standalone identifiers in the first cell.
entries=json.loads((root/'lessons/manifest.json').read_text())['lessons']
lines=(root/'CURRICULUM.md').read_text().splitlines()
groups={'111–114':range(111,115),'151–154':range(151,155),'192–194':range(192,195)}
def norm(s):
    m=re.fullmatch(r'(B?)(0*\d+)([a-z]?)',str(s),re.I)
    assert m,s
    return m[1].upper()+ (str(int(m[2])).zfill(2) if m[1] else str(int(m[2])))+m[3].lower()
rows={}
grouped=[]
for n,line in enumerate(lines,1):
    if not line.startswith('|'): continue
    cells=line.strip('|').split('|')
    label=cells[0].strip().replace('★','').replace('◆','').strip()
    if label in groups:
        ids=[str(i) for i in groups[label]]
        grouped.append(dict(line=n,label=label,topic=cells[1].strip(),ids=ids))
    elif re.fullmatch(r'B?\d+[a-z]?',label,re.I): ids=[norm(label)]
    else: continue
    for identity in ids: rows.setdefault(identity,[]).append((n,cells[1].strip()))
inventory=[]
for e in entries:
    identity=norm(e['id']); matches=rows.get(identity,[])
    assert matches,identity
    assert all(not(re.search(r'\bplanned\b|not yet available',topic,re.I) and e['published']) for n,topic in matches)
    path=root/'lessons'/(e['slug']+'.html'); assert path.is_file(),path
    inventory.append(dict(id=identity,slug=e['slug'],published=e['published'],rows=[n for n,t in matches],html_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(entries)==228 and len({e['id'] for e in inventory})==228
actual={p.stem for p in (root/'lessons').glob('*.html')}
assert actual=={e['slug'] for e in entries}
assert {p['id']:p['rows'] for p in inventory}=={p['id']:p['rows'] for p in report['pages']}
assert '179' not in {e['id'] for e in inventory}
save('independent-inventory.json',dict(manifest_pages=228,covered_pages=228,zero_row_pages=[],planned_disagreements=[],groups=grouped,pages=inventory,missing_html=[],unlisted_html=[]))
# Run exact shipped parser and CLI on independent boundary and contradiction probes.
spec=importlib.util.spec_from_file_location('checker',root/'tests/curriculum/check_curriculum.py')
checker=importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
cases=[('111–114',{'111','112','113','114'}),('111-114',{'111','112','113','114'}),('111—114',{'111','112','113','114'}),('B19',{'B19'}),('B19a/B19b',{'B19a','B19b'}),('B19a, B19b',{'B19a','B19b'}),('066b',{'66b'}),('114–111',set()),('111–B114',set()),('B19a–B19b',set()),('111–114 extra',set()),('[B19a](x)',set())]
results=[]
for label,expected in cases:
    got=checker.row_ids(label); assert got==expected,(label,got,expected)
    results.append(dict(label=label,actual=sorted(got),expected=sorted(expected)))
save('parser-boundaries.json',results)
cli_cases=[
('range-boundaries', ['110','111','114','115','111a','B111'], '| 111–114 | Related | Audit |', {'110','115','111a','B111'},set(),True),
('incidental', ['B19','B19a','B19b'], '## B19a/B19b\nB19a in prose\n| B19 | Parent B19a | [B19b](x) |', {'B19a','B19b'},set(),True),
('group-planned',['111','112'],'| 111–112 | Planned, not yet available | — |',set(),{'111','112'},True),
('one-planned-row-still-contradicts',['111'],'| 111 | Available | Audit |\n| 111 | Planned | — |',set(),{'111'},True),
('unpublished-not-excluded',['180'],'| 179 | Planned | — |',{'180'},set(),False),
('unpublished-planned-consistent',['180'],'| 179 | Planned | — |\n| 180 | Planned | — |',set(),set(),False)]
raw=[]
for name,ids,content,zeros,planned,published in cli_cases:
    with tempfile.TemporaryDirectory(prefix='rel21-verifier-') as d:
        r=pathlib.Path(d); (r/'lessons').mkdir()
        (r/'lessons/manifest.json').write_text(json.dumps({'lessons':[dict(id=i,slug=i.lower()+'-probe',published=published) for i in ids]}))
        (r/'CURRICULUM.md').write_text(content)
        before={str(p.relative_to(r)):p.read_bytes() for p in r.rglob('*') if p.is_file()}
        p=subprocess.run([sys.executable,'-B',str(root/'tests/curriculum/check_curriculum.py'),'--root',d],capture_output=True,text=True)
        data=json.loads(p.stdout)
        assert {x['id'] for x in data['issues'] if x['kind']=='zero_row_coverage'}==zeros,name
        assert {x['id'] for x in data['issues'] if x['kind']=='planned_row_disagreement'}==planned,name
        assert p.returncode==(1 if zeros or planned else 0) and not p.stderr,name
        assert before=={str(p.relative_to(r)):p.read_bytes() for p in r.rglob('*') if p.is_file()}
        raw.append(dict(name=name,ids=ids,curriculum=content,published=published,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
save('independent-cli-probes.json',raw)
# Historical evidence inspection only: never create or rerun RED.
changed=git('diff','--name-only',baseline,sha).decode().splitlines()
assert all(p.startswith(('tests/curriculum/','docs/evidence/rel-21-grouped/rel-21-grouped/')) for p in changed)
immutable={p:git('show',red+':'+p)==(root/p).read_bytes() for p in ['tests/curriculum/test_curriculum.py','tests/curriculum/__init__.py']}
assert all(immutable.values())
assert 'tests/curriculum/check_curriculum.py' not in git('ls-tree','-r','--name-only',red).decode().splitlines()
assert set(git('diff','--name-only',baseline,red).decode().splitlines())==set(immutable)
assert git('diff','--name-only',red,'beda5136').decode().splitlines()==['tests/curriculum/check_curriculum.py']
assert git('rev-parse',red+'^').decode().strip()==baseline
assert git('rev-parse','beda5136^').decode().strip()==red
redlog=(proof/'red.stderr.txt').read_text()
assert redlog.count('AssertionError: False is not true : Grouped curriculum checker not implemented')==8
assert 'FAILED (failures=8)' in redlog and (proof/'red.exit.txt').read_text().strip()=='exit_code=1'
assert 'Ran 8 tests' in (proof/'green.stderr.txt').read_text() and (proof/'green.exit.txt').read_text().strip()=='exit_code=0'
declaration=json.loads((proof/'implementation-declaration.json').read_text())
assert declaration==json.loads(pathlib.Path('/home/avist/agent-stack/lead/.lead/tasks/rel-21-grouped/tdd-declarations/rel-21-grouped.json').read_text())
assert declaration['before_red']==baseline and declaration['files']==['tests/curriculum/check_curriculum.py']
hashes=json.loads(pathlib.Path('/home/avist/agent-stack/lead/.lead/tasks/rel-21-grouped/evidence/builder/SHA256SUMS.json').read_text())
assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
content={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['CURRICULUM.md','lessons/manifest.json']}
assert all(git('show',baseline+':'+p)==(root/p).read_bytes() for p in content)
assert not git('status','--porcelain')
save('scope-and-history.json',dict(sha=sha,baseline=baseline,red=red,green='beda5136f63ea13c78634c03756b556ab4360413',changed=changed,tests_immutable=immutable,course_hashes=content,course_content_unchanged=True,red_evidence='Historical logs: 8 missing-implementation assertion failures, not rerun',green_evidence='Historical logs plus fresh focused 8 PASS',builder_hashes_verified=len(hashes),declaration_matches=True,source_dirty=False))
print('OK focused=8 independent_pages=228 parser_boundaries=12 cli_probes=6 source_clean=true historical_RED_GREEN=confirmed')
