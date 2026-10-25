from pathlib import Path
import json,hashlib,struct,subprocess
R=Path('/tmp/lead-wt/v-rel14-captured-20261025')
C=Path('/home/avist/agent-stack/lead/.lead/tasks/rel-14/evidence/external-capture')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=json.loads((C/'capture-result.json').read_text())
assert x['source_sha']=='67bf8c68d34e7cb33684570633915935cce5cb4e'
assert x['actor']=='lead:rel-14-capture' and x['lead_tip']=='a3dbf76'
assert x['exit_code']==0 and x['failures']==0 and x['source_status_after']==''
code="const fs=require('fs'),vm=require('vm'),c={};vm.createContext(c);vm.runInContext(fs.readFileSync('assets/search-index.js','utf8'),c);console.log(JSON.stringify(require('./assets/search.js').search(c.LessonSearchIndex,'baseline').slice(0,30).map(e=>({text:e.title,href:e.url}))));"
r=subprocess.run(['node','-e',code],cwd=R,capture_output=True,text=True,check=True)
expected=json.loads(r.stdout)
assert len(expected)==19
assert {(c['scheme'],c['label']) for c in x['captures']}=={(s,l) for s in ['file','http'] for l in ['desktop','mobile']}
def check_image(p,digest,w,h):
 b=p.read_bytes();assert b[:8]==b'\x89PNG\r\n\x1a\n'
 dims=struct.unpack('>II',b[16:24]);assert dims==(w,h),(p,dims)
 assert sha(p)==digest,p
 print(json.dumps(dict(file=str(p),sha256=digest,width=w,height=h,bytes=len(b))))
for c in x['captures']:
 assert c['input_value']=='baseline' and c['result_count']==19
 assert c['results']==expected
 assert c['pageerrors']==[] and c['failed_requests']==[]
 assert c['url'].startswith('file:///' if c['scheme']=='file' else 'http://127.0.0.1:')
 w=1280 if c['label']=='desktop' else 390
 assert c['viewport']==dict(width=w,height=900)
 check_image(C/Path(c['path']).name,c['sha256'],w,900)
 for e in c['results']:assert (R/e['href']).is_file() and e['text']
 print('DOM result text/href/order equals real search module: '+c['scheme']+' '+c['label'])
for scheme in ['file','http']:
 d=C/(scheme+'-lead-evidence');m=json.loads((d/'manifest.json').read_text())
 assert m['source_sha']==x['source_sha'] and m['reviewer']==x['actor']
 assert m['capture_url']==next(c['url'] for c in x['captures'] if c['scheme']==scheme)
 log=(d/'capture.log').read_text();assert 'exit=0' in log and 'lead-evidence' in log
 for s in m['screenshots']:
  check_image(d/s['path'],s['sha256'],s['viewport']['width'],900)
 print('raw lead-evidence manifest/log validated '+scheme+'; these pre-interaction images are distinct from search PNGs')
assert all(' 200 ' in line for line in x['http_logs'])
print('CAPTURE_AUDIT_PASS: four search PNGs and four raw lead-evidence images hashes/dimensions; 4x19 DOM records; all result targets exist; recorded pageerrors/failed_requests empty; HTTP log requests 200')
print('Visual review by verifier:rel-14-captured used view_image on ALL FOUR search PNGs. Visible baseline and first result links; readable mobile wrapping. Screenshots show viewport subsets, not all 19 links. HTTP Continue banner is absent in file views; unchanged assets/home.js fetches manifest and catches failure. Capture arrays report interaction observation only, not global console/network absence or a navigation click test. Producer lead:rel-14-capture, not reviewer.')
