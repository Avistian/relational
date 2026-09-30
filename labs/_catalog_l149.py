"""Reconstruct Figure3 mean bars from source SVG, not historical scalar logs."""
import hashlib,json,re,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
from relkit.weakness_l149 import rank_catalog
P=Path(__file__).resolve().parent;O=P/'evidence/l149';O.mkdir(exist_ok=True,parents=True)
u='https://arxiv.org/html/2407.20060v1/combined_prediction.svg'
b=urllib.request.urlopen(u,timeout=40).read();assert b== (P/'sources/l129/combined_prediction.svg').read_bytes()
(O/'figure3.svg').write_bytes(b)
root=ET.fromstring(b);bars={c:[] for c in ['#72b7a1','#e99675']}
for p in root.iter():
 if p.tag.endswith('path') and p.get('fill') in bars:
  nums=list(map(float,re.findall(r'-?\d+(?:\.\d+)?',p.attrib['d'])))
  if nums[1]<250:bars[p.get('fill')].append(dict(left=nums[0],top=nums[1],right=nums[2],bottom=nums[3]))
assert all(len(v)==15 for v in bars.values())
classification=['rel-stack/user-engagement','rel-stack/user-badge','rel-amazon/user-churn','rel-amazon/item-churn','rel-f1/driver-dnf','rel-f1/driver-top3','rel-hm/user-churn','rel-trial/study-outcome']
regression=['rel-stack/user-votes','rel-amazon/user-ltv','rel-amazon/item-ltv','rel-hm/item-sales','rel-f1/driver-position','rel-trial/study-adverse','rel-trial/site-success']
rows=[]
for i,task in enumerate(classification+regression):
 is_cls=i<8
 # Calibration uses printed tick locations. AUROC returned as fraction.
 def score(x):return (.5+(x-167.45)/(424.63606-167.45)*.4) if is_cls else (.6+(x-614.1642)/(798.48086-614.1642)*.4)
 g=bars['#72b7a1'][i];f=bars['#e99675'][i]
 rows.append(dict(task=task,rdl=score(g['right']),fe=score(f['right']),metric='AUROC' if is_cls else 'MAE',units='fraction' if is_cls else 'MAE divided by RDL MAE',variant='basic' if is_cls else 'GNN+LightGBM',split='test',source='RelBench-v1-Figure3',status='PLOT_DERIVED',display_resolution=.001 if is_cls else .01,geometry=dict(rdl=g,fe=f),original_scalar='NOT_AVAILABLE',uncertainty='Source bar endpoints reconstructed; display precision is not a statistical interval. Error bars not digitized.'))
rankings={metric:rank_catalog([r for r in rows if r['metric']==metric]) for metric in ['AUROC','MAE']}
assert rankings['AUROC'][0]['task']=='rel-f1/driver-top3' and rankings['MAE'][0]['task']=='rel-hm/item-sales'
# Separate value checks against visible calibrated chart labels/positions.
assert abs(rankings['AUROC'][0]['gap']+.0408)<.0002
assert abs(rankings['MAE'][0]['fe']-.652)<.002
report=dict(status='PASS',source_url=u,source_sha256=hashlib.sha256(b).hexdigest(),paper='https://arxiv.org/html/2407.20060v1#S6.F3',scope='15 user-study tasks, not every RelBench task; point-estimate ranking, no significance claim',rows=rows,rankings=rankings,regression_warning='Normalized MAE preserves within-task winner, but scale normalization affects cross-task order; raw MAEs are not recovered. Figure3 boosted RDL differs from basic Table7.',extraction='SVG path right endpoints calibrated to printed axes; task labels manually transcribed and visually checked; no original score or error-bar reconstruction')
(O/'catalog.json').write_text(json.dumps(report,indent=2))
print({m:[(r['task'],round(r['gap'],4)) for r in rs] for m,rs in rankings.items()})
