"""Add B05 without rewriting existing manifest entries or prior lesson titles."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b05-tabdpt-real-data-retrieval'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']=='B05' for x in m['lessons']):
 m['lessons'].append(dict(id='B05',slug=S,year=5,quarter=5,sortOrder=200.05,checkpoint=False,labPath='labs/'+S+'.ipynb',title='TabDPT: real-data pretraining and retrieval',published=True));m['version']+=1
p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;text=p.read_text();text=re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda x:x[1]+str(m['version']),text);p.write_text(text)
p=R/'plan/year-5-6-bridge.md';text=p.read_text().replace('B01–B04 and B04a prepared; 24 planned','B01–B05 and B04a prepared; 23 planned')
marker='### B05 ★ · TabDPT: real-data pretraining and retrieval\n'
if '**Prepared 2026-10-03:** [lesson](../lessons/'+S not in text:
 text=text.replace(marker,marker+'\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb). Complete 18-episode/144-prediction course audit and released-sampler counterexample. Banknote two-fold reproduction INCOMPLETE_SOURCE_PROTOCOL; no checkpoint inference; learner PENDING_WRITTEN_DEFENSE.\n\n')
p.write_text(text)
p=R/'CURRICULUM.md';text=p.read_text().replace('B01–B04 and B04a are prepared','B01–B05 and B04a are prepared').replace('The other 24 units remain planned.','B05 adds 18 real-column episodes, a released-sampler counterexample and source-gated two-fold banknote target. The other 23 units remain planned.').replace('[TabDPT: real-data pretraining and retrieval](./plan/year-5-6-bridge.md#b05) | Real-data episode, retrieval and contamination audit.','[TabDPT: real-data pretraining and retrieval](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): 18 episodes / 144 predictions; banknote source gate; defense pending.')
p.write_text(text)
p=R/'reference/curriculum.html';text=p.read_text().replace('B01–B04 and B04a prepared; 24 planned','B01–B05 and B04a prepared; 23 planned').replace('../plan/year-5-6-bridge.md#b05','../lessons/'+S+'.html').replace('Real-data episode, retrieval and contamination audit</td>','18 course episodes; sampler audit; banknote source gate; defense pending</td>');p.write_text(text)
for name,section in [('labs/README.md','\n### B05 · TabDPT: real-data pretraining and retrieval\n\n[Student notebook]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Protocol](b05-reproduction.md). Complete 18-episode / 144-prediction course audit and released-method counterexample. Banknote two-fold reproduction INCOMPLETE_SOURCE_PROTOCOL; learner defense pending.\n'),('RESOURCES.md','\n## B05 · TabDPT episodes, retrieval and contamination (2026-10-03)\n\nPrimary: [TabDPT v3 §3 and Appendix B.1](https://arxiv.org/html/2410.18164v3), [training release](https://github.com/layer6ai-labs/TabDPT-training/tree/af0340c5cdebe2ceb6b94c09d6f9564ee80b89df), [Turbo v1](https://arxiv.org/html/2608.01400v1). B05 pins original and later checkpoint provenance separately. Released use_knn retrieves before target removal; the paper-order course function removes it first. Complete local counterexample is not historical training or benchmark-effect evidence.\n')]:
 p=R/name;text=p.read_text()
 if section.split('\n')[1] not in text:p.write_text(text+section)
p=R/'.github/workflows/pages.yml';text=p.read_text()
if '# B05:' not in text:
 block='''          # B05: target-safe episodes, released sampler and original-paper source gate.
          mkdir -p public/labs/evidence/b05 public/labs/sources/b05 public/labs/figures/b05
          cp labs/b05-reproduction.md labs/_*b05*.py labs/_*b05*.json public/labs/
          cp labs/relkit/retrieval_b05.py public/labs/relkit/
          cp labs/evidence/b05/*.json labs/evidence/b05/reproducer.zip public/labs/evidence/b05/
          cp -r labs/sources/b05/. public/labs/sources/b05/
          cp labs/figures/b05/* public/labs/figures/b05/
          cp labs/solutions/b05-tabdpt-real-data-retrieval.ipynb public/labs/solutions/

'''
 text=text.replace('      - name: Setup Pages',block+'      - name: Setup Pages');p.write_text(text)
p=R/'.gitignore';text=p.read_text()
if '!labs/solutions/'+S+'.ipynb' not in text:p.write_text(text+'\n# B05 portable executed solution.\n!labs/solutions/'+S+'.ipynb\n')
print('Integrated B05; manifest version',m['version'])
