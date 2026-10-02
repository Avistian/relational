"""Record primary-source claims, historical gaps and exact local dependencies."""
import hashlib,json,urllib.request,importlib.metadata
from pathlib import Path
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent;S=P/'sources/l181';E=P/'evidence/l181'
urls={'LICENSE':'https://raw.githubusercontent.com/stanford-star/relbench/0d47fe0c8a1a51aaf97ab485f4a028e290f97c67/LICENSE','author-blog.html':'https://www.appsofa.com/blog/relgt-ac-graph-transformers-relational-databases'}
for name,url in urls.items():
 (S/name).write_bytes(urllib.request.urlopen(url).read())
links={}
for name in ['relgt-ac.html','author-blog.html']:
 soup=BeautifulSoup((S/name).read_text(),'html.parser')
 links[name]=sorted(set(a.get('href','') for a in soup.find_all('a') if any(s in a.get('href','') for s in ['github.com','huggingface.co'])))
# Preserve distinctions among v2's model recipe and the paper-described RelGT-AC.
audit=dict(date='2026-10-02',relgt_ac=dict(search_queries=['site:github.com "RelGT-AC"','site:appsofa.com "RelGT-AC" code'],inspected_primary_pages=['https://arxiv.org/html/2606.03040v1',urls['author-blog.html']],artifact_links_found=links,authenticated_model_source='NOT_LOCATED_IN_BOUNDED_SEARCH',global_absence_claim=False,seed_only_masking='paper equation1',evaluation_split='validation comparisons; no claim of reproduced heldout test',reported_seeds=[0,1,2],reported_epochs=50,reported_patience=10,reproduction='NOT_RUN_SOURCE_GAPS'),source_protocol=dict(target_removal='globally from the entire entity table before encoding',baseline_val_fit='train',baseline_test_fit='train+val',gnn_selection='minimum validation MAE; last tie wins',gnn_objective='L1Loss',gnn_fanouts=[128,64],gnn_epochs=10,gnn_seeds=list(range(5)),historical_seed_ids='NOT_ESTABLISHED',historical_environment='NOT_ESTABLISHED',graph_materialization='all rows, including after test cutoff; temporal sampler is later',fit_scope_risk='full-population numerical statistics; query-time feature access still NOT_CHECKED',same_day_event_availability='NOT_ESTABLISHED',full_sampler_audit='NOT_RUN_AFTER_TRAINING_HEALTH_STOP'),environment={n:importlib.metadata.version(n) for n in ['torch','pytorch-frame','torch-geometric','numpy','pandas','pyarrow','duckdb','scikit-learn']},links=urls)
(S/'protocol-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
ledger=json.loads((S/'source-ledger.json').read_text());ledger['files']={str(p.relative_to(P.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='source-ledger.json'}
(S/'source-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
print('Source and dependency audit recorded; RelGT-AC model source not authenticated')
