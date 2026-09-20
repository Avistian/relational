"""Verify source bytes, literal PDF table transcription and frozen prior evidence."""
import json,re,sys
from pathlib import Path
from pypdf import PdfReader
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hin_l094 import file_sha256
P=Path(__file__).resolve().parent;M=json.loads((P/'_sources_l094.json').read_text());S=P/'sources/hin-l094'
assert file_sha256(S/'survey.pdf')==M['paper']['sha256'];assert file_sha256(S/'table1.json')==M['paper']['table_sha256']
text=PdfReader(S/'survey.pdf').pages[4].extract_text();table=json.loads((S/'table1.json').read_text())
for name,row in table.items():
 line=next(x for x in text.splitlines() if x.startswith(name+' '));values=[int(v.replace(',','')) for v in line.split()[1:]];assert values==list(row.values()),name
prior=json.loads((S/'prior-evidence.json').read_text())
for item in prior.values():
 path=P/item['filename'];assert file_sha256(path)==item['sha256'];assert json.loads(path.read_text())==item['record']
assert len(prior['91']['record']['runs'])==10
assert prior['92']['record']['training_runs']==1 and prior['92']['record']['knn_per_training_run']==40
assert prior['93']['record']['completed_CS_fits']==0
r=json.loads((P/'_paper_l094_results.json').read_text());s=r['statistics']
assert s['all_stored_edges']==2*s['forward_edges'] and s['reverse_content_matches']
assert s['listed_forward_edges']+s['unlisted_forward_edges']==s['forward_edges']
assert r['implementation_sha256']==file_sha256(P/'relkit/hin_l094.py')
assert r['loader_sha256']==file_sha256(P/'relkit/oag_read_l094.py')
result={'status':'PASS','pdf_table_rows_verified':3,'prior_source_files_verified':3,'graph_hash':r['data_sha256'],'source_hash':r['implementation_sha256'],'evidence_boundary':'NN data audited; CS/OAG arithmetic only; no new training'}
(P/'_audit_l094_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
