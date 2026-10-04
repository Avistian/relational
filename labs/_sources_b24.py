"""Dated primary-source access receipts; no vendor rank is treated as independent evidence."""
import concurrent.futures,hashlib,json
from pathlib import Path
import requests
ITEMS=[
('LimiX','https://github.com/limix-ldm-ai/LimiX','Official repository inspected: LimiX-2 release, weights and inference route; version-specific terms. Defer joint imputation for this ranking-only proposal; reopen if missing-value modeling is the hypothesis.'),
('TabFM','https://arxiv.org/html/2609.37959v1','Primary report inspected: base, transformed/ensembled TabFM+ and LLM-assisted TabFM-Auto are separate operating points. Defer until a specific configuration and total selection/LLM budget are frozen.'),
('EXAONE','https://arxiv.org/html/2608.25774v1','Primary method inspected: feature and support-conditioned item processing with summary states. Candidate flat comparator; no B24 inference or independent rank claim.'),
('Nori','https://huggingface.co/Synthefy/Nori/blob/main/README.md','Official model card inspected; links implementation and Apache-2.0 terms. Candidate comparator; pin revision, task head and settings before execution.'),
('RT-J','https://star-project.stanford.edu/rt-j/','Author project page inspected. Keep separate from RT-v1. Full historical code/checkpoint/protocol authentication is not established by this page; defer execution until those artifacts are admitted.'),
('Seldon','https://www.neuralk.ai/white-paper/seldon-foundation-made-tabular','Provider report inspected; provider also authors TabBench. Optional external comparator, not independent confirmation. Excluded from this zero-paid-compute replay; reopen with approved access and a matched protocol.'),
('NEXUS','https://fundamental.tech/nexus','Official product page inspected. Product/API description is not a reproduced result. Excluded from this zero-paid-compute replay; reopen with accessible predictions, protocol and approved costs.'),
('SAP-RPT-1-OSS','https://huggingface.co/SAP/sap-rpt-1-oss','Official model card explicitly identifies renamed ConTextTab with unchanged checkpoint and functionality. One alias, not an additional independent model.'),
('RDB-PFN v5','https://arxiv.org/html/2603.03805v5','Primary paper and B23 frozen contract anchor the selected Table9 comparison. B24 fully replays released B23 predictions; whole-paper pretraining and historical availability remain outside established evidence.')]
def get(item):
 name,url,assessment=item
 try:
  r=requests.get(url,timeout=35);return dict(name=name,url=url,checked='2026-10-04',http_status=r.status_code,resolved_url=r.url,response_sha256=hashlib.sha256(r.content).hexdigest(),assessment=assessment,depth='PRIMARY_PAGE_INSPECTED',full_reproduction='NOT_RUN')
 except requests.RequestException as ex:return dict(name=name,url=url,checked='2026-10-04',http_status=None,error=type(ex).__name__,assessment=assessment,depth='BROWSER_TEXT_INSPECTED_HTTP_ARCHIVE_FAILED',full_reproduction='NOT_RUN')
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(get,ITEMS))
 out=dict(scope='Candidate source/access audit, not current leaderboard or model execution',rows=rows)
 p=Path(__file__).resolve().parent;p.joinpath('sources/b24/candidate-audit.json').write_text(json.dumps(out,indent=2)+'\n');print([(x['name'],x['http_status']) for x in rows])
