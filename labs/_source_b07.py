"""Authenticate archived primary evidence; refuse to invent the original benchmark."""
import hashlib,json
from pathlib import Path


def audit_sources(source):
    source=Path(source);inventory=json.loads((source/'inventory.json').read_text())
    good=[r for r in inventory if r['status']=='OK']
    for r in good:
        if hashlib.sha256((source/r['file']).read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Source artifact mismatch: '+r['file'])
    old=(source/'original/contexttab/contexttab.py').read_text();new=(source/'github/sap_rpt_oss/rpt.py').read_text()
    assert 'replace=True' in old and 'replace=False' in new
    tree=json.loads((source/'original-tree.json').read_text());files=[x['path'] for x in tree['tree'] if x['type']=='blob']
    hf=json.loads((source/'hf-model.json').read_text());hf_files=[x['rfilename'] for x in hf['siblings']]
    return dict(name='B07-CONTEXTTAB-BAGGING',status='INCOMPLETE_SOURCE_PROTOCOL',paper='arXiv:2506.10707v1 Table2, binning base versus without bagging, CARTE subset',published_targets=dict(base_accuracy_percent=76.,base_r2_percent=71.4,without_bagging_accuracy_delta_pp=-.4,without_bagging_r2_delta_pp=-.4),original_git_revision=json.loads((source/'original-commits.json').read_text())[0]['sha'],current_git_revision=json.loads((source/'github-head.json').read_text())['sha'],hf_revision=hf['sha'],hf_files=hf_files,archived_files=len(good),unavailable_requests=[dict(file=r['file'],error=r['error']) for r in inventory if r['status']!='OK'],documented_conflicts=['Original June code uses replacement in bagging; current code uses replace=False despite its replacement comment.','Original default checkpoint l2/base.pt is a direct-regression default, not authenticated as Table2 binning base.','Current wrapper defaults to a November2025 checkpoint, later than the June2025 paper.'],missing=['Authenticated Table2 binning-base checkpoint and its original configuration','Exact complete CARTE membership with original train/validation/test row identities','Benchmark evaluator including preprocessing, context/seed/bagging settings and aggregation'],source_tree_scope=files,run_dispatch='REFUSED_BEFORE_COMPUTE',fresh_paper_inference='NOT_RUN',full_pretraining='NOT_RUN',full_benchmark='NOT_RUN',paid_usd=0,release_alias='Current model card documents ConTextTab -> SAP-RPT-1-OSS; alias alone does not authenticate Table2 protocol')


if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=audit_sources(p/'sources/b07');(p/'evidence/b07/source-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['missing'])
