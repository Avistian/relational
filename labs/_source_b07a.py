"""Authenticate available evidence without filling absent historical protocol fields."""
import json,hashlib
from pathlib import Path

def audit_sources(root):
    root=Path(root);pin=json.loads((root/'source-pin.json').read_text())
    for name,sha in pin['files'].items():
        assert hashlib.sha256((root/'hyperfast'/name).read_bytes()).hexdigest()==sha,'Source artifact mismatch: '+name
    manifest=json.loads((root/'historical-manifest.json').read_text())
    for name,sha in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,'Historical artifact mismatch: '+name
    tree=json.loads((root/'original-tree.json').read_text())
    return dict(status='INCOMPLETE_SOURCE_PROTOCOL',target='2402.14335v1 Table7 banknote-authentication HyperFast',published_balanced_accuracy_percent=100.0,published_sd_percent=0.0,repetitions=10,time_budget_seconds_per_repetition=300,mini_test_train_cap=1000,full_dataset_train_test=[1097,275],current_commit=pin['commit'],publication_era_commit=tree['commit'],checkpoint='Same file43484094 referenced by both pinned configs; current downloaded bytes pinned separately. Historical byte identity not established by URL alone.',missing=['Original outer split row identities and RNG state','Ten mini-test subsample identities/seeds','Exact selected HyperFast inference configurations and five-minute search trajectory','Original evaluator and per-repeat predictions'],available=['Full released architecture and optional downstream optimizer','Current and publication-era inference wrappers','Checkpoint URL and current authenticated tensors','Paper target, repetition count, runtime envelope and search space'],unrun=['Original Table7 ten-repetition reproduction','Full meta-training and all-paper benchmark','MotherNet and iLTM executions'],dispatch='REFUSED until missing original inputs are authenticated; course CLI is a separate experiment')

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=audit_sources(p/'sources/b07a');(p/'evidence/b07a/source-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
