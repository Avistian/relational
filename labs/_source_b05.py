"""Authenticate source evidence and expose unresolved original-paper protocol identities."""
import csv,hashlib,json
from pathlib import Path

def source_gate(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'Source hash: '+name
    original=json.loads((root/'original-checkpoint-config.json').read_text())
    later=json.loads(json.loads((root/'checkpoint-header.json').read_text())['__metadata__']['cfg'])
    train=list(csv.DictReader((root/'training/data_splits/noleak_training_datasets.csv').open()))
    evals=list(csv.DictReader((root/'paper-era/tabdpt_datasets/data_splits/cls_datasets.csv').open()))
    bank=[r for r in evals if r['did']=='1462'];assert len(bank)==1 and bank[0]['tid']=='10093.0' and bank[0]['test']=='True'
    train_ids={int(float(r['did'])) for r in train};eval_ids={int(float(r['did'])) for r in evals if r['test']=='True'}
    return dict(name='B05-TABDPT-BANKNOTE-TWO-FOLD',status='INCOMPLETE_SOURCE_PROTOCOL',source_files=len(manifest['files']),original_source='5214f9267b49d3907b074be9d90a660054484416',paper_era_evaluator='eb5c0d9ff303d8fd8a053d9ac8be21af0d4e9157',training_source='af0340c5cdebe2ceb6b94c09d6f9564ee80b89df',original_weight_sha256=json.loads((root/'original-weight-download.json').read_text())['sha256'],original_heads=original['model']['nhead'],later_heads=later['model']['nhead'],dataset_id=1462,task_id=10093,folds=[0,1],context_size=2048,n_ensembles=8,declared_training_rows=len(train),training_unique_ids=len(train_ids),classification_test_ids=len(eval_ids),exact_id_intersection=sorted(train_ids&eval_ids),banknote_in_training_ids=1462 in train_ids,independence='NOT_ESTABLISHED: ID audit only; renamed/derived datasets require content/statistical review',missing=['Authenticated mapping from paper v3 result to original v1.0 versus v1.1 checkpoint and evaluator (4 vs8 heads; later evaluator loads1.1).','Exact bytes and row identities of both TabZilla banknote folds used for the paper; OpenML task ID alone is insufficient.','Original selected banknote per-fold reference scores/predictions and ensemble RNG mapping.'],source_discrepancy='Released pretraining sampler retrieves before target removal; paper Algorithm1 removes target first.',completed_benchmark_runs=0,full_pretraining='NOT_RUN',full_benchmark='NOT_RUN',historical_identity='NOT_ESTABLISHED')
if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=source_gate(p/'sources/b05');(p/'evidence/b05/source-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
