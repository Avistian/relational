"""Verify archived release evidence. Original Table12 execution remains gated."""
import hashlib,json
from pathlib import Path

def source_gate(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,item in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==item['sha256'],'Source hash: '+name
    inv={name:json.loads((root/(name+'-inventory.json')).read_text()) for name in ['mitra-classifier','mitra-regressor','mitra-finetune']}
    config=json.loads((root/'mitra-classifier/config.json').read_text())
    assert (config['dim'],config['n_layers'],config['n_heads'])==(512,12,4)
    assert 'second-generation' in (root/'mitra-finetune-README.md').read_text()
    return dict(name='B06-MITRA-TABLE12',status='INCOMPLETE_SOURCE_PROTOCOL',paper='2510.21204v1',target='Table12; TabRepo10fold classification; all six mixture settings',p=[0,.4,.5,.6,.7,1],source_files=len(manifest['files']),revisions={name:v['sha'] for name,v in inv.items()},observed_weight_files={name:[x['rfilename'] for x in v['siblings'] if x['rfilename'].endswith('.safetensors')] for name,v in inv.items()},missing=['Authenticated original six ablation checkpoint identities or complete original generator/pretraining code and RNG/configuration histories.','Original TabRepo10fold task manifest, exact split/row identities, preprocessing and inference settings mapped to Table12.','Original per-task/fold predictions and reference metrics plus full ranking comparator pool and evaluator version.'],later_release='mitra-finetune currently targets Mitra-v2; not authenticated original Table12 pretraining',completed_paper_runs=0,full_pretraining='NOT_RUN',full_benchmark='NOT_RUN',historical_identity='NOT_ESTABLISHED',readiness='Runnable evidence preflight only; exact paper trainer/evaluator cannot be reconstructed from authenticated artifacts yet',budget='Paper reports60hours on8A100s for one main run; not a quote for each ablation. No paid dispatch authorized by passing a source hash alone.')
if __name__=='__main__':
    p=Path(__file__).parent;r=source_gate(p/'sources/b06');(p/'evidence/b06/source-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
