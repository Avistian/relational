"""Exercise a real notebook-style run across harmless execution-file changes."""
import hashlib
import json
from pathlib import Path
import tempfile
import torch
from _paper_repro_l049 import run

HERE = Path(__file__).resolve().parent


def live(source, filename):
    namespace = {'__name__': '__main__'}
    exec(compile(source, filename, 'exec'), namespace)
    return {arg: namespace[name] for arg, name in [
        ('model_cls', 'ExcelFormer'), ('train_fn', 'train_excel'),
        ('predict_fn', 'predict_excel'), ('attention_fn', 'spa_attention'), ('mix_fn', 'feat_mix')]}


def snapshot(folder):
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in folder.rglob('*') if p.is_file() and p.name != 'summary.json'}


def check():
    torch.set_num_threads(1)
    source = (HERE / 'relkit/claim_models.py').read_text()
    out = Path(tempfile.mkdtemp(prefix='quality-l049-resume-'))
    first = run('smoke', str(out), device='cpu', **live(source, '<notebook-cell-first-run>'))
    before = snapshot(out)
    second = run('smoke', str(out), device='cpu', **live(source, '<notebook-cell-new-kernel>'))
    assert first['contract_sha256'] == second['contract_sha256'], 'Execution filename changed semantic identity'
    assert snapshot(out) == before, 'Unchanged resume rewrote completed seed artifacts'
    changed = source.replace('scores = q @', 'scores = 0.5 * q @')
    assert changed != source, 'Attention intervention fixture did not change source'
    try:
        run('smoke', str(out), device='cpu', **live(changed, '<notebook-cell-new-kernel>'))
    except AssertionError as error:
        assert 'Changed implementation/protocol' in str(error), error
    else:
        raise AssertionError('Changed attention was allowed to reuse old results')
    assert snapshot(out) == before, 'Rejected run changed completed artifacts'
    report = dict(status='PASS', output=str(out), unchanged_new_kernel_resume='PASS',
                  preserved_files=len(before), changed_attention_rejected='PASS',
                  scope='Real ExcelFormer smoke; dynamic notebook-style definitions with changed execution filename; saved bytes and mtimes preserved; no paper reproduction claim')
    (HERE.parent / 'reviews/lesson-quality-audit-047-070/049-resume.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    check()
