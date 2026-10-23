"""Read-only schema validation and exact notebook/HTML/teacher-copy inventory.

This checks file pairing and notebook schema, not rendered cell equivalence,
notebook execution, learner correctness, or paper reproduction.
"""
import json
from pathlib import Path
import warnings

import nbformat


# Exact historical omissions, not a blanket allowance for ignored solutions.
# labs/README.md "No prefilled answers" says teacher copies are gitignored;
# .gitignore:6 ignores labs/solutions/*, later explicit negations ship prepared
# solutions. README's FOUNDATION-058-070 block also states local solutions follow
# that convention. A new absent solution, even with an early number, must fail.
LOCAL_TEACHER_COPIES = (
    '0006-missingness.ipynb',
    '0007-class-imbalance.ipynb',
    '0008-metrics-calibration.ipynb',
    '0009-feature-engineering.ipynb',
    '0010-baseline-checkpoint.ipynb',
    '0011-decision-trees-partitions.ipynb',
    '0012-bagging-random-forest.ipynb',
    '0013-boosting-intuition.ipynb',
    '0014-xgboost.ipynb',
    '0015-lightgbm.ipynb',
    '0016-catboost.ipynb',
    '0017-hyperparameter-search.ipynb',
    '0018-ensembling-stacking.ipynb',
    '0019-when-trees-win.ipynb',
    '0020-q2-checkpoint.ipynb',
    '0021-data-splits-in-the-wild.ipynb',
    '0022-label-leakage-patterns.ipynb',
    '0023-statistical-comparison.ipynb',
    '0024-grinsztajn-benchmark.ipynb',
    '0025-inductive-bias-smoothness.ipynb',
    '0026-inductive-bias-rotation.ipynb',
    '0027-inductive-bias-uninformative-features.ipynb',
    '0028-mlp-resnet-baselines.ipynb',
    '0029-manual-fe-vs-automl.ipynb',
    '0030-q3-checkpoint.ipynb',
    '0031-embeddings-for-categoricals.ipynb',
    '0032-tabtransformer-preview.ipynb',
    '0033-when-to-stop-feature-engineering.ipynb',
    '0034-relational-data-without-rdl.ipynb',
    '0035-what-joins-destroy.ipynb',
    '0036-revisit-your-homework-pipeline.ipynb',
    '0037-document-a-baseline-package.ipynb',
    '0040-year-1-exit-exam.ipynb',
    '0042-mlp-resnet-baselines.ipynb',
    '0043-tabnet.ipynb',
    '0044-node.ipynb',
    '0045-tabtransformer.ipynb',
    '0046-ft-transformer.ipynb',
    '0047-saint.ipynb',
    '0048-dcnv2.ipynb',
    '0049-excelformer-trompt.ipynb',
    '0050-q1-checkpoint.ipynb',
    '0051-why-trees-still-win.ipynb',
    '0052-tabr-retrieval.ipynb',
    '0053-realmlp-strong-defaults.ipynb',
    '0054-tabm-parameter-efficient-ensembling.ipynb',
    '0055-tabred-temporal-splits.ipynb',
    '0056-tabarena-benchmark-literacy.ipynb',
    '0057-cross-family-ensembling.ipynb',
    '0058-surveys-meta-benchmarks.ipynb',
    '0059-validation-set-overfitting.ipynb',
    '0060-broad-model-comparison.ipynb',
    '0061-prior-data-fitted-networks.ipynb',
    '0062-tabpfn-v1.ipynb',
    '0063-synthetic-scm-prior.ipynb',
    '0064-tabpfn-v2.ipynb',
    '0065-tabpfn-query-embeddings.ipynb',
    '0066-tabicl-column-row-attention.ipynb',
    '0067-local-pfn-retrieval-finetuning.ipynb',
    '0068-pfns-under-temporal-shift.ipynb',
    '0069-tabpfn-open-environment-failures.ipynb',
    '0070-foundation-model-checkpoint.ipynb',
    '0071-vime-masked-tabular-ssl.ipynb',
    '0072-scarf-subtab-contrastive-views.ipynb',
    '0073-when-ssl-helps.ipynb',
    '0074-carte-cross-table-transfer.ipynb',
    '0075-pytorch-frame-row-encoder.ipynb',
    '0076-encoder-predictor-stack.ipynb',
    '0077-single-table-ceiling.ipynb',
    '0078-message-passing-preview.ipynb',
    '0079-neural-tabular-decision-guide.ipynb',
    '0080-year-2-exit-exam.ipynb',
)


def audit(root):
    """Return JSON-serializable counts, all validation results and pair gaps."""
    root = Path(root)
    students = {p.stem: p for p in (root / 'labs').glob('*.ipynb')}
    html = {p.stem: p for p in (root / 'labs/html').glob('*.html')}
    solutions = {p.stem: p for p in (root / 'labs/solutions').glob('*.ipynb')}
    relative = lambda p: p.relative_to(root).as_posix()
    groups = {
        'student_without_html': [relative(students[k]) for k in sorted(students.keys() - html.keys())],
        'html_without_student': [relative(html[k]) for k in sorted(html.keys() - students.keys())],
        'student_without_solution': [relative(students[k]) for k in sorted(students.keys() - solutions.keys())],
        'solution_without_student': [relative(solutions[k]) for k in sorted(solutions.keys() - students.keys())],
    }
    template = 'labs/README.md: fill-in convention identifies LAB-TEMPLATE.ipynb as authoring template, not a lesson.'
    allowed = {
        'student_without_html': {'labs/LAB-TEMPLATE.ipynb': template},
        'student_without_solution': {
            **{'labs/' + name: 'labs/README.md: No prefilled answers; .gitignore:6 keeps teacher copies local (explicit historical path).'
               for name in LOCAL_TEACHER_COPIES},
            'labs/LAB-TEMPLATE.ipynb': template,
        },
        'html_without_student': {
            'labs/html/foundation-sequence.html': 'notebooks.html#foundation-materials links this lessons 058-070 materials directory, not a notebook render.',
            'labs/html/0113-scaling-ogb-solution.html': 'labs/_execute_l113.py:7,13 explicitly exports the solution notebook to this suffixed HTML.',
        },
        'solution_without_student': {},
    }
    notebooks = sorted(root.rglob('*.ipynb'))
    report = {
        'counts': {'notebooks': len(notebooks), 'students': len(students), 'html': len(html), 'solutions': len(solutions)},
        'validated': [], 'errors': [], 'warnings': [], 'pairs': groups,
        'exceptions': {group: {path: allowed[group][path] for path in paths if path in allowed[group]}
                       for group, paths in groups.items()},
        'nbformat_version': nbformat.__version__,
    }
    for path in notebooks:
        name = relative(path)
        report['validated'].append(name)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            try:
                # Parse without version conversion; validate every actual notebook,
                # including archived source/evidence notebooks outside student pairs.
                document = json.loads(path.read_text(encoding='utf-8'))
                nbformat.validate(document)
            except (OSError, ValueError, nbformat.ValidationError) as error:
                report['errors'].append(f'{name}: {type(error).__name__}: {error}')
            for warning in emitted:
                report['warnings'].append({'path': name, 'category': warning.category.__name__, 'message': str(warning.message)})
    for group, paths in groups.items():
        for path in paths:
            if path not in allowed[group]:
                report['errors'].append(f'{group}: {path}')
    return report


if __name__ == '__main__':
    import sys
    report = audit(Path(__file__).resolve().parents[2])
    print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(bool(report['errors']))
