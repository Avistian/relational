"""Diagnostic comparison only; not the frozen canonical regression.

Run from repository root with .venv/bin/python -B and this file path.
Loads the authentic pre-repair unittest bytes from the builder base commit.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from tests.browser_checks import test_output_paths as repaired
from tests.browser_checks.test_history_free_retention import HistoryFreeRetentionTests

BASE = '15fa7dc501cab9eaf7ac551e6d4fe5bd1b60792b'
source = subprocess.check_output([
    'git', 'show', f'{BASE}:tests/browser_checks/test_output_paths.py'])
legacy = types.ModuleType('legacy_output_paths')
legacy.__file__ = str(ROOT / 'tests/browser_checks/test_output_paths.py')
exec(compile(source, legacy.__file__, 'exec'), legacy.__dict__)
print(json.dumps({'legacy_commit': BASE, 'legacy_module_sha256':
                  hashlib.sha256(source).hexdigest()}), flush=True)


def run(label, module, root):
    print('\nCASE ' + label, flush=True)
    print('historical object:', subprocess.run(
        ['git', 'cat-file', '-e', '503704f^{commit}'], cwd=root,
        capture_output=True).returncode, flush=True)
    print('commit count:', subprocess.check_output(
        ['git', 'rev-list', '--count', 'HEAD'], cwd=root, text=True).strip(),
        flush=True)
    print('remotes:', subprocess.check_output(
        ['git', 'remote'], cwd=root, text=True).splitlines(), flush=True)
    with patch.object(module, 'ROOT', root):
        case = module.OutputPathsTests(
            'test_generated_directories_ignored_and_tracked_pngs_retained')
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(case)
    print(json.dumps({'case': label, 'run': result.testsRun,
                      'failures': len(result.failures),
                      'errors': len(result.errors),
                      'skipped': len(result.skipped)}), flush=True)
    return len(result.failures), len(result.errors)


# No fixture setup/import exception is treated as a successful diagnostic.
assert run('legacy/full-history/intact', legacy, ROOT) == (0, 0)
fixture = HistoryFreeRetentionTests()
try:
    fixture.setUp()
    assert run('legacy/single-commit/intact', legacy, fixture.root) == (0, 1)
    assert run('repaired/single-commit/intact', repaired, fixture.root) == (0, 0)
    from tests.browser_checks.retained_reviews import ORIGINAL_REVIEW_PATHS
    victim = next(p for p in ORIGINAL_REVIEW_PATHS if p.endswith('.png'))
    print('deleted fixture PNG:', victim, flush=True)
    (fixture.root / victim).unlink()
    assert run('legacy/single-commit/worktree-deletion', legacy, fixture.root) == (0, 1)
    assert run('repaired/single-commit/worktree-deletion', repaired, fixture.root) == (1, 0)
    fixture.git('rm', '--cached', '--', victim)
    assert run('legacy/single-commit/index-deletion', legacy, fixture.root) == (0, 1)
    assert run('repaired/single-commit/index-deletion', repaired, fixture.root) == (1, 0)
finally:
    fixture.doCleanups()
print('DIAGNOSTIC COMPARISON PASS', flush=True)
