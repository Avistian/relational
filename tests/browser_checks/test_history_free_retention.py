"""Run the public retention unittest against a single-commit Git fixture."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import test_output_paths
from .retained_reviews import ORIGINAL_REVIEW_PATHS


class HistoryFreeRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rel15-retention-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / '.gitignore').write_bytes(
            (test_output_paths.ROOT / '.gitignore').read_bytes())
        # Only path existence/tracking matters to this public test. These are
        # synthetic fixture bytes, never copies or edits of published PNGs.
        for name in ORIGINAL_REVIEW_PATHS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'retention fixture\n')
        self.git('init', '--quiet')
        self.git('add', '.gitignore')
        self.git('add', '--force', 'reviews')
        self.git('commit', '--no-gpg-sign', '--quiet', '-m', 'fixture')
        self.assertEqual(self.git('rev-list', '--count', 'HEAD').stdout.strip(), '1')
        self.assertEqual(self.git('remote').stdout, '')
        missing = self.git('cat-file', '-e', '503704f^{commit}', check=False)
        self.assertNotEqual(missing.returncode, 0)

    def git(self, *args, check=True):
        return subprocess.run(['git', *args], cwd=self.root,
                              text=True, capture_output=True, check=check,
                              env=dict(os.environ,
                                       GIT_AUTHOR_NAME='Retention Fixture',
                                       GIT_AUTHOR_EMAIL='retention@example.invalid',
                                       GIT_COMMITTER_NAME='Retention Fixture',
                                       GIT_COMMITTER_EMAIL='retention@example.invalid'))

    def retention_result(self):
        case = test_output_paths.OutputPathsTests(
            'test_generated_directories_ignored_and_tracked_pngs_retained')
        result = unittest.TestResult()
        with patch.object(test_output_paths, 'ROOT', self.root), \
                contextlib.redirect_stderr(io.StringIO()):
            case.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.skipped, [])
        return result

    def test_public_retention_check_passes_without_historical_objects(self):
        result = self.retention_result()
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.failures, [], result.failures)

    def test_public_retention_check_rejects_deleted_original_png(self):
        victim = next(p for p in ORIGINAL_REVIEW_PATHS if p.endswith('.png'))
        # A missing worktree file must fail even while still indexed.
        (self.root / victim).unlink()
        result = self.retention_result()
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertIn('AssertionError', result.failures[0][1])
        # Removing it from the index must also fail the original inventory check.
        self.git('rm', '--cached', '--', victim)
        result = self.retention_result()
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertIn('AssertionError', result.failures[0][1])
