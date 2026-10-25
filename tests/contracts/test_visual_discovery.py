"""Exercise public discovery without recursively discovering this regression."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from collections import Counter


ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    'check_course_visuals',
    'check_lesson_visuals',
    'check_visual_details',
    'check_lesson_openings',
)


def case_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from case_ids(item)
        else:
            yield item.id()


class VisualDiscovery(unittest.TestCase):
    def test_discovery_runs_every_original_visual_contract(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            expected = [identity for name in MODULES for identity in case_ids(
                unittest.TestLoader().loadTestsFromModule(importlib.import_module(name)))]
        finally:
            sys.path.pop(0)
        self.assertGreaterEqual(len(expected), 13)
        # A fresh process ensures imports here cannot supply the wrapper's setup.
        probe = '''
import json
import unittest

def ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from ids(item)
        else:
            yield item.id()

suite = unittest.TestLoader().discover('tests/contracts', pattern='test_visual_contracts.py')
print(json.dumps(list(ids(suite))), flush=True)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful() or bool(result.skipped))
'''
        result = subprocess.run([sys.executable, '-B', '-c', probe], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(Counter(json.loads(result.stdout)), Counter(expected), result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        for identity in expected:
            self.assertIn(identity, result.stderr)


if __name__ == '__main__':
    unittest.main()
