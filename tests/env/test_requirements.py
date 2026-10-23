"""Exercise the coverage check through its command-line interface."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tests/env/check_requirements.py"


class RequirementCoverageTests(unittest.TestCase):
    def check(self, requirements, lock):
        return subprocess.run(
            [sys.executable, "-B", str(CHECKER), str(requirements), str(lock)],
            capture_output=True, text=True, check=False,
        )

    def test_repository_baseline_is_explicit(self):
        result = self.check(ROOT / "requirements-labs.txt", ROOT / "requirements-labs.lock.txt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("requirements=34 locked=121 covered=17 baseline_gaps=17", result.stdout)
        self.assertIn("refresh lock: Piotr", result.stdout)
        self.assertIn("KNOWN GAP: torch-geometric", result.stdout)
        self.assertIn("KNOWN GAP: rtdl-num-embeddings", result.stdout)

    def test_normalized_names_extras_comments_and_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / "requirements.txt"
            lock = Path(directory) / "lock.txt"
            requirements.write_text(
                "# ignored comment\nDemo_Package[extra]>=99  # inline comment\n"
                "other.package<1; python_version >= '3.0'\nDEMO-package>=100\n"
            )
            lock.write_text("demo-package==0.1\nOTHER_package==999\n")
            result = self.check(requirements, lock)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("requirements=2 locked=2 covered=2 baseline_gaps=0", result.stdout)

    def test_disposable_new_unlocked_dependency_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / "requirements.txt"
            requirements.write_text(
                (ROOT / "requirements-labs.txt").read_text() + "\nw1-new-unlocked-package>=1\n"
            )
            result = self.check(requirements, ROOT / "requirements-labs.lock.txt")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("UNEXPECTED GAP: w1-new-unlocked-package", result.stdout)
            self.assertIn("FAIL: 1 new unlocked package(s)", result.stdout)

    def test_existing_covered_dependency_cannot_become_a_silent_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.txt"
            lock.write_text("\n".join(
                line for line in (ROOT / "requirements-labs.lock.txt").read_text().splitlines()
                if not line.startswith("numpy==")
            ))
            result = self.check(ROOT / "requirements-labs.txt", lock)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("UNEXPECTED GAP: numpy", result.stdout)


if __name__ == "__main__":
    unittest.main()
