"""The PR workflow must execute the ticket's checks without deployment access."""
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


class PullRequestChecks(unittest.TestCase):
    def test_pull_requests_run_all_five_checks_without_deploying(self):
        path = ROOT / '.github/workflows/pr-checks.yml'
        self.assertTrue(path.is_file(), 'PRs need a validation workflow')
        workflow = yaml.safe_load(path.read_text())
        # PyYAML's YAML 1.1 loader treats the GitHub Actions key `on` as True.
        trigger = workflow.get('on', workflow.get(True))
        self.assertIn('pull_request', trigger)
        self.assertNotIn('pull_request_target', trigger)
        self.assertEqual(workflow['permissions'], {'contents': 'read'})
        jobs = workflow['jobs']
        self.assertTrue(jobs)
        required = {
            'python3 labs/_test_b05_source_secrets.py',
            'python3 scripts/refresh_lesson_visuals.py --check',
            'python3 labs/_check_pages_checkout.py',
            'python3 scripts/search_index.py --check',
            "python -m unittest discover -s tests -p 'test_*.py'",
        }
        commands = set()
        for job in jobs.values():
            self.assertEqual(job['runs-on'], 'ubuntu-latest')
            self.assertNotIn('permissions', job)
            self.assertNotIn('environment', job)
            for step in job['steps']:
                if 'run' in step:
                    commands.update(step['run'].splitlines())
                self.assertNotIn('continue-on-error', step)
                self.assertNotIn('if', step)
                self.assertNotIn('deploy', step.get('uses', '').lower())
        self.assertTrue(required <= commands, f'Missing checks: {required - commands}')
        self.assertNotIn('check_course_visuals.py', path.read_text())
        self.assertNotRegex(path.read_text(), r'\$\{\{\s*secrets[.\[]')

    def test_hosted_runner_reclaims_capacity_before_checkout(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/pr-checks.yml').read_text())
        for job in workflow['jobs'].values():
            steps = job['steps']
            checkout = next(i for i, step in enumerate(steps)
                            if step.get('uses', '').startswith('actions/checkout@'))
            setup = steps[:checkout]
            self.assertTrue(setup, 'The 3.84 GiB repository plus export/build copies need hosted-runner space before checkout')
            cleanup = '\n'.join(step.get('run', '') for step in setup)
            self.assertIn('df -h', cleanup)
            self.assertIn('df -i', cleanup)
            self.assertIn('sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc', cleanup)
            self.assertNotIn('GITHUB_WORKSPACE', cleanup)
