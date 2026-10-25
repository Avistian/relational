"""New grouped-row contract: disposable sources, public JSON CLI, no training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).with_name('check_curriculum.py')


class GroupedRows(unittest.TestCase):
    def run_audit(self, entries, rows, prose=''):
        self.assertTrue(CHECKER.is_file(), 'Grouped curriculum checker not implemented')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lessons').mkdir()
            (root / 'lessons/manifest.json').write_text(json.dumps({'lessons': entries}))
            (root / 'CURRICULUM.md').write_text(
                prose + '\n| # | Topic | Lab |\n|---|---|---|\n' + rows)
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = subprocess.run([sys.executable, '-B', str(CHECKER), '--root', str(root)],
                                    capture_output=True, text=True)
            self.assertEqual(result.stderr, '', result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 0 if report['status'] == 'PASS' else 1)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            return report

    @staticmethod
    def entry(identity, published=True):
        return {'id': identity, 'slug': str(identity).lower() + '-fixture', 'published': published}

    def test_intentional_numeric_range_covers_every_related_page(self):
        report = self.run_audit([self.entry(i) for i in range(111, 115)],
                                '| 111–114 | OGB benchmark task | One reproduction |\n')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['manifest_pages'], 4)
        self.assertEqual(report['covered_pages'], 4)
        self.assertEqual(len({p['rows'][0] for p in report['pages']}), 1)
        self.assertEqual(report['issues'], [])

    def test_explicit_multipart_group_and_separate_suffix_rows_pass(self):
        report = self.run_audit([self.entry(i) for i in ('B19', 'B19a', 'B19b', '66b')],
            '| B19 ★ | Parent | Audit |\n| B19a/B19b ◆ | Related multipart pages | Audit |\n'
            '| 066b | Suffix | Audit |\n')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['covered_pages'], 4)

    def test_parent_and_incidental_mentions_do_not_cover_suffix(self):
        report = self.run_audit([self.entry(i) for i in ('B19', 'B19a', 'B19b')],
            '| B19 | Parent mentions B19a | B19b lab |\n',
            prose='## B19a–B19b\nThese pages are mentioned in prose.')
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual({i['id'] for i in report['issues'] if i['kind'] == 'zero_row_coverage'},
                         {'B19a', 'B19b'})
        self.assertEqual(report['covered_pages'], 1)

    def test_range_does_not_hide_an_uncovered_page(self):
        report = self.run_audit([self.entry(i) for i in (111, 112, 113, 114, 115)],
                                '| 111-114 | Portfolio | Report |\n')
        self.assertEqual(report['covered_pages'], 4)
        self.assertEqual([(i['kind'], i['id']) for i in report['issues']],
                         [('zero_row_coverage', '115')])

    def test_planned_published_page_is_a_disagreement_even_in_group(self):
        report = self.run_audit([self.entry(111), self.entry(112)],
                                '| 111–112 | Portfolio — planned, not yet available | — |\n')
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['covered_pages'], 2)
        self.assertEqual({(i['kind'], i['id']) for i in report['issues']},
            {('planned_row_disagreement', '111'), ('planned_row_disagreement', '112')})

    def test_planned_absent_and_unpublished_pages_are_consistent(self):
        report = self.run_audit([self.entry(178), self.entry(180, False)],
            '| 178 | Available | Audit |\n| 179 | Failure modes — planned, not yet available | — |\n'
            '| 180 | Planned | — |\n')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['covered_pages'], 2)

    def test_unpublished_manifest_page_still_needs_a_row(self):
        report = self.run_audit([self.entry(180, False)], '| 179 | Planned | — |\n')
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['issues'][0]['kind'], 'zero_row_coverage')

    def test_empty_manifest_cannot_claim_pass(self):
        report = self.run_audit([], '| 111 | Portfolio | — |\n')
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['issues'][0]['kind'], 'invalid_manifest')


if __name__ == '__main__':
    unittest.main()
