"""The real read-only export import closure must be available on the PR runner."""
import ast
from pathlib import Path
import re
import shlex
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReadOnlyExportDependencies(unittest.TestCase):
    def test_runner_installs_readonly_export_import_requirements_before_discovery(self):
        # Inspect the actual exporter, including its local imports. Do not import
        # CairoSVG here: this config contract must also run with stdlib alone.
        pending = [ROOT / 'scripts/export_architecture_routes.py']
        visited = set()
        external = set()
        while pending:
            source = pending.pop()
            if source in visited:
                continue
            visited.add(source)
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0, f'Audit relative import in {source}')
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    top = module.split('.')[0]
                    if top in sys.stdlib_module_names:
                        continue
                    candidates = [directory / (top + '.py') for directory in
                                  (source.parent, ROOT / 'scripts', ROOT / 'labs')]
                    local = next((path for path in candidates if path.is_file()), None)
                    if local is None:
                        external.add(top)
                    else:
                        pending.append(local)

        distributions = {'cairosvg': 'cairosvg'}
        self.assertFalse(external - distributions.keys(),
                         f'Audit new exporter imports: {external - distributions.keys()}')
        required = {distributions[module] for module in external}
        self.assertTrue(required, 'Expected the real exporter third-party import closure')

        workflow = (ROOT / '.github/workflows/pr-checks.yml').read_text()
        # This workflow uses single-line run commands. Fail closed if discovery
        # moves into a block instead of silently accepting an empty command list.
        commands = re.findall(r'^\s+run: (.+)$', workflow, re.MULTILINE)
        discovery = "python -m unittest discover -s tests -p 'test_*.py'"
        self.assertEqual(commands.count(discovery), 1, 'Expected unfiltered unit discovery')
        installed = set()
        for command in commands[:commands.index(discovery)]:
            argv = shlex.split(command)
            if argv[:4] == ['python', '-m', 'pip', 'install']:
                installed.update(re.split(r'[<>=!~;\[]', arg, maxsplit=1)[0].lower()
                                 for arg in argv[4:] if not arg.startswith('-'))
        self.assertFalse(required - installed,
                         f'Read-only exporter imports {sorted(external)}; install '
                         f'{sorted(required - installed)} before unit discovery')


if __name__ == '__main__':
    unittest.main()
