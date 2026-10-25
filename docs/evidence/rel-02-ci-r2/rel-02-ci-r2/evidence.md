# Human evidence

Source: 2e9d0aa91b792dbacdf90bde3ad824cd7c9d3dfe
Reviewer: NOT_RUN: separate counted verifier pending

BEFORE failures: 1
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 1, "failed": 1, "errors": 0, "skipped": 0}
AssertionError: {'cairosvg'} is not false : Read-only exporter imports ['cairosvg']; install ['cairosvg'] before unit discovery
Ran 1 test in 0.027s
FAILED (failures=1)
```

## GREEN
```text
LEAD_RESULT={"run": 1, "failed": 0, "errors": 0, "skipped": 0}
Ran 1 test in 0.024s
OK
```

## VERIFICATION
```text
Deleted-entry mutation rejected:
unlisted lesson: 0001-single-table-assumption.html
authored gap: 1
{
  "counts": {
    "html": 222,
    "notebooks": 410,
    "solutions": 148,
    "students": 221
  },
```

<details>
<summary>Exact commands (all arguments preserved)</summary>

```sh
# RED and GREEN
/home/avist/Projects/relational/.venv/bin/python -B -c '
import importlib.util,json,pathlib,sys,unittest
root=pathlib.Path.cwd(); suite=unittest.TestSuite(); ids=json.loads(sys.argv[1])
for entry in ids:
    file, identity=entry.split('"'"'::'"'"',1)
    spec=importlib.util.spec_from_file_location('"'"'lead_replay_'"'"'+str(len(suite._tests)),root/file)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target=module
    for name in identity.split('"'"'.'"'"'): target=getattr(target,name)
    cls=getattr(module,identity.split('"'"'.'"'"')[0]); suite.addTest(cls(identity.split('"'"'.'"'"')[-1]))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('"'"'LEAD_RESULT='"'"'+json.dumps(dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))))
sys.exit(0 if result.wasSuccessful() else 1)
' '["tests/ci/test_readonly_export_dependencies.py::ReadOnlyExportDependencies.test_runner_installs_readonly_export_import_requirements_before_discovery"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
