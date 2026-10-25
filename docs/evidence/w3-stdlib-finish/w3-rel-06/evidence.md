# Human evidence

Source: 7134f5d33be84c5e98d90f8f444ffd9ba740d44b
Reviewer: verifier:w3-rel-06

BEFORE failures: 2
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 1, "failed": 2, "errors": 0, "skipped": 0}
FAILED (failures=2)
```

## GREEN
```text
LEAD_RESULT={"run": 1, "failed": 0, "errors": 0, "skipped": 0}
OK
```

## VERIFICATION
```text
Deleted-entry mutation rejected:
unlisted lesson: 0001-single-table-assumption.html
authored gap: 1
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
' '["tests/update_manifest/test_boolean_numbers.py::BooleanNumbers.test_check_rejects_boolean_version_and_year_without_writes"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
