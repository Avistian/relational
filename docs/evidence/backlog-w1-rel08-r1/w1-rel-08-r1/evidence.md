# Human evidence

Source: ae115b5720c5257867014a38aacf7b982e314502
Reviewer: verifier:w1-rel-08-r1

BEFORE failures: 1
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 2, "failed": 1, "errors": 0, "skipped": 0}
FAILED (failures=1)
```

## GREEN
```text
LEAD_RESULT={"run": 2, "failed": 0, "errors": 0, "skipped": 0}
OK
```

## VERIFICATION
```text
    "notebooks": 410,
    "students": 221
    "solutions": 148,
  "errors": [],
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
' '["tests/notebooks/test_duplicate_ids.py::CellIdentityChecks.test_duplicate_cell_ids_block", "tests/notebooks/test_duplicate_ids.py::CellIdentityChecks.test_missing_cell_ids_remain_nonblocking_warnings"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
