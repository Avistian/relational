# Human evidence

Source: 31e07db926d3b08855aa2b651bb51cd75c75719e
Reviewer: verifier:w1-rel-20

BEFORE failures: 4
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 4, "failed": 4, "errors": 0, "skipped": 0}
FAILED (failures=4)
```

## GREEN
```text
LEAD_RESULT={"run": 4, "failed": 0, "errors": 0, "skipped": 0}
OK
```

## VERIFICATION
```text
FAIL hash check:
  x: sha256 mismatch
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
' '["tests/env/test_requirements.py::RequirementCoverageTests.test_repository_baseline_is_explicit", "tests/env/test_requirements.py::RequirementCoverageTests.test_normalized_names_extras_comments_and_versions", "tests/env/test_requirements.py::RequirementCoverageTests.test_disposable_new_unlocked_dependency_fails", "tests/env/test_requirements.py::RequirementCoverageTests.test_existing_covered_dependency_cannot_become_a_silent_gap"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
