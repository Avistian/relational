# Human evidence

Source: a1f7b15a288fb432ef4aa24b28b2ae4a36212c32
Reviewer: verifier:rel-01

BEFORE failures: 1
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 1, "failed": 1, "errors": 0, "skipped": 0}
FAIL: test_publication_scratch_uses_supported_step_runner_context (lead_replay_0.PullRequestChecks.test_publication_scratch_uses_supported_step_runner_context)
FAILED (failures=1)
```

## GREEN
```text
LEAD_RESULT={"run": 1, "failed": 0, "errors": 0, "skipped": 0}
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
' '["tests/ci/test_pr_checks.py::PullRequestChecks.test_publication_scratch_uses_supported_step_runner_context"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
