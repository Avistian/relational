# Human evidence

Source: 030df3dc3452b590d52cce14a04476fbef4ca8cc
Reviewer: verifier:w1-rel-05

BEFORE failures: 10
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 8, "failed": 10, "errors": 0, "skipped": 0}
AssertionError: Manifest integrity checker is not implemented
FAILED (failures=10)
```

## GREEN
```text
Deleted-entry mutation rejected:
unlisted lesson: 0001-single-table-assumption.html
authored gap: 1
LEAD_RESULT={"run": 8, "failed": 0, "errors": 0, "skipped": 0}
OK
```

## VERIFICATION
```text
Deleted-entry mutation rejected:
unlisted lesson: 0001-single-table-assumption.html
authored gap: 1
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
' '["tests/manifest/test_manifest.py::ManifestIntegrity.test_real_manifest_references_and_coverage", "tests/manifest/test_manifest.py::ManifestIntegrity.test_deleted_entry_in_disposable_manifest_is_rejected", "tests/manifest/test_manifest.py::ManifestIntegrity.test_missing_entry_file_is_rejected", "tests/manifest/test_manifest.py::ManifestIntegrity.test_missing_referenced_notebook_is_rejected", "tests/manifest/test_manifest.py::ManifestIntegrity.test_missing_referenced_python_lab_is_rejected", "tests/manifest/test_manifest.py::ManifestIntegrity.test_duplicate_numeric_and_bridge_ids_are_rejected", "tests/manifest/test_manifest.py::ManifestIntegrity.test_authored_range_gap_is_rejected_even_at_endpoint", "tests/manifest/test_manifest.py::ManifestIntegrity.test_only_documented_179_gap_is_allowed"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
