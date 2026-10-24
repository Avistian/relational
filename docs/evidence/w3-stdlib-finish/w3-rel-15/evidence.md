# Human evidence

Source: 9da188814c8c16647275bf6aa539afa7b1115e84
Reviewer: separate-verifier

BEFORE failures: 18
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 7, "failed": 18, "errors": 0, "skipped": 0}
```

## GREEN
```text
LEAD_RESULT={"run": 7, "failed": 0, "errors": 0, "skipped": 0}
```

## VERIFICATION
```text
Deleted-entry mutation rejected:
unlisted lesson: 0001-single-table-assumption.html
authored gap: 1
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
' '["tests/browser_checks/test_output_paths.py::OutputPathsTests.test_out_selects_review_directory_for_all_scripts", "tests/browser_checks/test_output_paths.py::OutputPathsTests.test_out_wins_over_report_dir_in_either_order", "tests/browser_checks/test_output_paths.py::OutputPathsTests.test_defaults_preserve_existing_destinations", "tests/browser_checks/test_output_paths.py::OutputPathsTests.test_legacy_report_dir_preserves_suffixes", "tests/browser_checks/test_output_paths.py::OutputPathsTests.test_live_defaults_and_out", "tests/browser_checks/test_output_paths.py::OutputPathsTests.test_review_literals_only_occur_in_argument_defaults_and_export_is_preserved", "tests/browser_checks/test_output_paths.py::OutputPathsTests.test_generated_directories_ignored_and_tracked_pngs_retained"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
