# L069 reproduction and evidence contract

Primary source: [Realistic Evaluation of TabPFN v2 in Open Environments](https://arxiv.org/html/2505.16226v1). [Lesson](../lessons/0069-tabpfn-open-environment-failures.html).

## What ships

Student notebook, local executed teacher solution (ignored by Git under the course convention),
prepared student HTML, numerical mechanism/architecture/result figures, visible canonical code,
behavioral checks, source/checkpoint provenance and committed author evidence.

## Three separate claims

1. **Operator/architecture:** the specified local functions implement the named computation;
   reduced PFNs are not official checkpoint architectures. Read their explicit omissions.
2. **Measured evidence:** `_verify_l069_results.json` contains the actual local run or frozen-result
   reanalysis. Score outputs identify which implementation produced them.
3. **Paper results:** full original pretraining/benchmark replication is NOT_ESTABLISHED.
   No resource preset silently changes this verdict into MATCH.

## Regeneration

From the repository root, install `requirements-labs.txt`, then:

```bash
.venv/bin/python labs/_run_foundation.py --lesson 69 --preset lab --output labs/data/cache/foundation/l069-rerun.json
.venv/bin/python labs/_build_l069.py
```

The default notebook track needs no pretrained download. The explicit post-EXIT gate uses
an isolated package directory for historical packages; it may download immutable checkpoints.
Historical package/checkpoint hashes are in `_sources_foundation.json`; TALENT source tables
are pinned separately in `_sources_l058.json`. TabReD and public-table loaders retain their
earlier source/data identities. Do not treat a package version as a checkpoint identity.

The `smoke` preset is a small execution check. `closer` increases supported training budgets,
or reruns an already full frozen-result audit. Historical checkpoint experiments retain their
declared small protocol unless their runner explicitly says otherwise. `paper` deliberately
raises an error: the full heterogeneous paper protocols are not implemented.

For unattended execution, the supplied CPU operator is:

```bash
modal run --detach modal/foundation_repro.py --lesson 69 --preset closer
```

No Modal job or live Colab browser run is claimed by packaging this command.

## Evaluation boundaries

Read the lesson for exact dataset roster/caps, split seeds, model/inference seeds, candidates,
metric direction, context policy and omitted paper components. Compare methods on aligned
rows. Average model seeds inside datasets before ranking. Conditional seed intervals do not
cover dataset or temporal-split uncertainty. Frozen result tables are not new model fits.

The common local checkpoint records predictions, targets, selection traces and costs.
L064 explicitly reuses L070 predictions; do not count those as additional evidence.
Current-version arms have separate statuses; historical-v2 scores cannot stand in for v3.

## Delivery evidence

See `_execution_foundation_results.json`, `_source_check_foundation_results.json` and
`_delivery_foundation_results.json` for performed checks. Browser rendering, copied Pages
staging, live Colab and post-push deployment are independent checks. User mastery remains
unassessed until a completed EXIT and explanation are reviewed.
