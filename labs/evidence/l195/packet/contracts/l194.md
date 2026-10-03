# L194 — complete evidence analysis and report

Approved scope: authenticate/replay L193 evidence; report every one of the 21 RDBLearn v1 tasks; no source repair or new model inference. Full target remains 567 validation candidates and 63 selected tests. Status INCOMPLETE_SOURCE_PREPROCESSING_GATE, fresh inference NOT_RUN, historical identity NOT_ESTABLISHED, learner PENDING_WRITTEN_DEFENSE.

## Exact commands

From repository root:

```bash
python3 labs/_budget_l194.py .venv/bin/python labs/_replay_l194.py
python3 labs/_budget_l194.py .venv/bin/python labs/_verify_l194.py
python3 labs/_budget_l194.py .venv/bin/python labs/_build_l194.py
python3 labs/_budget_l194.py .venv/bin/python labs/_execute_l194.py
python3 labs/_budget_l194.py .venv/bin/python labs/_delivery_l194.py
```

The portable notebook uses Python standard library only and embeds the packet with an archive hash. Repository build/check commands use the existing project environment (nbformat, nbclient, nbconvert, matplotlib, BeautifulSoup, Playwright). No dependency added. The notebook implements report logic inline; no trained model is hidden behind a replay claim. `_prepare_l194.py` is the author-time snapshot command and requires the preserved L193 inputs. Replaying checks hashes and key invariants, not the truth of arbitrary self-signed evidence.

## Frozen identity and deviation ledger

Paper https://arxiv.org/html/2602.18495v1, Tables 1–3. Inherited source release 0.1.2 / b5b03ebf8091547285a6e06cba53d2d1a40cb171, FastDFS 0.2.1; official splits; support cap 10000; depths 2/3/4 and TabPFNv2/v2.5/LimiX-16M. Course seeds 0/1/2 are not historical seed identities. Saved source diagnostic replay is not a fresh original-source run. The shared source gate remains failed; real-task trigger frequency and score impact are unknown. Raw data, temporal feature availability, checkpoint bytes, historical environment, regression normalizers and backend executor are still unresolved.

Analysis comparator fixed before calculating gaps: AutoGluon+DFS for all 21 tasks. Positive oriented gaps favor RDBLearn. Published scalar signs are descriptive, not paired prediction replay, significance or causal attribution. Do not aggregate different metric units. Scores/seed SDs for all fresh tasks remain null. Label-coverage, path-ablation and cold-start plans remain NOT_RUN. Cold-start strata are observational; report class counts and undefined AUROC explicitly.

## Cost and continuation

$0 new cloud/API; 1800 seconds aggregate local numerical/build/check execution, including failed attempts. The wrapper records elapsed time and kills descendant processes at cutoff. No paid model run, deployment or fresh training is authorized. A source repair is a separately named experiment; before dispatch it needs data/checkpoint admission, validated executor, regression definitions and measured full-grid cost. L193's $10 ceiling is not proof the full search is affordable. Stop and report INCOMPLETE instead of dropping tasks/seeds/candidates.

## Artifacts and evidence

`evidence/l194/report.json` and `report.md` are generated from authenticated `packet/` inputs. `input-manifest.json` pins their bytes; original L193 hashes are also retained inside the packet. `_verify_l194_results.json` records independent paper extraction and adversarial cases. `_execution_l194_results.json` records empty-directory notebook execution. `_delivery_l194_results.json` records browser and link checks. `_checkout_l194_results.json` records local isolated-index Pages checks when present. None establishes live Colab, deployment or learner mastery.
