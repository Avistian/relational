# L111 · OGB benchmark contract

Course setup/baseline experiment, not a paper-score reproduction. Full ogbn-arxiv, official year split, official OGB 1.3.6 library-agnostic loader and Evaluator. Constant class fitted using training labels only; smallest class wins a tie. No neural training, tuning or paid compute.

Raw archive: https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
SHA-256: `49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276`.
The notebook verifies this hash before extracting to a disposable directory and asking the official loader to parse raw CSV files. This avoids old processed pickle compatibility issues without changing shared caches. Split and label hashes are in `evidence/l111/summary.json`.

From the repository root:

```bash
.venv/bin/python labs/_check_l111.py
L111_ARCHIVE=labs/data/l112/arxiv.zip .venv/bin/python labs/relkit/ogb_contract_l111.py
.venv/bin/python labs/_figures_l111.py
.venv/bin/python labs/_build_l111.py
.venv/bin/python labs/_execute_l111.py
```

Without the existing archive, omit `L111_ARCHIVE`: the visible operator downloads it to `l111-data/arxiv.zip`. The standalone notebook also downloads the archive when absent. It contains all implementation code and two live TODO/CHECK functions. Runtime versions are recorded in `evidence/l111/environment.json`.

Official accuracy equals independent correct/total on all three splits. Raw features and graph are loaded, but this constant predictor does not use them. This verifies a baseline and evaluation contract, not a competitive learned predictor. L112 supplies the separately named complete published GCN lane.

No full-paper parity, historical execution identity, live Colab UI or learner mastery is claimed. Learner status: PENDING_WRITTEN_DEFENSE. Publication checks are recorded in `reviews/lessons-103-135/`.
