# L070 immutable public data inputs

These are the exact cached OpenML Parquet files used by the historical L070 experiment, retained so a fresh checkout can independently reconstruct its original row and byte identities. They contain the full source table and target before the lesson's documented 600-row cap. See `manifest.json` for each original OpenML data ID, source URL, byte count and SHA256.

- Pima Indians diabetes: [OpenML 37](https://www.openml.org/d/37), 768 rows, eight features.
- Blood transfusion service center: [OpenML 1464](https://www.openml.org/d/1464), 748 rows, four features.
- KC1 software defects: [OpenML 1067](https://www.openml.org/d/1067), 2,109 rows, 21 features.
- Phoneme: [OpenML 1489](https://www.openml.org/d/1489), 5,404 rows, five features.

These are public benchmark inputs attributed to their OpenML source records, not newly collected data. The breast-cancer control comes from sklearn's bundled Wisconsin Diagnostic Breast Cancer dataset and is checked by its decoded-data hash; no external download is required.

`labs/_prepare_l070_v2.py` verifies the bundle and stages missing files into the conventional course cache. It verifies an existing matching cache and rejects a conflicting one without overwriting it. Rewriting a logically identical Parquet table may change file bytes, so fetch-and-reserialize is not a substitute for this original byte snapshot.
