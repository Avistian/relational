# L165 reproduction contract

## Selected historical target

Fey, Kocijan, Lopez, Lenssen and Leskovec, *KumoRFM: A Foundation Model for In-Context Learning on Relational Data* (2025), Table 2, rel-f1 / driver-dnf, KumoRFM **in-context**, 82.41 AUROC on the paper's 0–100 scale (0.8241 on a 0–1 scale). The adjacent 82.63 is the fine-tuned arm and is not our target. This is one selected result, not the full 30-task paper.

Primary report: [archived original](https://web.archive.org/web/20260702201348id_/https://kumo.ai/research/kumo_relational_foundation_model.pdf). Retrieved 2026-10-01; SHA256 `805febc5e9af8a8e9d34c6123fa37af4d9c2f29ea2d1d04be0e87d63e249cfa2`. Local [PDF](sources/l165/paper.pdf), [extracted text](sources/l165/paper.txt), [retrieval ledger](sources/l165/source-ledger.json).

## Protocol and availability audit

| Requirement | Source / observation | Reproduction consequence |
|---|---|---|
| Published arm and score | §4.1, Table 2: in-context 82.41 | Target identified; not independently measured |
| Model identity | Report describes v1, §2.3 | Matching weight bytes, tensor dimensions and complete inference implementation not obtained |
| Pretraining | §2: real public + synthetic data; report states no RelBench training/tuning for ICL | Exact corpus manifest, exclusion audit and checkpoint provenance unresolved |
| Test population | §4: supplied RelBench test sets | Exact archive hashes, historical query keys and task implementation not specified in sufficient detail for identity |
| Context construction | §2.2 and Eq.1: historical labels and sampled subgraphs | Exact count, sampling configuration, seeds and per-query context keys unresolved |
| Feature/label clocks | §2.2 time-consistency claim | No released matching implementation inspected; historical row availability not established |
| Scoring | AUROC; provided test sets | No per-query historical predictions, seeds or uncertainty artifacts obtained |
| Source availability | Original URL redirects to current NVIDIA product overview; old kumo-ai/kumo-rfm API lookup returns 404 | Observed endpoint failures are not proof that no release exists anywhere |
| Version boundary | Current research page's paper link points to arXiv2604.12596 (KumoRFM-2) | Modern client/service success does not establish v1 identity |

**Historical reproduction NOT_RUN; model/protocol fidelity NOT_ESTABLISHED.** No fabricated v1 trainer is shipped. If matching artifacts are obtained: pin all bytes, reconstruct complete task keys and label semantics, reproduce exact context policy/seeds, verify masking and owner cutoffs, obtain resource estimate under the standing USD10 aggregate ceiling, then evaluate every selected test query and independently rescore. Unknown historical settings remain unknown; do not invent five seeds or a tolerance.

## Executed local experiment

**L165 KumoRFM-v1 Context and Reproduction Audit**. Standard-library CPU, zero cloud/API spend, deterministic exhaustive fixtures. No training, model inference, service calls or benchmark dataset downloads. Integer time units are illustrative, not F1 calendar semantics.

- 144 context cases: 4 anchors × 3 horizons × 3 arrival delays × 4 query times. Earlier anchor, fully matured label and available label required.
- 96 graph cases: 4 owner cutoffs × 3 hop limits × all 8 masks of 3 edges. Filter rows before traversal; late bridges cannot reveal further rows.
- 24 prediction orderings of the same 4 invented query records. Complete entity/cutoff keys; tie-aware AUROC 0.875. These are not seeds or benchmark performance.
- Two isolated hidden-label values produce the same course query/context packet. Model-internal masking is outside this audit.
- Independent path enumeration, 1,134 pairwise AUROC oracle cases, three incorrect learner implementations rejected.

The simplified graph has static undirected FK-like links and explicit availability times. It does not model updated values, key changes, arbitrary label-generating SQL, or the proprietary graph transformer. The course uses inclusive fact/arrival cutoffs and strict earlier context anchors. Historical production boundary conventions remain unverified.

## Re-run

From repository root, Python3.10+:

```sh
python3 labs/_run_l165.py
python3 labs/_verify_l165.py
```

The supplied portable solution embeds inputs, all three mechanisms, checks and runner. It needs only Python's standard library. Development build/execution uses the course `.venv` (nbformat, nbconvert, nbclient, matplotlib, Playwright); see receipt versions. For bounded author commands use `python3 labs/_budget_l165.py <command>`. Its 1800-second aggregate allowance includes recorded checks plus conservative preflight reservation. Budget exhaustion stops the command; no paid fallback. Author-created fixtures/proofs do not demonstrate learner mastery.

## Evidence labels

Local audit COMPLETE; published result NOT_RUN; historical fidelity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. Notebook/browser/build receipts are separate. Live Colab and deployment NOT_CHECKED. No push or publication requested.
