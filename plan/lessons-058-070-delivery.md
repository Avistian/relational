# Lessons 058–070: from benchmark evidence to defensible in-context learning

Authorized on 2026-09-09: create all thirteen integer lessons and push to main.
The existing Year 2 plan supplies the design; the user additionally requested quality
beyond the preceding packages. The mission is unchanged. Lettered extension units
remain separate and are neither claimed complete nor silently substituted for integer units.

## Learning progression

58 builds a benchmark evidence map and tests dataset-selection sensitivity; 59 exposes
adaptive validation overfitting; 60 freezes and evaluates a broad five-family comparison.
61 derives and trains an amortized Bayesian predictor; 62 traces the TabPFN v1 mask and
runs historical weights; 63 constructs and intervenes on an SCM prior; 64 traces the
Nature-v2 table architecture and audits accuracy/cost; 65 constructs honest query-role
embeddings; 66 implements TabICL's inducing-point column mechanism and traces the full
model; 67 builds retrieval and adaptation with disjoint queries; 68 changes causal
mechanisms over time and tests temporal eligibility; 69 diagnoses distinct open-environment
failures; 70 combines the implementation, measurement and skepticism into a frozen comparison.

## Deliverables and acceptance

Every unit ships authored HTML, a printable reference, a standalone student notebook,
an executed teacher solution, prepared student HTML, source/protocol provenance, measured
evidence and regeneration commands. New models have an end-to-end architecture explanation
and portable figure, alongside separate numerical mechanism traces. Shared components carry
retrieval, prediction and teach-back activities. TODOs must be consequential, remain blank,
and feed the live experiment rather than only a mirrored check.

The author records paper claims, operator parity, fresh measurements, frozen-result
reanalysis and unrun reproduction work separately. Historical model versions are explicit.
Dataset/seed/split/checkpoint identities accompany results. Checkpoints use real tables,
multiple seeds, paired comparisons, ranks and costs; no predetermined winner is required.
Source, behavioral, notebook, link, Pages-staging and available browser checks precede push.
Publication is verified against the pushed commit, independently of local rendering.

## Quality review

Audit each unit for a teachable derivation, numerical worked example, a prediction that
can be falsified, a useful failed case, a live implementation exercise, and a defensible
EXIT artifact. Check every result against its originating data. In particular, query
labels must never enter context or embedding extraction; repeated validation decisions
must be accounted for; source parity must not be described as paper reproduction.

## Completed local evidence and delivery review

- Thirteen solution notebooks executed after final notebook regeneration; 174 code cells.
- 462 saved score records reconstructed, including explicitly reused L064 predictions.
- L070 includes 105 selected task/model/seed records across seven named arms. The current
  extension contains actual TabPFN-2.5-synthetic, TabPFN-3 and TabICLv2 checkpoint inference.
  The five-dataset Friedman test has p=0.433; the descriptive mean-rank ordering is not
  evidence of an established overall winner. Historical and expanded panels remain separate.
- Copied-weight v1 encoder-block outputs and input gradients agree with the pinned source;
  full v1/v2/TabICL architecture or paper-table reproduction is not claimed.
- All 13 lessons and 13 prepared labs passed Chromium checks at 1100px and 375px. Prediction
  commits/reveals, slider endpoints/reset, image decoding, local requests and page overflow
  were checked. Mobile numerical figures scroll horizontally to retain readable labels.
- 419 local artifact links resolve in a copied tree built with the actual Pages copy commands.
- Source archives preserve the exact measured operators when later CLI/resume guards differ.
  A fresh five-family smoke passed, resuming preserved results, and altered data identity
  was rejected. Portable notebook paths and Python syntax checks passed.
- Teacher solutions remain local and ignored under the course convention. Student TODOs
  are blank. Live Colab and Modal execution remain NOT_RUN. Remote publication is checked
  after the authorized push, independently of these local results.

Rebuild: `python labs/_build_foundation.py`; execute local solutions with
`python labs/_execute_foundation.py`; check delivery with
`python labs/_delivery_check_foundation.py`. Real-browser checks additionally require
Playwright and Chromium: `python labs/_browser_check_foundation.py`.
