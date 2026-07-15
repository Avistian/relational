# Thesis Dossier

The living, skeptic-facing argument for the mission's contrarian bet. This is not lesson notes — it is the
**case**: claims, the evidence each lesson adds, and (just as important) the **counter-evidence**. By the
time results matter (Y3–Y6), this should read as an honest brief a reviewer could not easily dismiss.

**Update ritual.** After each lesson, add one line to the Evidence Ledger: what the lesson contributed
*for* the thesis, *against* it, or *to the honest bar* it must clear. Revise the Current Verdict when the
balance shifts. Never delete counter-evidence — a thesis that only accumulates supporting points is
propaganda, not a case. Related: `MISSION.md` (why), `NOTES.md` (teaching standards), lesson "thesis
bridge" callouts (the raw material).

---

## The thesis

> Relational deep learning (RDL) and foundation relational models are **undervalued**: by learning
> directly over a database's relational structure instead of a hand-flattened single table, they can
> unlock predictive value the dominant single-table paradigm systematically discards.

### Sub-claims (each must be defended, not assumed)

- **C1 — Flattening is lossy.** Collapsing a relational database into one design matrix throws away
  structure (entity identity, shared groups, event sequences, many-to-many links) that carries signal.
- **C2 — The loss is *recoverable* by learning over structure.** A model that operates on the relational
  graph can exploit what flattening discarded, end-to-end, without hand-crafted joins.
- **C3 — The gain is *real and fair*.** The advantage survives honest evaluation — temporal splits, no
  leakage, and a genuinely strong single-table baseline (tuned GBDT + leak-free stacked ensemble), not a
  strawman.
- **C4 — The field undervalues this.** Relative to attention/compute spent elsewhere, the relational
  frontier is neglected given its potential.

---

## Evidence ledger

Legend: **[FOR]** supports a sub-claim · **[BAR]** raises the honest baseline the thesis must beat ·
**[AGAINST]** genuine counter-evidence.

| Lesson | Contribution | Type | Bears on |
|--------|--------------|------|----------|
| L001 | RDL learns over the relational entity graph; flattening is the lossy step it replaces (Fey 2024). | FOR | C1, C2 |
| L002 | Point-in-time flattening is not just lossy but leakage-prone — the manual pipeline is fragile. | FOR | C1 |
| L004 | Honest evaluation needs nested CV / grouped splits — the discipline any thesis claim must meet. | BAR | C3 |
| L009 | Manual relational features (DFS/Featuretools) already recover *some* structure by hand — RDL's premise is to learn this end-to-end. | FOR | C2 |
| L010 | A reproducible single-table baseline (leak-free HistGBDT) is strong and cheap — the floor is high. | BAR | C3 |
| L011–L016 | GBDTs (XGBoost/LightGBM/CatBoost) are stubborn, well-engineered baselines; tuning a strong default barely moves it. | BAR | C3 |
| L018 | The real single-table bar is a *leak-free stacked ensemble* of diverse tuned models, not one default. | BAR | C3 |
| L019 | Grinsztajn: on *typical* tabular data trees win via three inductive biases; on clean data an MLP won — DL is not weak at tables. | AGAINST | C3, C4 |
| L020 | Q2 checkpoint: a sensible default GBDT reproduces published results; a big "win" should trigger leak suspicion. Bridge: the flat `adult` table discards employer identity, shared households, job sequence a model over the source DB could exploit. | BAR + FOR | C1, C3 |
| L021 | Random splits are optimistic on drifting data (random-CV 0.846 vs temporal 0.758); honest eval needs temporal splits. TabReD: on real industrial data, time-based splits change rankings and shrink XGBoost's margin. RelBench/RDL evaluate with strict time cutoffs by construction. | BAR + FOR | C3, C1 |
| L022 | Kapoor & Narayanan: leakage across 17 fields / 329 papers; a leaked feature makes a complex model appear to crush LR (demo: gap +0.217) but the win collapses to a tie (−0.009) once removed. A big relational-vs-GBDT margin is therefore a *leak hypothesis first*; every reported RDL gain must travel with a provenance/leakage audit (model info sheet), and RelBench's point-in-time cutoffs are the structural defence. | BAR | C3 |
| L023 | Demšar: a reported gap is a random variable, so an RDL "win" needs a significance test — and the obvious one lies. A naive paired t-test on CV folds is anticonservative (demo: +0.0098 gap, naive p=1.2e−5 vs corrected p=0.19); the fix is the corrected resampled t-test on one dataset and Friedman + Nemenyi CD across many. C3 is won by a gap a skeptic's *test* cannot dissolve, never by a bigger mean. | BAR | C3 |
| L024 | Grinsztajn benchmark: the fair way to compare model families is a random-search *budget curve* (default + ceiling), on curated datasets, normalized per dataset — not one tuned peak. GBTs beat NN families at every budget and are stronger defaults (repro: GBT vs MLP on credit-g, gap +0.062 default → +0.015 tuned, never closing). This raises the bar (a strong, cheap-to-tune GBT default). But the whole contest lives *inside the single-table world* — the benchmark curates away relational structure, so beating a GBT on Grinsztajn's terms is table stakes; the thesis (C1) attacks the flattening that happens before either contestant sees the data. | BAR + FOR | C3, C1 |

---

## The honest bar (what "beating the incumbent" requires)

Assembled from Q1–Q2. To make the thesis legible to a skeptic, an RDL result must:

1. Use a **fair-comparison contract** (fixed data, split, metric, tuning budget, preprocessing scope; L020).
2. Beat a **tuned** GBDT *and* a **leak-free OOF stacked ensemble** (L018), not a single default.
3. Hold under **temporal / grouped splits** with no leakage (L002–L005), not just random IID.
4. Report the **gap size and verdict honestly**; a suspiciously large win implies a leak or an unfair
   reference (L020).
5. **Prove the gap is not noise** with a correct significance test — a corrected resampled t-test on one
   dataset, or a Friedman + Nemenyi rank test (CD diagram) across tasks — plus an effect size, not a bare
   mean (L023).

---

## Skeptic's strongest objections (and our current answer)

- **"Trees already win on tabular data — why bother?"** (Grinsztajn, L019). *Answer so far:* that result is
  about *single-table* data whose biases fit trees; the thesis is that the single-table *representation*
  discards relational structure, a different axis. Not yet demonstrated — this is the Y3–Y4 burden.
- **"Just flatten harder / engineer more features."** *Answer so far:* DFS/Featuretools show manual
  recovery is possible but ad hoc and leakage-prone (L002, L009); the bet is that learning it end-to-end
  beats hand-crafting. Undemonstrated at scale yet.
- **"Modern tabular nets (RealMLP/TabM/TabPFN) already close the gap."** *Answer so far:* acknowledged —
  they narrow the single-table tree–DL gap, which is *orthogonal* to exploiting cross-table structure. To
  be tested against on the relational frontier, not dodged.
- **"Maybe RDL's reported wins are just leakage too."** (Kapoor & Narayanan, L022). *Answer so far:* the
  right worry, and the reason every RDL result in this program must ship a leakage audit (model info sheet)
  and lean on RelBench's structural point-in-time cutoffs. A suspiciously large win is treated as a leak
  hypothesis before a method hypothesis — the thesis is only credible if it survives that scrutiny.

---

## Current verdict (updated 2026-07-08, after L020 / Q2)

**Undecided, and honestly so.** Through Q2 the dossier has mostly *raised the bar* (C3): the single-table
incumbent is strong, well-engineered, and hard to beat, and DL is not weak at tables. The genuinely
*supporting* evidence (C1, C2) is still conceptual — flattening is demonstrably lossy and leakage-prone,
and manual feature synthesis hints structure is recoverable, but no result yet shows a relational model
*beating the fair bar by keeping structure*. That demonstration is the Y1-exit → Y3–Y4 burden. Standing
honestly on a high baseline is the point: it is what will make an eventual win credible.
