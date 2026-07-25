# 0085 — Compute policy: the escalation ladder, Modal wired up, Binder dropped

**Date:** 2026-07-25
**Status:** Infrastructure (no lesson published this session)
**Trigger:** user sidenote after L036 — *"how to go around fact that deep learning models will need more
compute than I have on laptop without GPU… make some remarks for future when you can't produce a lesson
because of computing problems."*

## The decision

**Modal and Colab are not two options for the same job — they have different operators.** That asymmetry is
the whole policy:

- **Modal runs unattended from a CLI token**, so the *agent* can launch a job, wait, and come back with a
  verified number without the user present. It is the authoring runtime.
- **Colab needs a human in a browser tab** and disconnects after ~90 min of no *tab interaction* (not of no
  computation). It is the learner's runtime, already wired into every lab via `@colab-bootstrap`.

So: a number the lesson must quote → **Modal**. A GPU the student needs to do the lab → **Colab**, with a
smaller CPU config that still passes the CHECK cells.

Full policy, with the escalation ladder and the honesty rule, is **NOTES standard #20**; the operational
version is now in the `lab-authoring` skill (new *Compute budget* section).

## The ladder (short form)

0. **Suspect thread oversubscription before hardware** — it has been the real cause every time so far
   (L031 HistGB 21 s → 0.28 s; L036 LightGBM 210 s → 9.2 s). Free, 10–20×.
1. Shrink the measurement, not the claim (identical config across arms; state it).
2. Change the scope, not the paper (forward-only / key-parts / gradual across labs — standard #18).
3. Tier C synthetic, or a torch-free stand-in (L030).
4. **Modal** for an agent-side verified number.
5. **Colab-GPU** for a student-side GPU lab.

**Honesty rule:** a downscaled run is a *different experiment*. State the config in the lab intro and the
learning record, never quote a shrunken result as the paper's, and if the affordable version cannot support
the claim, say the claim is unsupported rather than shipping a toy that looks like evidence.

## What shipped

- **`modal/common.py`** — shared harness (App `relational-labs`, `image_cpu` / `image_gpu`, volume
  `relational-artifacts` at `/artifacts`, two smoke functions, one entrypoint with `--gpu`). Pattern copied
  from `~/Projects/curie-llm/modal/common.py` at the user's direction — the user had already built a Modal
  setup there (22 job files, `modal run --detach`, volumes for artifacts).
- **`modal/README.md`** — runbook: status, workflow, cost table, gotchas.
- **NOTES standard #20** and the `lab-authoring` **Compute budget** section.

## Verified live (both smoke tests pass)

The token from `curie-llm` was **already on this box** (`~/.modal.toml`, workspace `pszar92`; CLI
`~/.local/bin/modal`, client 1.4.1, pipx — *not* in the project `.venv`). **No signup was needed; the
escalation rung is live today.** Agent shell calls need `full_network` — the sandbox blocks modal.com.

```
CPU image: {'python': '3.12.6', 'cores': 24, 'sklearn': '1.9.0', 'lightgbm': '4.7.0', 'fitted_trees': 50}
GPU image: {'torch': '2.13.0+cu130', 'cuda_available': True, 'device': 'Tesla T4', 'matmul_ok': True}
```

Cost frame: Starter is $0 with **$30/month** of credits (no rollover, no card) ≈ **50 T4-hours**, ~14
A100-40GB-hours; CPU at $0.135/core-hour makes batch CPU work effectively free. **RelBench scale (Y3–Y4)
will not fit in $30** → apply for Modal's academic credits (up to $10k) *before* Year 3.

### Three findings from the smoke test (why smoke tests exist)

1. **Return only plain stdlib types from a remote function.** The return value is unpickled *locally*, where
   there is no torch. `torch.__version__` is a `TorchVersion` instance, not a `str`, so returning it raw
   died with `DeserializationError`. Fixed with `str()`.
2. **One `@app.local_entrypoint()` per file** or `modal run <file>` cannot pick a target (hence `--gpu` as a
   flag rather than a second entrypoint).
3. **Version drift is real:** the cloud image resolves to Python 3.12.6 / sklearn 1.9.0 against the local
   3.12.3. Fine for a self-contained cloud experiment; **pin exact versions if a cloud number must be
   comparable to a laptop number.**

## Binder dropped (user decision)

Auditing the compute paths surfaced a defect: **`binder/requirements.txt` omitted torch, xgboost, lightgbm
and catboost** while `notebooks.html` advertised Binder as "the *real* environment". The advertised
zero-setup run path had therefore been **silently broken for every lab from L014 on (~20 labs)** since the
gallery shipped at L012 — nobody noticed because the failure only appears after a Binder build completes.

Given the choice between paying for slow Binder image builds and consolidating, the user chose to drop it.
**Colab is now the single canonical run-anywhere runtime** (working bootstrap, free T4 available). Removed:
the `binder/` directory, `binderUrl()` and the "Run on Binder" link in `assets/notebooks.js`, the legend
bullet in `notebooks.html` ("four options" → three), and the Binder mentions in `index.html`,
`labs/README.md`, `labs/_colab.py`, and the nine `_build_l0NN.py` scripts that would otherwise re-emit the
dead path if rerun. Gallery re-verified headlessly: 2 published rows, links `View · Source .ipynb · Open in
Colab`, no Binder URL, correct Colab target.

The `@colab-bootstrap` prose inside ~30 already-built notebooks still says "or Binder it does nothing" —
harmless (it *is* a no-op anywhere but Colab) and not worth rebuilding and re-rendering every lab;
`labs/_colab.py` is the source of truth and self-corrects as labs are rebuilt. `public-test/` is a stale
snapshot copy and was left untouched.

## When this actually bites

L037–040 and much of early Y2 are CPU-fine. First real trigger: **L042** (MLP/ResNet *trained*, not
forward-only). Hard requirement by **L062–L068** (TabPFN / TabICL) and unavoidable in **Y3–Y4** (GNNs;
RelBench `rel-amazon` is millions of nodes). The harness is deliberately in place *before* the first lesson
that depends on it — per the policy's own advice, never smoke-test during a lesson you are trying to ship.

## Next

Lesson 037 — **Document a baseline package**: package the (now audited and fixed) homework pipeline into a
reproducible baseline script. CPU-only; no Modal needed.
