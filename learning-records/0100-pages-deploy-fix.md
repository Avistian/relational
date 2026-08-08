# 0100 — GitHub Pages deploy: the retry wrapper was the bug

**Date:** 2026-08-08
**Status:** Fixed and verified (two consecutive green runs, live site current).
**Trigger:** User (`/teach`): *"There were github pages publish error, fix that first and the create lesson 43"*.

## Symptom
Run [31099288142](https://github.com/Avistian/relational/actions/runs/31099288142) (the paper-mirror
doctrine merge, `be919fbe`) failed after **12m22s**. `build` was green in 8s and uploaded a valid artifact;
`deploy` burned ~12 minutes and ended `Attempts exhausted, made 3 attempts`.

## Root cause
A Pages deployment is **keyed on the commit SHA** (the log shows `ID: be919fbe…` = the SHA).

1. The Pages backend stalled: `deploy-pages` polled `Current status: deployment_queued` for ~10 minutes.
2. At its `timeout` (default 600000 ms) the action **cancelled that deployment** — by design.
3. `Wandalen/wretry.action` then retried the *same action on the same SHA*. Re-creating the deployment
   returned the already-cancelled record, so attempts #2 and #3 failed in ~5s each with
   `##[error]Deployment cancelled.`

So the retry wrapper could never recover from a timeout — it only ever helped for *fast* transient
failures — and it converted one slow deploy into a hard red run with a misleading error.

## What was NOT wrong (checked before changing anything)
- Pages config correct: `build_type: "workflow"`, source `main`.
- `github-pages` environment has only a branch policy, and it allows `main` (no wait timer / reviewer gate).
- The artifact was valid and the **site was never stale for L042** — the earlier "lesson 42" push
  (run 31047408437) deployed fine. The failing commit changed only `NOTES.md`, a learning record, and skill
  files, i.e. **no site content at all**.

## The fix (`.github/workflows/pages.yml`)
- **Removed the `wretry` wrapper**; `actions/deploy-pages@v5` is called directly. Transient status-report
  errors are already retried *inside* the action (`error_count`, default 10), which is what the wrapper was
  originally added for (L15/L17 "try again later").
- **Added `paths-ignore`** for teaching bookkeeping (`NOTES.md`, `MISSION.md`, `RESOURCES.md`, `SESSION.md`,
  `GLOSSARY.md`, `CURRICULUM*.md`, `misconceptions.md`, `thesis-dossier.md`, `learning-records/**`,
  `reviews/**`, `plan/**`, `modal/**`, `solutions/**`, `.agents/**`, `.cursor/**`) so a doc-only commit —
  exactly the commit that failed — no longer spends a deployment. A commit touching both docs and site
  files still deploys. `.github/**` is deliberately **not** ignored, so workflow changes self-test.
- **Bumped off deprecated Node 20:** `checkout@v4→v5`, `configure-pages@v5→v6`,
  `upload-pages-artifact@v3→v5` (which pins `upload-artifact@v7`). `deploy-pages@v5` was already node24.

## Dead end worth remembering
First attempt raised `timeout` to `1200000`. The run warned:
`timeout value is greater than the allowed maximum - timeout set to the maximum of 600000 milliseconds`.
**The deploy timeout cannot be extended past the 10-minute default** — the knob is clamped. Removed the
override and documented the cap in the workflow comment rather than shipping a setting that does nothing
but emit a warning.

## Recovery path for a future stall
A stalled/cancelled deployment is tied to its SHA, so the reliable recovery is **deploying a new commit**
(verified: `5476db7` and `98eae16` both deployed in ~10s). Re-running the same failed run re-uses the same
SHA and was not verified to recover.

## Verification
- Runs [31265747042](https://github.com/Avistian/relational/actions/runs/31265747042) and
  [31265807207](https://github.com/Avistian/relational/actions/runs/31265807207): both **success**
  (build 7s, deploy 10s), the second with **no annotations** at all.
- Live site: `lessons/manifest.json` → 42 lessons, last = L042; `/`, `lessons/0042-*.html`,
  `labs/html/0042-*.html`, `assets/protocol-bakeoff-viz.js`, `flashcards.html` all HTTP **200**.
