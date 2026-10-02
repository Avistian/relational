# Lesson 197 Implementation Plan

**Goal:** Build the approved evidence-backed landscape essay and complete audit.
**Architecture:** Copy authenticated predecessor evidence and source functions into an isolated lesson packet. Recompute reports without writing predecessor files; provide an independent verifier and three live student contracts. Build HTML and notebooks from shared lesson prose and visible source.
**Tech Stack:** Python, NumPy, BeautifulSoup, SciPy, nbformat/nbclient, HTML/CSS/JavaScript, Playwright.

1. Freeze protocol and inputs in labs/evidence/l197. Add a budget runner. Write behavior tests for coverage, evidence admission and rubric completeness; run RED before implementing labs/relkit/landscape_l197.py.
2. Implement labs/_audit_l197.py using pinned replay functions, compare complete reports to original bytes and retain statuses. Independently rerun predecessor verification in the isolated packet; test missing/corrupt inputs and false evidence promotion. All execution through labs/_budget_l197.py.
3. Write lessons/content/0197-year-5-essay.md, evidence worked essay and blank learner template. Build reference/year-5-essay.html, notebooks and lesson via labs/_build_l197.py. Extend the existing architecture widget without changing route behavior; add landscape writing feedback. All student functions feed final output and fail on wrong implementations.
4. Execute solution from an empty directory and require exact report equality. Check widget states, keyboard/reset/no-JS/print, figures, links, source parity, deterministic generation and manifest galleries. Update curriculum, resources, notes, dossier and author-preparation learning record.
5. Seal deliverables and run actual Pages build in temporary Git-index checkout with required predecessor packages. Verify copied hashes and links, preserve user's index, report execution budget and remaining evidence limits. No deployment.

Approved execution continues in the shared workspace because required predecessor evidence is uncommitted here. No unrelated changes will be committed or reverted. Numerical and delivery commands have an aggregate 1800-second wall-time cutoff. Local documentation/build preparation is also recorded conservatively in the ledger.
