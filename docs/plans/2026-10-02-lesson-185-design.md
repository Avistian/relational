# Lesson 185: approved design and frozen protocol

User approved the presented scope on 2026-10-02. Experiment: `L185 relational-shortcut-intervention-v1`.
Full original synthetic experiment, not a published benchmark. Five seeds 0–4; 1,000 companies ×20 customers; company-disjoint 600/200/200 train/validation/test. All features and actions precede outcome; fixed recipe, no hyperparameter search or test selection. Save every seed's normalized input tables, complete-key held-out predictions, paired interventions, metrics, environment and hashes. Exact rerun required within the captured environment.

Implementation details frozen before numerical execution: company demand U~Bernoulli(.5); company badge B=U XOR Bernoulli(.05); customer action A~Bernoulli(.1+.8U); independent customer noise E~Uniform(0,1); Y=1[E<.05+.85U+.05A]. Companies share U and B; conditional customer noises independent. Baseline no cross-customer interference. Company split uses independent SeedSequence streams. Features available at time0; A at1; query cutoff1; outcome2. Query identity (customer_id,cutoff). Badges can be set independently without changing U or A. Actions replace the A mechanism; identical E is reused in paired potential outcomes. No post-outcome features.

Train-only conditional frequency predictors: badge-only, action-only and (U,A). Unseen cell fallback train global mean. No validation-based model choice; score all three on validation and test. Test population estimates: unadjusted action risk difference, U-standardized action risk difference, paired action effect, paired badge effect, expected effects, and policy gains over observed outcomes. Also set badge1 and action1 for all test customers and retain their baseline. Report every seed and sample SD across seeds (not an IID customer confidence interval).

Theory: population ATE(action)=.05; ATE(badge)=0. With balanced U, observed action association=.73; badge association=.801; E[Y|B=1]= .9005 vs E[Y|B=0]=.0995. E[Y|do(B=b)]= .5 for either b. These analytic values are checked independently before displaying; correct any arithmetic transcription before numerical execution. RCD is optional context only, not implemented.

Numerical budget: USD0 cloud/API; 600 aggregate wall seconds including tests, execution, reruns and notebook validation. One enclosing subprocess timer per numerical job, persistent ledger, lock, timeout at remaining budget; timeout/failure charged. No silent seed/data reduction. INCOMPLETE on exhaustion. Source downloads and artifact rendering are nonnumerical preparation; no billed resources.

Deliver complete lesson, reference, visible implementation, three TODO/CHECK tasks, executed portable solution, reports, figures, source ledger, browser/link checks and clean-index Pages check. No deployment requested. Learner PENDING_WRITTEN_DEFENSE. Preserve unrelated existing changes.

Alternatives: conceptual-only misses executable learning; unrelated paper reproduction changes this curriculum unit. Approved original synthetic experiment is selected. Missing preceding lessons will not be described as completed prerequisites.

Pre-run verification tolerance: each seed's adjusted and paired action effect within .025 of .05; badge effect exactly zero; independent AUROC within 1e-12; fresh rerun tables and report exactly equal. These are simulator verification bounds, not inferential confidence intervals or paper-score targets.
