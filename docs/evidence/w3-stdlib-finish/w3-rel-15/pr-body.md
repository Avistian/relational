Add `--out` to all four browser review scripts. It selects the review/report root
and takes precedence over `--report-dir`, while defaults, legacy suffixes and
`assets/course-visuals` exports retain their destinations. Ignore generated reviews
and scratch files while preserving tracked artifacts.

Validation: genuine tests-only RED commit (7 tests, 18 assertion failures), frozen
GREEN fixtures (7 passing), configured lead-verify (74 passing, clean GREEN source).
Stdlib CLI-prefix/AST fixtures only; browser execution and canonical replay NOT_RUN.
Separate review is pending. Local draft only; publication is held.
