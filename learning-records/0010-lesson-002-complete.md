# Lesson 002 complete — design matrix & leakage

User classified join-sketch features A–F with explicit prediction time **t**. Scored 5/6: correct on all temporal features (A, B, C, E, F). Misclassified **D (country)** as unsafe, reasoning that new countries might appear after the label window — conflating population drift with leakage. Correct framing: static customer attributes knowable at **t** are safe unless retroactively back-filled.

**Evidence:** cold classification of A–F in chat; articulated temporal test (“uses future orders”) consistently.

**Implications:** Lesson 003 (train/valid/test & stratified CV) unlocked. Do not re-teach PIT aggregates; connect splits to pipeline bleed from L002. User already uses grouped CV — L003 covers i.i.d. stratified splits; L004 will deepen grouped/nested CV.
