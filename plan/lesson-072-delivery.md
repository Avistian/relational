# Lesson 072 delivery design

User request: create L072. Follow-up: drastically improve architecture visuals, use a
model-specific form showing the whole solution, and connect ideas within/between lessons.

The teaching question is how to construct two observations of a tabular row and test what
the encoder learned. Narrative: VIME repairs values → SCARF recognizes companions → derive
candidate competition → SubTab reconciles partial observations → evaluate a frozen probe →
L073 varies the conditions under which self-supervision helps.

SCARF diagram: concrete corruption, two branches and shared parameters, projection,
all-row comparison, objective, discard boundary, encoder-to-prediction handoff.
SubTab diagram: explicit feature coverage, three shared computation paths, decoder and
projection branches, pair losses, inference aggregation and class head. Different visual
grammar follows the actual structure, rather than reusing a generic chain of boxes.

Three TODOs feed live training. Numeric fixtures precede implementation. Audit SubTab's
released loss values and gradients. Run three real datasets and three paired seeds with raw,
random, SCARF, zero-corruption, subset reconstruction, and subset joint arms. Include local
limitations, exact split IDs/predictions, and exploratory dataset-balanced ranks. No user
completion or mastery inferred. Verify executed solution, notebook figures, responsive
browser interactions and copied Pages links. Do not publish or launch cloud training.
