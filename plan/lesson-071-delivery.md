# Lesson 071: VIME masked tabular self-supervision

Follows the approved Year 2 Q4 curriculum. Deliver a self-contained lesson, printable reference,
student and executed solution notebooks, visible PyTorch implementation, source audit, and
label-efficiency experiment. Authored availability does not change learner completion.

Design: trace marginal replacement and collision-aware masks; derive both pretext losses;
show the encoder and both heads; distinguish frozen source behavior from fine-tuning;
implement the source's logit-variance consistency term. Compare identical labeled budgets,
splits and downstream selection across scratch, frozen VIME-self, fine-tuned VIME-self,
reconstruction-only fine-tuning and frozen VIME-semi. Use offline sklearn digits for a bounded
local study. Supply MNIST follow-up operators for Colab and Modal with explicit protocol gaps.

Alternatives: a theory-only page misses the required label-efficiency lab; a full clinical or
genomics replication needs restricted data. The offline teaching experiment plus public MNIST
follow-up provides a runnable package while keeping the paper claims separately cited.

Validation: independent arithmetic/collision/split/gradient tests first; pinned released-code
corruption parity; fresh local fits; executed solution; student TODO wiring; notebook image
payloads and browser controls at desktop/mobile widths; copied Pages staging and local links.
No claim of live Colab execution or deployment without testing those environments.
