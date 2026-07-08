/**
 * Prediction-before-reveal widget (the pretesting effect).
 *
 * Ask the learner to COMMIT a prediction before they read the answer. Committing first —
 * even wrongly — primes far stronger encoding than reading the result cold (pretesting effect).
 * Place this near the TOP of a lesson, before the section that reveals the number/outcome.
 *
 * The learner picks an option (locked once chosen), then reads on. A "Reveal" button shows
 * the truth, whether their prediction matched, and a one-line explanation.
 *
 * Usage:
 *   <script src="../assets/predict.js"></script>
 *   Predict.mount(document.getElementById("predict1"), {
 *     prompt: "Before reading: on clean, all-informative features, who wins — tree or MLP?",
 *     options: [
 *       { label: "The MLP (neural net) wins", value: "mlp" },
 *       { label: "The tree ensemble wins",   value: "tree" }
 *     ],
 *     correct: "mlp",
 *     reveal: "The MLP won 0.965 vs 0.936 — trees only pull ahead once junk features pile up."
 *   });
 *
 * Config:
 *   prompt      the prediction question (string)
 *   options     [{ label, value }] — keep labels similar length
 *   correct     value of the option that turns out true (optional; omit for open predictions)
 *   reveal      text shown when the learner clicks Reveal
 *   shuffle     shuffle option order (default true)
 */
(function (global) {
  "use strict";

  function shuffleArray(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  function mount(container, config) {
    config = config || {};
    var options = config.shuffle !== false ? shuffleArray(config.options || []) : (config.options || []).slice();

    container.innerHTML = "";
    container.classList.add("predict");

    var tag = document.createElement("div");
    tag.className = "predict-tag";
    tag.textContent = "Predict first";
    container.appendChild(tag);

    var qEl = document.createElement("div");
    qEl.className = "predict-prompt";
    qEl.textContent = config.prompt || "";
    container.appendChild(qEl);

    var optsEl = document.createElement("div");
    optsEl.className = "predict-options";
    container.appendChild(optsEl);

    var revealBtn = document.createElement("button");
    revealBtn.type = "button";
    revealBtn.className = "predict-reveal";
    revealBtn.textContent = "Reveal the answer";
    revealBtn.disabled = true;

    var outcome = document.createElement("div");
    outcome.className = "predict-outcome";

    var committed = null;
    var byValue = {};

    options.forEach(function (opt) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "predict-option";
      btn.textContent = opt.label;
      btn.dataset.value = opt.value;
      byValue[opt.value] = btn;
      btn.addEventListener("click", function () {
        if (committed) return;
        committed = opt.value;
        var buttons = optsEl.querySelectorAll(".predict-option");
        buttons.forEach(function (b) { b.disabled = true; });
        btn.classList.add("committed");
        revealBtn.disabled = false;
        revealBtn.textContent = "Committed — reveal the answer";
      });
      optsEl.appendChild(btn);
    });

    revealBtn.addEventListener("click", function () {
      if (!committed || revealBtn.dataset.done) return;
      revealBtn.dataset.done = "1";
      revealBtn.disabled = true;
      var matched = config.correct ? committed === config.correct : null;
      if (config.correct && byValue[config.correct]) {
        byValue[config.correct].classList.add("truth");
      }
      var lead = matched === null ? "" : (matched ? "You predicted correctly. " : "Your prediction missed — good, now it will stick. ");
      outcome.textContent = lead + (config.reveal || "");
      outcome.classList.add(matched === false ? "missed" : "matched");
    });

    container.appendChild(revealBtn);
    container.appendChild(outcome);

    return { getCommitted: function () { return committed; } };
  }

  global.Predict = { mount: mount };
})(window);
