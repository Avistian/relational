/**
 * Teach-back / Feynman widget (free recall + generation effect).
 *
 * Recognition (multiple choice) is the weakest retrieval; explaining a concept in your OWN words is
 * the strongest, and it trains the mission's goal — "explain the full stack without hand-waving".
 * The learner writes an explanation, THEN reveals a model answer + a checklist of the points a good
 * answer must hit, and self-grades against it. Prose can't be machine-graded here, so the widget also
 * nudges the learner to paste their explanation to the teacher (Cursor chat) for real feedback.
 *
 * Usage (once per lesson, on the load-bearing idea):
 *   <div id="teachback1"></div>
 *   <script src="../assets/teachback.js"></script>
 *   Teachback.mount(document.getElementById("teachback1"), {
 *     prompt: "In your own words: why is a tree's lack of rotational invariance an ADVANTAGE on tables?",
 *     points: [
 *       "Table columns are individually meaningful (age, balance, …)",
 *       "A tree splits on one axis at a time, exploiting that",
 *       "A rotation-invariant MLP treats a real column and a scrambled mixture the same",
 *       "So invariance discards the structure that made columns informative"
 *     ],
 *     model: "Each column of a table means something on its own, so the meaningful directions ARE the axes. A tree splits one feature at a time, so it respects those axes; an (approximately) rotation-invariant MLP would learn the same thing after mixing columns, which means it cannot tell a meaningful column from an arbitrary linear combination — it throws the structure away."
 *   });
 *
 * Config:
 *   prompt   the explain-in-your-own-words question
 *   points   key points a good answer hits (self-graded checklist)
 *   model    a model answer to compare against
 *   minChars minimum characters before reveal unlocks (default 20 — forces an attempt)
 */
(function (global) {
  "use strict";

  function mount(container, config) {
    config = config || {};
    var points = config.points || [];
    var minChars = config.minChars || 20;

    container.innerHTML = "";
    container.classList.add("teachback");

    var tag = document.createElement("div");
    tag.className = "tb-tag";
    tag.textContent = "Explain it";
    container.appendChild(tag);

    var prompt = document.createElement("div");
    prompt.className = "tb-prompt";
    prompt.textContent = config.prompt || "";
    container.appendChild(prompt);

    var ta = document.createElement("textarea");
    ta.className = "tb-input";
    ta.rows = 4;
    ta.placeholder = "Write your explanation from memory — full sentences, no peeking. A rough attempt still counts.";
    container.appendChild(ta);

    var reveal = document.createElement("button");
    reveal.type = "button";
    reveal.className = "tb-reveal";
    reveal.textContent = "Write a little more to unlock the model answer";
    reveal.disabled = true;
    container.appendChild(reveal);

    var answer = document.createElement("div");
    answer.className = "tb-answer";
    answer.style.display = "none";
    container.appendChild(answer);

    function currentLen() { return (ta.value || "").trim().length; }

    ta.addEventListener("input", function () {
      var ok = currentLen() >= minChars;
      reveal.disabled = !ok;
      reveal.textContent = ok ? "Reveal model answer & self-check" : "Write a little more to unlock the model answer";
    });

    reveal.addEventListener("click", function () {
      if (reveal.disabled || reveal.dataset.done) return;
      reveal.dataset.done = "1";
      reveal.disabled = true;
      ta.disabled = true;
      answer.style.display = "";

      var hitLine = document.createElement("div");
      hitLine.className = "tb-hits";
      answer.appendChild(hitLine);

      var checkedCount = 0;
      function updateHits() {
        hitLine.textContent = "Self-check — points your explanation hit: " + checkedCount + " / " + points.length;
        hitLine.className = "tb-hits" + (checkedCount === points.length && points.length ? " full" : "");
      }

      var h = document.createElement("div");
      h.className = "tb-h";
      h.textContent = "Did your explanation cover these? (tick the ones you got)";
      answer.appendChild(h);

      var list = document.createElement("div");
      list.className = "tb-points";
      points.forEach(function (pt, i) {
        var row = document.createElement("label");
        row.className = "tb-point";
        var cb = document.createElement("input");
        cb.type = "checkbox";
        cb.className = "tb-cb";
        cb.dataset.idx = String(i);
        var span = document.createElement("span");
        span.textContent = pt;
        cb.addEventListener("change", function () {
          checkedCount += cb.checked ? 1 : -1;
          updateHits();
        });
        row.appendChild(cb);
        row.appendChild(span);
        list.appendChild(row);
      });
      answer.appendChild(list);
      updateHits();

      var modelH = document.createElement("div");
      modelH.className = "tb-h";
      modelH.textContent = "Model answer";
      answer.appendChild(modelH);
      var model = document.createElement("div");
      model.className = "tb-model";
      model.textContent = config.model || "";
      answer.appendChild(model);

      var nudge = document.createElement("div");
      nudge.className = "tb-nudge";
      nudge.textContent = "For real feedback, paste your explanation to your teacher in Cursor chat — self-grading catches gaps, but a second reader catches hand-waving.";
      answer.appendChild(nudge);
    });

    return {
      getText: function () { return ta.value; },
      isRevealed: function () { return !!reveal.dataset.done; }
    };
  }

  global.Teachback = { mount: mount };
})(window);
