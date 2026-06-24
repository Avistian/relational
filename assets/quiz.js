/**
 * Reusable multiple-choice quiz widget.
 * Usage: Quiz.mount(containerElement, { question, options, correct, explain })
 * Each option: { label: string, value: string }
 * All option labels should be equal length (words/chars) when possible.
 */
(function (global) {
  "use strict";

  function mount(container, config) {
    var question = config.question;
    var options = config.options;
    var correct = config.correct;
    var explain = config.explain || "";

    container.innerHTML = "";
    container.classList.add("quiz");

    var qEl = document.createElement("div");
    qEl.className = "quiz-question";
    qEl.textContent = question;
    container.appendChild(qEl);

    var optsEl = document.createElement("div");
    optsEl.className = "quiz-options";
    container.appendChild(optsEl);

    var feedbackEl = document.createElement("div");
    feedbackEl.className = "quiz-feedback";
    container.appendChild(feedbackEl);

    var answered = false;

    options.forEach(function (opt) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "quiz-option";
      btn.textContent = opt.label;
      btn.addEventListener("click", function () {
        if (answered) return;
        answered = true;

        var buttons = optsEl.querySelectorAll(".quiz-option");
        buttons.forEach(function (b) {
          b.disabled = true;
        });

        if (opt.value === correct) {
          btn.classList.add("correct");
          feedbackEl.textContent = "Correct. " + explain;
          feedbackEl.className = "quiz-feedback correct";
        } else {
          btn.classList.add("wrong");
          buttons.forEach(function (b) {
            if (b.textContent === findLabel(options, correct)) {
              b.classList.add("correct");
            }
          });
          feedbackEl.textContent = "Not quite. " + explain;
          feedbackEl.className = "quiz-feedback wrong";
        }
      });
      optsEl.appendChild(btn);
    });
  }

  function findLabel(options, value) {
    for (var i = 0; i < options.length; i++) {
      if (options[i].value === value) return options[i].label;
    }
    return "";
  }

  global.Quiz = { mount: mount };
})(window);
