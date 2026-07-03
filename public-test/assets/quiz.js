/**
 * Reusable multiple-choice quiz widget.
 * Usage: Quiz.mount(containerElement, { question, options, correct, explain, shuffle })
 * Each option: { label: string, value: string }
 * All option labels should be equal length (words/chars) when possible.
 * Options are shuffled on mount by default (shuffle: false to disable).
 */
(function (global) {
  "use strict";

  function shuffleArray(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
    return a;
  }

  function mount(container, config) {
    var question = config.question;
    var options = config.options;
    var correct = config.correct;
    var explain = config.explain || "";
    var shuffle = config.shuffle !== false;

    if (shuffle) {
      options = shuffleArray(options);
    }

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
    var buttonsByValue = {};

    options.forEach(function (opt) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "quiz-option";
      btn.textContent = opt.label;
      btn.dataset.value = opt.value;
      buttonsByValue[opt.value] = btn;
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
          var correctBtn = buttonsByValue[correct];
          if (correctBtn) {
            correctBtn.classList.add("correct");
          }
          feedbackEl.textContent = "Not quite. " + explain;
          feedbackEl.className = "quiz-feedback wrong";
        }
      });
      optsEl.appendChild(btn);
    });
  }

  global.Quiz = { mount: mount };
})(window);
