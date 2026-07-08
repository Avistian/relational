/**
 * Spaced-retrieval warm-up widget (storage strength over fluency).
 *
 * Draws from window.RETRIEVAL_POOL (assets/retrieval-pool.js) and presents a few
 * interleaved questions from EARLIER lessons — never the lesson currently being read —
 * so practice is spaced and mixed across quarters, not massed on fresh material.
 *
 * Scheduling: a Leitner system persisted in localStorage (falls back to in-memory if
 * localStorage is unavailable, e.g. some file:// contexts). Each item has a box 1..5;
 * a correct answer promotes it (longer interval), a wrong answer resets it to box 1.
 * Items whose next-due day has arrived are shown first.
 *
 * Usage (bottom of a lesson, after loading retrieval-pool.js):
 *   <script src="../assets/retrieval-pool.js"></script>
 *   <script src="../assets/retrieval-bank.js"></script>
 *   RetrievalBank.mount(document.getElementById("warmup"), { upTo: 19, count: 3 });
 *
 * Config:
 *   upTo        current lesson number; only items with lesson < upTo are eligible
 *   count       how many questions to present (default 3)
 *   storageKey  localStorage key (default "rdl-retrieval")
 *   pool        override the pool (default window.RETRIEVAL_POOL) — for testing
 *   now         override "now" in ms (default Date.now()) — for testing
 *   storage     override storage object {getItem,setItem} — for testing
 *   shuffle     shuffle option order (default true) — quiz-fairness standard #2
 */
(function (global) {
  "use strict";

  // Days from today at which an item is next due, indexed by Leitner box (expanding).
  var INTERVALS = { 1: 0, 2: 1, 3: 3, 4: 7, 5: 16 };
  var MAX_BOX = 5;
  var MS_PER_DAY = 86400000;

  function dayIndex(nowMs) {
    return Math.floor(nowMs / MS_PER_DAY);
  }

  function makeStorage(injected) {
    if (injected) return injected;
    try {
      var t = "__rb_test__";
      global.localStorage.setItem(t, "1");
      global.localStorage.removeItem(t);
      return global.localStorage;
    } catch (e) {
      // file:// or privacy mode — degrade to a session-only in-memory store.
      var mem = {};
      return {
        getItem: function (k) { return k in mem ? mem[k] : null; },
        setItem: function (k, v) { mem[k] = String(v); }
      };
    }
  }

  function loadState(storage, key) {
    try {
      var raw = storage.getItem(key);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function saveState(storage, key, state) {
    try { storage.setItem(key, JSON.stringify(state)); } catch (e) { /* no-op */ }
  }

  function shuffleArray(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  // Choose `count` items from earlier lessons, prioritising due items, then interleaving
  // so no two adjacent questions share a concept or quarter where avoidable.
  function selectItems(pool, state, today, upTo, count) {
    var eligible = pool.filter(function (q) { return q.lesson < upTo; });

    eligible.forEach(function (q) {
      var s = state[q.id] || { box: 1, due: 0, seen: 0 };
      q._box = s.box;
      q._due = s.due;
      q._seen = s.seen || 0;
      q._overdue = today - s.due; // >= 0 means due
    });

    // Due first (most overdue, then lowest box = least mastered, then least seen),
    // then not-yet-due as filler (soonest due first).
    var due = eligible.filter(function (q) { return q._overdue >= 0; });
    var notDue = eligible.filter(function (q) { return q._overdue < 0; });
    due.sort(function (a, b) {
      return (b._overdue - a._overdue) || (a._box - b._box) || (a._seen - b._seen) || (a.lesson - b.lesson);
    });
    notDue.sort(function (a, b) { return a._due - b._due; });

    var ranked = due.concat(notDue);
    return interleave(ranked, count);
  }

  // Greedily pick `count`, preferring the next candidate that differs in concept and
  // quarter from the one just picked (interleaving for discrimination).
  function interleave(ranked, count) {
    var picked = [];
    var pool = ranked.slice();
    while (picked.length < count && pool.length) {
      var last = picked[picked.length - 1];
      var idx = 0;
      if (last) {
        for (var i = 0; i < pool.length; i++) {
          if (pool[i].concept !== last.concept && pool[i].quarter !== last.quarter) { idx = i; break; }
          if (pool[i].concept !== last.concept) { idx = i; } // fallback: at least differ concept
        }
      }
      picked.push(pool.splice(idx, 1)[0]);
    }
    return picked;
  }

  function renderItem(container, item, onAnswer, shuffle) {
    var block = document.createElement("div");
    block.className = "quiz rb-item";

    var meta = document.createElement("div");
    meta.className = "rb-meta";
    meta.textContent = "From Lesson " + String(item.lesson).padStart(3, "0") +
      " · " + item.quarter + (item.misconception ? " · watch-point" : "");
    block.appendChild(meta);

    var qEl = document.createElement("div");
    qEl.className = "quiz-question";
    qEl.textContent = item.question;
    block.appendChild(qEl);

    var optsEl = document.createElement("div");
    optsEl.className = "quiz-options";
    block.appendChild(optsEl);

    var feedbackEl = document.createElement("div");
    feedbackEl.className = "quiz-feedback";
    block.appendChild(feedbackEl);

    var options = shuffle !== false ? shuffleArray(item.options) : item.options.slice();
    var answered = false;
    var byValue = {};

    options.forEach(function (opt) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "quiz-option";
      btn.textContent = opt.label;
      btn.dataset.value = opt.value;
      byValue[opt.value] = btn;
      btn.addEventListener("click", function () {
        if (answered) return;
        answered = true;
        var buttons = optsEl.querySelectorAll(".quiz-option");
        buttons.forEach(function (b) { b.disabled = true; });
        var correct = opt.value === item.correct;
        if (correct) {
          btn.classList.add("correct");
          feedbackEl.textContent = "Correct. " + item.explain;
          feedbackEl.className = "quiz-feedback correct";
        } else {
          btn.classList.add("wrong");
          if (byValue[item.correct]) byValue[item.correct].classList.add("correct");
          feedbackEl.textContent = "Not quite. " + item.explain;
          feedbackEl.className = "quiz-feedback wrong";
        }
        onAnswer(item, correct);
      });
      optsEl.appendChild(btn);
    });

    container.appendChild(block);
  }

  function mount(container, config) {
    config = config || {};
    var pool = config.pool || global.RETRIEVAL_POOL || [];
    var count = config.count || 3;
    var upTo = config.upTo || 9999;
    var key = config.storageKey || "rdl-retrieval";
    var nowMs = config.now || Date.now();
    var today = dayIndex(nowMs);
    var storage = makeStorage(config.storage);
    var state = loadState(storage, key);

    container.innerHTML = "";
    container.classList.add("retrieval-bank");

    var head = document.createElement("div");
    head.className = "rb-head";
    head.innerHTML = "<strong>Warm-up — retrieval practice.</strong> A few questions from earlier " +
      "lessons, before today's new material. Answer from memory: effortful recall (even a wrong guess) " +
      "builds far more durable memory than re-reading.";
    container.appendChild(head);

    var items = selectItems(pool, state, today, upTo, count);

    if (!items.length) {
      var empty = document.createElement("p");
      empty.className = "rb-empty";
      empty.textContent = "No earlier lessons to recall yet — this warm-up fills in as the course grows.";
      container.appendChild(empty);
      return { items: [] };
    }

    var answeredCount = 0;
    var correctCount = 0;
    var summary = document.createElement("div");
    summary.className = "rb-summary";

    function onAnswer(item, wasCorrect) {
      var s = state[item.id] || { box: 1, due: 0, seen: 0 };
      s.seen = (s.seen || 0) + 1;
      s.box = wasCorrect ? Math.min((s.box || 1) + 1, MAX_BOX) : 1;
      s.due = today + INTERVALS[s.box];
      s.last = today;
      state[item.id] = s;
      saveState(storage, key, state);

      answeredCount += 1;
      if (wasCorrect) correctCount += 1;
      if (answeredCount === items.length) {
        summary.textContent = "Recalled " + correctCount + " / " + items.length +
          ". Missed ones return sooner; solid ones return later.";
        summary.classList.add(correctCount === items.length ? "correct" : "partial");
      }
    }

    items.forEach(function (item) { renderItem(container, item, onAnswer, config.shuffle); });
    container.appendChild(summary);

    return { items: items }; // returned for tests
  }

  global.RetrievalBank = { mount: mount, _selectItems: selectItems, _dayIndex: dayIndex };
})(window);
