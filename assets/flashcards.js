/**
 * Spaced flashcard review (self-rated recall) for the core-paper deck.
 *
 * Front = a recall prompt; the learner recalls the claim FROM MEMORY, then reveals the back
 * and self-rates "Got it" / "Missed". Leitner-scheduled in localStorage (same intervals as the
 * retrieval bank): a missed card returns next session, a solid one much later.
 *
 * Self-rating is the honest signal for a free-recall card (there is no machine grader). The value
 * is the effortful recall before revealing — not the button.
 *
 * Usage (e.g. flashcards.html):
 *   <script src="../assets/paper-deck.js"></script>   (or "assets/…" from root)
 *   <script src="../assets/flashcards.js"></script>
 *   Flashcards.mount(document.getElementById("deck"), { deck: window.PAPER_DECK });
 *
 * Config:
 *   deck        array of cards (default window.PAPER_DECK)
 *   upTo        if set, only cards with lesson < upTo are eligible (spacing in a lesson context)
 *   storageKey  localStorage key (default "rdl-flashcards")
 *   now, storage  test overrides (see retrieval-bank.js)
 */
(function (global) {
  "use strict";

  var INTERVALS = { 1: 0, 2: 1, 3: 3, 4: 7, 5: 16 };
  var MAX_BOX = 5;
  var MS_PER_DAY = 86400000;

  function dayIndex(nowMs) { return Math.floor(nowMs / MS_PER_DAY); }

  function makeStorage(injected) {
    if (injected) return injected;
    try {
      var t = "__fc_test__";
      global.localStorage.setItem(t, "1");
      global.localStorage.removeItem(t);
      return global.localStorage;
    } catch (e) {
      var mem = {};
      return { getItem: function (k) { return k in mem ? mem[k] : null; }, setItem: function (k, v) { mem[k] = String(v); } };
    }
  }
  function loadState(s, k) { try { var r = s.getItem(k); return r ? JSON.parse(r) : {}; } catch (e) { return {}; } }
  function saveState(s, k, st) { try { s.setItem(k, JSON.stringify(st)); } catch (e) { /* no-op */ } }

  function buildQueue(deck, state, today, upTo) {
    var eligible = deck.filter(function (c) { return upTo ? c.lesson < upTo : true; });
    eligible.forEach(function (c) {
      var s = state[c.id] || { box: 1, due: 0, seen: 0 };
      c._box = s.box; c._due = s.due; c._overdue = today - s.due;
    });
    var due = eligible.filter(function (c) { return c._overdue >= 0; });
    due.sort(function (a, b) { return (b._overdue - a._overdue) || (a._box - b._box) || (a.lesson - b.lesson); });
    return { due: due, all: eligible };
  }

  function mount(container, config) {
    config = config || {};
    var deck = config.deck || global.PAPER_DECK || [];
    var key = config.storageKey || "rdl-flashcards";
    var nowMs = config.now || Date.now();
    var today = dayIndex(nowMs);
    var storage = makeStorage(config.storage);
    var state = loadState(storage, key);

    container.innerHTML = "";
    container.classList.add("flashcards");

    var q = buildQueue(deck, state, today, config.upTo);
    var queue = q.due.slice();
    var reviewingAll = false;

    var head = document.createElement("div");
    head.className = "fc-head";
    container.appendChild(head);
    var cardEl = document.createElement("div");
    cardEl.className = "fc-card";
    container.appendChild(cardEl);

    var reviewed = 0, got = 0;

    function renderDone() {
      head.textContent = "";
      cardEl.className = "fc-card fc-done";
      cardEl.innerHTML = "";
      var msg = document.createElement("p");
      msg.className = "fc-donemsg";
      msg.textContent = reviewed
        ? ("Reviewed " + reviewed + " card" + (reviewed === 1 ? "" : "s") + " · recalled " + got + ". Come back tomorrow — the schedule spaces the rest.")
        : "All caught up — no cards are due right now.";
      cardEl.appendChild(msg);
      if (!reviewingAll && q.all.length) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "fc-btn fc-review-all";
        btn.textContent = "Review all anyway";
        btn.addEventListener("click", function () {
          reviewingAll = true;
          queue = q.all.slice().sort(function (a, b) { return a.lesson - b.lesson; });
          next();
        });
        cardEl.appendChild(btn);
      }
    }

    function rate(card, gotIt) {
      var s = state[card.id] || { box: 1, due: 0, seen: 0 };
      s.seen = (s.seen || 0) + 1;
      s.box = gotIt ? Math.min((s.box || 1) + 1, MAX_BOX) : 1;
      s.due = today + INTERVALS[s.box];
      s.last = today;
      state[card.id] = s;
      saveState(storage, key, state);
      reviewed += 1; if (gotIt) got += 1;
      next();
    }

    function renderCard(card) {
      head.textContent = (queue.length + 1) + " to review" + (reviewingAll ? " (all)" : " (due)");
      cardEl.className = "fc-card";
      cardEl.innerHTML = "";

      var meta = document.createElement("div");
      meta.className = "fc-meta";
      meta.textContent = card.paper + " (" + card.year + ") · Lesson " + String(card.lesson).padStart(3, "0");
      cardEl.appendChild(meta);

      var front = document.createElement("div");
      front.className = "fc-front";
      front.textContent = card.front;
      cardEl.appendChild(front);

      var back = document.createElement("div");
      back.className = "fc-back";
      back.style.display = "none";
      back.textContent = card.back;
      cardEl.appendChild(back);

      var show = document.createElement("button");
      show.type = "button";
      show.className = "fc-btn fc-show";
      show.textContent = "Show answer";
      cardEl.appendChild(show);

      var rateRow = document.createElement("div");
      rateRow.className = "fc-rate";
      rateRow.style.display = "none";
      var missed = document.createElement("button");
      missed.type = "button"; missed.className = "fc-btn fc-missed"; missed.textContent = "Missed it";
      var gotBtn = document.createElement("button");
      gotBtn.type = "button"; gotBtn.className = "fc-btn fc-got"; gotBtn.textContent = "Got it";
      rateRow.appendChild(missed); rateRow.appendChild(gotBtn);
      cardEl.appendChild(rateRow);

      show.addEventListener("click", function () {
        back.style.display = "";
        show.style.display = "none";
        rateRow.style.display = "";
      });
      missed.addEventListener("click", function () { rate(card, false); });
      gotBtn.addEventListener("click", function () { rate(card, true); });
    }

    function next() {
      if (!queue.length) { renderDone(); return; }
      renderCard(queue.shift());
    }

    next();
    return { dueCount: q.due.length, _state: state };
  }

  global.Flashcards = { mount: mount };
})(window);
