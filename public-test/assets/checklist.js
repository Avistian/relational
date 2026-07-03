/**
 * Reusable interactive checklist widget — for checkpoint/capstone lessons.
 *
 * Renders a titled list of checkboxes (e.g. the leakage-spine rubric). When every
 * item is ticked, a success message appears: a tight feedback loop that turns a
 * passive list into an active self-audit. Reusable across the Q1–Q4 checkpoints.
 *
 * Usage:
 *   Checklist.mount(container, {
 *     title: "Leakage-spine rubric",
 *     items: [{ label: "...", hint: "Lesson 005" }, ...],
 *     done: "All boxes ticked — your baseline is checkpoint-ready."
 *   });
 *
 * Companion CSS lives in the lesson <style> block (class prefix .chk).
 */
(function (global) {
  "use strict";

  function mount(container, config) {
    var items = config.items || [];
    var doneMsg = config.done || "All items complete.";

    container.innerHTML = "";
    container.classList.add("chk");

    if (config.title) {
      var h = document.createElement("div");
      h.className = "chk-title";
      h.textContent = config.title;
      container.appendChild(h);
    }

    var list = document.createElement("ul");
    list.className = "chk-list";
    container.appendChild(list);

    var feedback = document.createElement("div");
    feedback.className = "chk-feedback";
    container.appendChild(feedback);

    var boxes = [];

    items.forEach(function (item, i) {
      var li = document.createElement("li");
      li.className = "chk-item";

      var id = "chk-" + Math.random().toString(36).slice(2) + "-" + i;
      var box = document.createElement("input");
      box.type = "checkbox";
      box.id = id;
      box.addEventListener("change", update);
      boxes.push(box);

      var lbl = document.createElement("label");
      lbl.setAttribute("for", id);
      lbl.innerHTML = item.label + (item.hint ? ' <span class="chk-hint">' + item.hint + "</span>" : "");

      li.appendChild(box);
      li.appendChild(lbl);
      list.appendChild(li);
    });

    function update() {
      var done = boxes.every(function (b) { return b.checked; });
      var count = boxes.filter(function (b) { return b.checked; }).length;
      if (done) {
        feedback.textContent = doneMsg;
        feedback.className = "chk-feedback done";
      } else {
        feedback.textContent = count + " / " + boxes.length + " ticked";
        feedback.className = "chk-feedback";
      }
    }

    update();
  }

  global.Checklist = { mount: mount };
})(window);
