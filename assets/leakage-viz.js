/**
 * Point-in-time join / leakage timeline visualizer.
 * Shows orders on a timeline relative to prediction cutoff t; toggling features
 * highlights safe (past-only) vs leaking (future) windows.
 * Usage: LeakageViz.mount(containerElement)
 * Expected states documented for lesson verification (L002).
 */
(function (global) {
  "use strict";

  var FEATURES = {
    B: {
      label: "B — spend_30d",
      safe: true,
      caption: "Safe: sums orders in (t−30d, t] — only past behavior."
    },
    C: {
      label: "C — future_orders",
      safe: false,
      caption: "Leak: counts orders strictly after t — future information."
    },
    F: {
      label: "F — avg_order_after_t",
      safe: false,
      caption: "Leak: averages amounts after t — direct future peek."
    },
    E: {
      label: "E — days_since_last_order",
      safe: true,
      caption: "Safe: MAX(order_date) restricted to orders ≤ t."
    }
  };

  var ORDERS = [
    { day: 10, amount: 40 },
    { day: 25, amount: 55 },
    { day: 42, amount: 30 },
    { day: 58, amount: 70 },
    { day: 75, amount: 45 },
    { day: 88, amount: 60 }
  ];

  var CUTOFF = 50;

  function mount(container) {
    container.innerHTML = "";
    container.className = "leak-viz";

    var intro = document.createElement("p");
    intro.className = "leak-viz-intro";
    intro.textContent =
      "Prediction time t = day " + CUTOFF + ". Label uses churn in the 30 days after t. " +
      "Toggle a feature to see which orders it uses.";
    container.appendChild(intro);

    var controls = document.createElement("div");
    controls.className = "leak-viz-controls";
    container.appendChild(controls);

    var timeline = document.createElement("div");
    timeline.className = "leak-viz-timeline";
    container.appendChild(timeline);

    var axis = document.createElement("div");
    axis.className = "leak-viz-axis";
    axis.innerHTML =
      "<span>day 0</span><span class=\"leak-viz-cutoff\">t = " + CUTOFF + "</span><span>day 100</span>";
    timeline.appendChild(axis);

    var track = document.createElement("div");
    track.className = "leak-viz-track";
    timeline.appendChild(track);

    var caption = document.createElement("p");
    caption.className = "leak-viz-caption";
    container.appendChild(caption);

    var active = "B";

    function renderTrack(featKey) {
      track.innerHTML = "";
      ORDERS.forEach(function (o) {
        var dot = document.createElement("div");
        dot.className = "leak-viz-order";
        dot.style.left = o.day + "%";
        dot.title = "day " + o.day + ", $" + o.amount;

        var inPast = o.day <= CUTOFF;
        var inFuture = o.day > CUTOFF;
        var used = false;

        if (featKey === "B") {
          used = o.day > CUTOFF - 30 && o.day <= CUTOFF;
        } else if (featKey === "C" || featKey === "F") {
          used = inFuture;
        } else if (featKey === "E") {
          used = inPast;
        }

        if (used) dot.classList.add("leak-viz-used");
        if (inFuture) dot.classList.add("leak-viz-future");
        track.appendChild(dot);
      });

      var cut = document.createElement("div");
      cut.className = "leak-viz-line";
      cut.style.left = CUTOFF + "%";
      track.appendChild(cut);

      var f = FEATURES[featKey];
      caption.textContent = f.caption;
      caption.className = "leak-viz-caption " + (f.safe ? "leak-viz-safe" : "leak-viz-leak");
    }

    Object.keys(FEATURES).forEach(function (key) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "leak-viz-btn" + (key === active ? " active" : "");
      btn.textContent = FEATURES[key].label;
      btn.addEventListener("click", function () {
        active = key;
        controls.querySelectorAll(".leak-viz-btn").forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        renderTrack(key);
      });
      controls.appendChild(btn);
    });

    renderTrack(active);
  }

  global.LeakageViz = { mount: mount };
})(window);
