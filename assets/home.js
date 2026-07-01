/**
 * Home page lesson navigator — reads lessons/manifest.json and renders
 * year/quarter sections. Stores last-opened lesson in localStorage.
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "rdl-last-lesson";

  function padId(n) {
    return String(n).padStart(4, "0");
  }

  function quarterLabel(q) {
    return "Q" + q;
  }

  function renderContinue(container, lessons) {
    var lastSlug = global.localStorage.getItem(STORAGE_KEY);
    var target = null;
    if (lastSlug) {
      target = lessons.filter(function (l) {
        return l.slug === lastSlug && l.published;
      })[0];
    }
    if (!target) {
      var published = lessons.filter(function (l) { return l.published; });
      target = published[published.length - 1];
    }
    if (!target) return;

    var wrap = document.createElement("div");
    wrap.className = "continue-banner";
    var a = document.createElement("a");
    a.href = "lessons/" + target.slug + ".html";
    a.innerHTML =
      "<span class=\"continue-label\">Continue</span>" +
      "<span class=\"continue-title\">Lesson " + padId(target.id) + " · " + target.title + "</span>";
    a.addEventListener("click", function () {
      global.localStorage.setItem(STORAGE_KEY, target.slug);
    });
    wrap.appendChild(a);
    container.appendChild(wrap);
  }

  function groupLessons(lessons) {
    var byYear = {};
    lessons.forEach(function (l) {
      if (!l.published) return;
      if (!byYear[l.year]) byYear[l.year] = {};
      if (!byYear[l.year][l.quarter]) byYear[l.year][l.quarter] = [];
      byYear[l.year][l.quarter].push(l);
    });
    return byYear;
  }

  function renderYearBlock(year, quarters, container) {
    var block = document.createElement("details");
    block.className = "year-block";
    block.open = year === 1;

    var summary = document.createElement("summary");
    summary.className = "year-summary";
    summary.textContent = "Year " + year;
    block.appendChild(summary);

    var qKeys = Object.keys(quarters).sort(function (a, b) {
      return Number(a) - Number(b);
    });

    qKeys.forEach(function (q) {
      var qLessons = quarters[q];
      var qSection = document.createElement("div");
      qSection.className = "quarter-section";

      var qHead = document.createElement("h3");
      qHead.className = "quarter-head";
      var hasCheckpoint = qLessons.some(function (l) { return l.checkpoint; });
      qHead.textContent = quarterLabel(q) + (hasCheckpoint ? " · checkpoint" : "");
      qSection.appendChild(qHead);

      var ul = document.createElement("ul");
      ul.className = "lesson-list";

      qLessons.forEach(function (l) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = "lessons/" + l.slug + ".html";
        a.innerHTML =
          "<span class=\"num\">Lesson " + padId(l.id) + "</span>" +
          "<span class=\"title\">" + l.title + "</span>";
        a.addEventListener("click", function () {
          global.localStorage.setItem(STORAGE_KEY, l.slug);
        });
        li.appendChild(a);
        if (l.labPath) {
          var lab = document.createElement("span");
          lab.className = "lab-link";
          var labA = document.createElement("a");
          labA.href = l.labPath;
          labA.textContent = "lab";
          lab.appendChild(labA);
          li.appendChild(lab);
        }
        ul.appendChild(li);
      });

      qSection.appendChild(ul);
      block.appendChild(qSection);
    });

    container.appendChild(block);
  }

  function mount(config) {
    var navEl = document.getElementById(config.navId || "lesson-nav");
    var continueEl = document.getElementById(config.continueId || "continue-nav");
    if (!navEl) return;

    fetch(config.manifestUrl || "lessons/manifest.json")
      .then(function (r) {
        if (!r.ok) throw new Error("manifest fetch failed");
        return r.json();
      })
      .then(function (data) {
        var lessons = (data.lessons || []).slice().sort(function (a, b) {
          return a.id - b.id;
        });
        if (continueEl) renderContinue(continueEl, lessons);
        var grouped = groupLessons(lessons);
        var years = Object.keys(grouped).sort(function (a, b) {
          return Number(a) - Number(b);
        });
        years.forEach(function (y) {
          renderYearBlock(Number(y), grouped[y], navEl);
        });
      })
      .catch(function () {
        navEl.innerHTML =
          "<p class=\"nav-error\">Could not load lesson list. " +
          "<a href=\"reference/curriculum.html\">See full curriculum</a>.</p>";
      });
  }

  global.HomeNav = { mount: mount };
})(window);
