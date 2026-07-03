/**
 * Notebook gallery — reads lessons/manifest.json and lists every lab with:
 *   View (rendered HTML) · Source (.ipynb) · Run on Binder (real env) · Open in Colab.
 * Binder runs the true environment (relkit + OpenML fetch); Colab needs a repo clone.
 */
(function (global) {
  "use strict";

  var REPO = "Avistian/relational";
  var BRANCH = "main";

  function padId(n) { return String(n).padStart(4, "0"); }

  function htmlPathFor(labPath) {
    // labs/0012-foo.ipynb -> labs/html/0012-foo.html
    var base = labPath.replace(/^labs\//, "").replace(/\.ipynb$/, "");
    return "labs/html/" + base + ".html";
  }

  function binderUrl(labPath) {
    return "https://mybinder.org/v2/gh/" + REPO + "/" + BRANCH +
      "?labpath=" + encodeURIComponent(labPath);
  }

  function colabUrl(labPath) {
    return "https://colab.research.google.com/github/" + REPO + "/blob/" + BRANCH + "/" + labPath;
  }

  function row(lesson) {
    var li = document.createElement("li");

    var head = document.createElement("div");
    head.className = "nb-head";
    head.innerHTML =
      "<span class=\"num\">Lesson " + padId(lesson.id) + "</span>" +
      "<span class=\"title\">" + lesson.title + "</span>";
    li.appendChild(head);

    var links = document.createElement("div");
    links.className = "nb-links";

    function add(href, text, cls, newTab) {
      var a = document.createElement("a");
      a.href = href;
      a.textContent = text;
      if (cls) a.className = cls;
      if (newTab) { a.target = "_blank"; a.rel = "noopener"; }
      links.appendChild(a);
    }

    add(htmlPathFor(lesson.labPath), "View", "nb-view");
    add(lesson.labPath, "Source .ipynb", "nb-src");
    add(binderUrl(lesson.labPath), "Run on Binder", "nb-run", true);
    add(colabUrl(lesson.labPath), "Open in Colab", "nb-run", true);

    li.appendChild(links);
    return li;
  }

  function manifestFetchUrl(base) {
    base = base || "lessons/manifest.json";
    var meta = document.querySelector('meta[name="rdl-manifest-version"]');
    var v = meta && meta.getAttribute("content");
    if (!v) return base;
    return base + (base.indexOf("?") >= 0 ? "&" : "?") + "v=" + encodeURIComponent(v);
  }

  function mount(config) {
    config = config || {};
    var el = document.getElementById(config.listId || "nb-list");
    if (!el) return;

    fetch(manifestFetchUrl(config.manifestUrl))
      .then(function (r) { if (!r.ok) throw new Error("manifest"); return r.json(); })
      .then(function (data) {
        var labs = (data.lessons || [])
          .filter(function (l) { return l.published && l.labPath; })
          .sort(function (a, b) { return b.id - a.id; }); // newest first
        if (!labs.length) { el.innerHTML = "<p class=\"nav-error\">No notebooks yet.</p>"; return; }
        var ul = document.createElement("ul");
        ul.className = "nb-gallery";
        labs.forEach(function (l) { ul.appendChild(row(l)); });
        el.innerHTML = "";
        el.appendChild(ul);
      })
      .catch(function () {
        el.innerHTML = "<p class=\"nav-error\">Could not load the notebook list.</p>";
      });
  }

  global.Notebooks = { mount: mount };
})(window);
