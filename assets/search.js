/* Lesson search: pure search(index, query), usable from node and the browser.
   Rank: title > heading > glossary term. Case/diacritic-insensitive; multi-word AND
   (every word must match somewhere in the entry). */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LessonSearch = api;
})(typeof self !== "undefined" ? self : this, function () {
  function fold(s) {
    return String(s == null ? "" : s).normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  }

  function search(index, query) {
    var words = fold(query).split(/\s+/).filter(Boolean);
    if (!words.length || !Array.isArray(index)) return [];
    var phrase = words.join(" ");
    var scored = [];
    index.forEach(function (entry, pos) {
      var title = fold(entry.title);
      var heads = (entry.headings || []).map(fold);
      var terms = (entry.glossary || []).map(fold);
      var score = 0;
      for (var i = 0; i < words.length; i++) {
        var w = words[i], s = 0;
        if (title.indexOf(w) >= 0) s = 100;
        else if (heads.some(function (h) { return h.indexOf(w) >= 0; })) s = 10;
        else if (terms.some(function (t) { return t.indexOf(w) >= 0; })) s = 1;
        if (!s) return;
        score += s;
      }
      if (title.indexOf(phrase) >= 0) score += 50;
      else if (heads.some(function (h) { return h.indexOf(phrase) >= 0; })) score += 5;
      else if (terms.some(function (t) { return t.indexOf(phrase) >= 0; })) score += 1;
      scored.push({ entry: entry, score: score, pos: pos });
    });
    scored.sort(function (a, b) { return b.score - a.score || a.pos - b.pos; });
    return scored.map(function (r) { return r.entry; });
  }

  return { search: search, fold: fold };
});
