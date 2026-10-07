/* Enhance static pictorial stories and figures; no network or dependencies. */
(() => {
  'use strict';
  document.querySelectorAll('.visual-story').forEach(story => {
    const stages = [...story.querySelectorAll('[data-visual-step]')];
    const controls = document.createElement('div'); controls.className = 'vs-controls';
    controls.setAttribute('role', 'group'); controls.setAttribute('aria-label', 'Highlight a stage of the visual explanation');
    const live = document.createElement('p'); live.className = 'vs-live'; live.setAttribute('aria-live', 'polite');
    ['Whole picture', ...stages.map((s, i) => `${i + 1}. ${s.querySelector('h3').textContent.replace(/^\d+\s*/, '').trim()}`)].forEach((label, index) => {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = label;
      button.setAttribute('aria-pressed', String(index === 0));
      button.addEventListener('click', () => {
        stages.forEach((s, i) => s.classList.toggle('vs-active', i === index - 1));
        controls.querySelectorAll('button').forEach((b, i) => b.setAttribute('aria-pressed', String(i === index)));
        live.textContent = index ? stages[index - 1].querySelector('.vs-explanation').textContent : 'All three stages shown together.';
      });
      controls.append(button);
    });
    story.querySelector('header').after(controls); story.append(live);
  });
  // Older mounted widgets predate figure controls. Preserve their controls and
  // give the rendered SVG a focusable inspection region and the same viewer.
  document.querySelectorAll('.early-course svg').forEach((svg, i) => {
    if (svg.closest('figure,.course-visual,.visual-story')) return;
    const figure = document.createElement('figure'); figure.className = 'legacy-visual';
    figure.id = 'legacy-visual-' + i; figure.tabIndex = 0;
    figure.setAttribute('aria-label', 'Interactive diagram; scroll to inspect wide details');
    svg.before(figure); figure.append(svg);
    const tools = document.createElement('div'); tools.className = 'vs-figure-tools';
    const link = document.createElement('a'); link.href = '#' + figure.id;
    link.dataset.enlarge = ''; link.dataset.figureSvg = ''; link.textContent = 'Inspect diagram';
    tools.append(link); figure.append(tools);
  });
  let dialog, opener, objectURL;
  function openFigure(link, trigger) {
    opener = trigger;
    if (!dialog) {
      dialog = document.createElement('dialog'); dialog.className = 'vs-dialog'; dialog.setAttribute('aria-label', 'Enlarged lesson figure');
      const bar = document.createElement('div'); bar.className = 'vs-dialog-bar';
      const title = document.createElement('span'); title.textContent = 'Full-size figure · scroll horizontally or vertically to inspect details';
      const close = document.createElement('button'); close.type = 'button'; close.textContent = 'Close'; close.addEventListener('click', () => dialog.close());
      const fit = document.createElement('button'); fit.type = 'button'; fit.textContent = 'Fit whole figure';
      fit.setAttribute('aria-pressed', 'false');
      fit.addEventListener('click', () => { const fitted = dialog.classList.toggle('cv-fit'); fit.setAttribute('aria-pressed', String(fitted)); fit.textContent = fitted ? 'Read full-size details' : 'Fit whole figure'; });
      bar.append(title, fit, close); dialog.append(bar);
      const canvas = document.createElement('div'); canvas.className = 'vs-dialog-canvas'; canvas.tabIndex = 0;
      canvas.setAttribute('role', 'region'); canvas.setAttribute('aria-label', 'Scrollable enlarged figure'); dialog.append(canvas);
      dialog.addEventListener('close', () => { if (objectURL) URL.revokeObjectURL(objectURL); objectURL = null; if (opener?.isConnected) opener.focus(); });
      document.body.append(dialog);
    }
    const canvas = dialog.querySelector('.vs-dialog-canvas'); canvas.replaceChildren();
    const figure = link.closest('figure'); const image = document.createElement('img');
    if (link.hasAttribute('data-figure-svg')) {
      const svg = figure.querySelector('svg').cloneNode(true);
      // Inline diagrams can rely on inherited styles; include their computed styles.
      const originalNodes = [figure.querySelector('svg'), ...figure.querySelector('svg').querySelectorAll('*')];
      [svg, ...svg.querySelectorAll('*')].forEach((node, i) => {
        const style = getComputedStyle(originalNodes[i]);
        for (const name of ['font-family','font-size','font-weight','fill','stroke','stroke-width','text-anchor']) node.style.setProperty(name, style.getPropertyValue(name));
      });
      svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      objectURL = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type:'image/svg+xml'})); image.src = objectURL;
    } else image.src = link.href;
    image.alt = figure?.querySelector('img')?.alt || figure?.querySelector('figcaption')?.textContent || 'Enlarged architecture diagram';
    dialog.classList.remove('cv-fit'); const fit = dialog.querySelector('[aria-pressed]'); fit.setAttribute('aria-pressed','false'); fit.textContent = 'Fit whole figure';
    canvas.append(image); dialog.showModal(); canvas.scrollLeft = 0; canvas.scrollTop = 0;
  }
  document.querySelectorAll('.vs-figure-tools a[data-enlarge]').forEach(link => {
    const button = document.createElement('button'); button.type = 'button'; button.textContent = 'Enlarge figure';
    button.addEventListener('click', () => openFigure(link, button)); link.after(button);
  });
  // Keep the retrieval answer available on paper without changing saved state.
  const printState = new Map();
  window.addEventListener('beforeprint', () => document.querySelectorAll('.vs-predict').forEach(d => { printState.set(d, d.open); d.open = true; }));
  window.addEventListener('afterprint', () => { printState.forEach((open, d) => { d.open = open; }); printState.clear(); });
})();
