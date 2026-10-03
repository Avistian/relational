/* Practice workflow only: never records or sends external participation. */
(() => {
  const host = document.getElementById('community-feedback');
  if (!host) return;
  host.className = 'community-widget';
  host.innerHTML = '<h3>Practice: what can you claim now?</h3><div class="controls">' +
    '<label>Actual thread exists<select id="community-post"><option value="no">No thread yet</option><option value="yes">Thread URL recorded</option></select></label>' +
    '<label>Actual response exists<select id="community-reply"><option value="no">No response yet</option><option value="yes">Response recorded</option></select></label>' +
    '<label>Response has a documented check<select id="community-check"><option value="no">Check still pending</option><option value="yes">Check documented</option></select></label></div>' +
    '<output aria-live="polite"></output><p class="community-note"></p><button type="button">Reset practice</button>';
  const fields = [...host.querySelectorAll('select')];
  function update() {
    const [posted, reply, verified] = fields.map(x => x.value === 'yes');
    const state = !posted ? 'DRAFT_ONLY' : !reply ? 'AWAITING_RESPONSE' : !verified ? 'RESPONSE_UNVERIFIED' : 'FEEDBACK_CHECKED';
    host.dataset.state = state;
    host.querySelector('output').textContent = state;
    const descriptions = {
      DRAFT_ONLY: 'A prepared question is not a community conversation. A response or check selected here cannot supply a missing thread.',
      AWAITING_RESPONSE: 'Keep the question open. Silence supplies no evidence for or against the claim.',
      RESPONSE_UNVERIFIED: 'Translate the actual reply into a check; a confident answer is not a reproduced result.',
      FEEDBACK_CHECKED: 'Record what the check supports or contradicts. This state alone does not establish historical paper validity or learner mastery.'
    };
    host.querySelector('.community-note').textContent = descriptions[state] + ' Practice only; the actual record remains DRAFT_ONLY.';
  }
  fields.forEach(x => x.addEventListener('change', update));
  host.querySelector('button').addEventListener('click', () => { fields.forEach(x => x.value = 'no'); update(); });
  update();
})();
