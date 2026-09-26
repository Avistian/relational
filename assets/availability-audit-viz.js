/* Expected states: query=8; event=3. Arrival 9: BLOCK; arrival 7: ALLOW.
 * The baseline remains arrival=9. Native controls support keyboard interaction.
 */
(function(global){'use strict';
 function mount(container){
  container.className='audit-clock';
  container.innerHTML='<label>Correction arrival day: <output>9</output><input aria-label="Correction arrival day" type="range" min="3" max="12" step="1" value="9"></label><div class="audit-comparison"><section><h3>Baseline · arrival day 9</h3><p>Event day 3 &lt; query day 8 ✓<br>Arrival day 9 ≤ query day 8 ✗</p><p class="audit-status" data-legal="false">BLOCK · unavailable at prediction</p></section><section><h3>Changed arrival</h3><p class="audit-trace"></p><p class="audit-status audit-live" aria-live="polite"></p></section></div><p class="audit-explain">Only arrival time changes. The event date and prediction time stay fixed.</p><button type="button">Reset to day 9</button>';
  const slider=container.querySelector('input'),out=container.querySelector('output'),trace=container.querySelector('.audit-trace'),status=container.querySelector('.audit-live');
  function update(){const day=Number(slider.value),legal=day<=8;out.textContent=day;trace.innerHTML='Event day 3 &lt; query day 8 ✓<br>Arrival day '+day+' ≤ query day 8 '+(legal?'✓':'✗');status.dataset.legal=String(legal);status.textContent=legal?'ALLOW · available at prediction':'BLOCK · unavailable at prediction';}
  slider.addEventListener('input',update);container.querySelector('button').addEventListener('click',()=>{slider.value=9;update();});update();return {setArrival(day){slider.value=Math.max(3,Math.min(12,day));update();}};
 }
 global.AvailabilityAuditViz={mount};
})(window);
