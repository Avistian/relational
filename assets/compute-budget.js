/* Reusable serial-session scenario calculator. Does not launch work. */
(function(g){'use strict';
function assess(cost,wall,active,required,available,session,scientific,stop){
 const total=active===null||wall===null?null:active+wall;
 const missing=[['cost',cost],['wall_seconds',wall],['active_seconds',active],['required_gib',required]].filter(x=>x[1]===null).map(x=>x[0]);
 const status=!scientific?'SCIENTIFIC_STOP':cost!==null&&cost>stop?'OVER_BUDGET':required!==null&&required>available?'MEMORY_LIMIT':total!==null&&total>session?'EXCEEDS_SESSION':missing.length?'INCOMPLETE_MEASUREMENT':'FEASIBLE_SCENARIO';
 return {status,missing,serial_session_seconds:total};
}
function mount(el,evidence){
 el.innerHTML='<div class="cb-controls"><label class="cb-wide">Evidence or planning case<select data-case aria-label="Planning case"><option value="forecast">L176 · original conservative forecast</option><option value="observed">L176 · historical worker time</option><option value="pretrain">RT · reported pretraining scenario</option><option value="finetune">RT · reported fine-tuning scenario</option><option value="stopped">L175 · failed temporal gate</option></select></label><label>Session<select data-session aria-label="Session"><option value="60">Baseline · 1 hour</option><option value="180">Extended · 3 hours</option></select></label><label>Active work assumption<select data-active aria-label="Active work assumption"><option value="unknown">Unknown / not measured</option><option value="1200">Assume 20 minutes</option><option value="3600">Assume 60 minutes</option></select></label><label>Required GPU memory assumption<select data-memory aria-label="GPU memory assumption"><option value="unknown">Unknown / no full measure</option><option value="16">Assume 16 GiB</option><option value="32">Assume 32 GiB</option></select></label><label>Available GPU memory<select data-device aria-label="Available GPU memory"><option value="24">24 GiB</option><option value="8">8 GiB</option></select></label><button type="button" data-reset>Reset assumptions</button></div><output class="cb-result" aria-live="polite"></output><div class="cb-meters"><label>Planned cost against $8 stop<meter data-cost-meter min="0" max="8"></meter></label><label>Serial time against selected session<meter data-time-meter min="0" max="3600"></meter></label></div><p class="cb-assumption">Assumptions are inputs, not new measurements. $2 of the $10 cap stays protected. This calculator uses a serial schedule; it does not overlap active work with machine time.</p>';
 const find=s=>el.querySelector(s),out=find('output');
 function update(){
 const key=find('[data-case]').value,minutes=Number(find('[data-session]').value),active=find('[data-active]').value==='unknown'?null:Number(find('[data-active]').value),memory=find('[data-memory]').value==='unknown'?null:Number(find('[data-memory]').value),device=Number(find('[data-device]').value);
 let cost,wall,scientific=true,note;
 if(key==='forecast'||key==='observed'){cost=evidence.l176.reserved_usd;wall=(key==='forecast'?evidence.l176.forecast_seconds:evidence.l176.worker_seconds)+120;note='L176 reservation; add 120 s of assumed startup. Recorded allocator peak is '+evidence.l176.max_allocator_gib.toFixed(3)+' GiB; total required memory is unknown.';}
 else if(key==='stopped'){cost=evidence.l175.reserved_usd;wall=930;scientific=false;note='Historical audit reservation. Temporal validity failed; GPU inference was NOT_RUN. Time shown is an assumed 930 s audit allowance.';}
 else {const row=evidence.rt_price_scenarios[key==='pretrain'?'pretraining':'fine_tuning'];cost=row.gpu_only_usd_40gb;wall=row.reported_elapsed_hours*3600;note='Eight concurrent A100s at the 40 GB base price: GPU-only hypothetical cost. Excludes CPU, RAM and overhead; allocation not authorized.';}
 const a=assess(cost,wall,active,memory,device,minutes*60,scientific,8);
 // A measured tensor allocation already rules out this smaller device for the unchanged workload.
 if((key==='forecast'||key==='observed')&&device<evidence.l176.max_allocator_gib&&a.status!=='OVER_BUDGET')a.status='MEMORY_LIMIT';
 el.dataset.state=[key,minutes,active===null?'unknown':active,memory===null?'unknown':memory,device].join('/');out.dataset.verdict=a.status;
 out.innerHTML='<strong>'+a.status+'</strong>Cost basis: $'+cost.toFixed(6)+' / $8 planned stop.<br>Machine time: '+(wall/60).toFixed(2)+' min; active work: '+(active===null?'unknown':(active/60)+' min')+'; serial total: '+(a.serial_session_seconds===null?'unknown':(a.serial_session_seconds/60).toFixed(2)+' min')+'.<br>Missing inputs: '+(a.missing.join(', ')||'none in this hypothetical scenario')+'.<br>'+note;
 find('[data-cost-meter]').value=Math.min(cost,8);find('[data-time-meter]').max=minutes*60;find('[data-time-meter]').value=a.serial_session_seconds===null?0:Math.min(a.serial_session_seconds,minutes*60);
 find('[data-time-meter]').setAttribute('aria-label',a.serial_session_seconds===null?'Time unknown':a.serial_session_seconds+' seconds');
 }
 el.querySelectorAll('select').forEach(x=>x.addEventListener('change',update));find('[data-reset]').addEventListener('click',()=>{find('[data-case]').value='forecast';find('[data-session]').value='60';find('[data-active]').value='unknown';find('[data-memory]').value='unknown';find('[data-device]').value='24';update();});update();
}
g.ComputeBudget={mount,assess};
})(window);
