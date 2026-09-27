(() => {
  const host=document.getElementById('l120-cutoff'); if(!host)return;
  host.innerHTML='<label for="l120-time">Prediction day</label> <input id="l120-time" type="range" min="3" max="12" value="7"> <button type="button">Reset cutoff</button><p><output aria-live="polite"></output></p>';
  const input=host.querySelector('input'),out=host.querySelector('output');
  const compute=t=>[1,2,5].filter((id,i)=>[3,5,11][i]<=t && [3,8,11][i]<=t);
  function update(){const t=+input.value,ids=compute(t);out.textContent=`Day ${t}: person90 can use event rows ${ids.join(', ')}. ${t<8?'Event2 is still unavailable.':'Event2 has arrived.'} ${t<11?'Event5 is in the future.':'Event5 is now visible.'}`;}
  input.addEventListener('input',update);host.querySelector('button').addEventListener('click',()=>{input.value=7;update();});window.L120Cutoff={compute};update();
})();
