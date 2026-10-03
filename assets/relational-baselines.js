/* Scalar illustration only: not full-model outputs or measured benchmark scores. */
document.querySelectorAll('[data-b11-trace]').forEach(board=>{
  const day=board.querySelector('[name=cutoff]'),value=board.querySelector('[name=product]'),out=board.querySelector('output');
  function render(){const second=Number(day.value)>=11,p=Number(value.value),route=second?(3+p+3)/2:3,token=second?(0+1+3+2+p)/5:1;
    board.querySelector('[data-day]').textContent=day.value;board.querySelector('[data-product]').textContent=value.value;
    out.dataset.route=route;out.dataset.token=token;out.dataset.count=second?2:1;
    out.textContent=`Day ${day.value}: ${second?'B0 and B1 admitted':'Only B0 admitted; B1 has not arrived'}. RelGNN scalar route update = ${route.toFixed(2)}. RelGT scalar target attention update = ${token.toFixed(2)}. ${second?'Changing P1 now affects both updates.':'Changing P1 cannot affect either admitted context.'}`;
  }
  day.addEventListener('input',render);value.addEventListener('input',render);board.querySelector('button').addEventListener('click',()=>{day.value=11;value.value=6;render()});render();
});
