/* Worked examples only. Defaults: A=[2,4], B=[3,2] => B; twelve workers=$2.437776.
 * Change A's second seed to 1 => A; twenty workers with $3 committed => refuse.
 * Native controls, explicit baseline, reset, and plain text readouts support keyboard use. */
(() => {
  const selection = document.getElementById('l135-selection');
  if (selection) {
    selection.innerHTML = `<p><strong>Predict:</strong> can one excellent seed rescue an unstable configuration?</p><label for="l135-score">A, second seed MAE <output id="l135-score-value">4.0</output></label><input id="l135-score" type="range" min="1" max="6" step="0.5" value="4"><p>Fixed: A seed 1 = 2; B seeds = [3, 2]. Baseline winner: B.</p><p id="l135-selection-result" aria-live="polite"></p><button type="button" id="l135-selection-reset">Reset selection</button>`;
    const input=selection.querySelector('input');
    const update=()=>{const value=Number(input.value),mean=(2+value)/2;selection.querySelector('output').textContent=value.toFixed(1);selection.querySelector('#l135-selection-result').textContent=`A mean = ${mean.toFixed(2)}; B mean = 2.50. Winner: ${mean<=2.5?'A':'B'}${mean===2.5?' (exact tie, ID order)':''}. The best single score does not select the configuration.`;};
    input.addEventListener('input',update);selection.querySelector('button').addEventListener('click',()=>{input.value=4;update();});update();
  }
  const budget=document.getElementById('l135-budget');
  if(budget){
    budget.innerHTML=`<p><strong>Predict:</strong> how many new workers fit after $3 is already committed?</p><label for="l135-workers">New workers <output>12</output></label><input id="l135-workers" type="range" min="1" max="30" step="1" value="12"><p>Fixed: $3 existing commitments + $3 overhead; $10 cap. Each worker reserves 900 seconds × $0.00022572/s.</p><p id="l135-budget-result" aria-live="polite"></p><p>Baseline: 12 workers reserve $2.437776; total with overhead $8.437776.</p><button type="button">Reset budget</button>`;
    const input=budget.querySelector('input');const update=()=>{const n=Number(input.value),cost=n*900*.00022572,total=6+cost;budget.querySelector('output').textContent=n;budget.querySelector('#l135-budget-result').textContent=`New reservation $${cost.toFixed(6)}; total with overhead $${total.toFixed(6)}. ${total<=10?'ALLOW launch':'REFUSE launch'}: $${Math.abs(10-total).toFixed(6)} ${total<=10?'remains':'over cap'}.`;};
    input.addEventListener('input',update);budget.querySelector('button').addEventListener('click',()=>{input.value=12;update();});update();
  }
  if(window.RetrievalBank) RetrievalBank.mount(document.getElementById('warmup'),{upTo:135,count:3});
  if(window.Teachback) Teachback.mount(document.getElementById('l135-teachback'),{prompt:'Why can a completely executed tuning study still fail to establish general superiority?',points:['One database and temporal split','Validation selected the winner','Seeds measure fit variation, not search or dataset uncertainty','Test results cannot guide a new search without changing the claim'],model:'The search answers a bounded decision on one task. Validation chooses the configuration, then fresh fits measure how that frozen choice behaves. Their seed spread does not cover database variation or the uncertainty in choosing a winner. Using the test outcome to alter the search would make test part of development.'});
})();
