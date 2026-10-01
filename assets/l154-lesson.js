(function(){'use strict';
RetrievalBank.mount(document.getElementById('warmup'),{upTo:153,count:3});
PortfolioEvidence.mount(document.getElementById('l154-evidence'));
Predict.mount(document.getElementById('predict'),{
 prompt:'A model has lower MAE than a published baseline. What can this report claim?',
 options:[{label:'A descriptive difference across evidence sources',value:'context'},
          {label:'A demonstrated victory under matched conditions',value:'victory'}],
 correct:'context',reveal:'The difference is descriptive context. A fresh matched baseline is still missing; matching the metric name is insufficient.'});
Teachback.mount(document.getElementById('teachback'),{
 prompt:'Why can three entry artifacts support only two completed test tasks and no local superiority claim?',
 points:['The recommendation entry contains a validation pilot, not five completed fits.',
         'Published baselines cannot replace locally executed matched comparisons.',
         'Seed SD describes variability on fixed test populations.',
         'AUROC, MAE and MAP need separate units and task-level conclusions.'],
 model:'Two tasks have complete fixed-protocol seed sets. The third has useful feasibility evidence, but no test result. No fresh matched FE or tree comparisons have been executed, so local superiority is not established. I retain per-task units, report the missing task, and defer mastery until my defense is reviewed.'});
})();
