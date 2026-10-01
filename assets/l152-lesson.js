(function(){
'use strict';
RetrievalBank.mount(document.getElementById('warmup'),{upTo:152,count:3});
RegressionCalibrationViz.mount(document.getElementById('l152-calibration'));
Predict.mount(document.getElementById('predict'),{prompt:'What does a zero empirical median violation establish?',options:[{label:'The bin satisfies empirical inequalities',value:'empirical'},{label:'The model guarantees future calibration',value:'future'}],correct:'empirical',reveal:'The bin satisfies empirical inequalities. Finite counts, repeated queries and grouping prevent a guarantee of future conditional calibration.'});
Teachback.mount(document.getElementById('teachback'),{prompt:'Why can a median-balanced prediction have a nonzero mean residual? Defend one measured bin and its evidence limits.',points:['L1 targets a conditional median; squared error targets a mean.','Strict inequalities keep ties explicit.','Validation fixes bin edges; sample counts limit conclusions.','Fresh execution and historical identity are separate.'],model:'For outcomes0,1,1,9, predicting1 yields25%below,50%equal,25%above. Its mean residual is1.75 because the right tail moves the mean. A bin can satisfy empirical median inequalities without proving conditional population calibration. Preserve counts, the fixed edges and the full seed evidence.'});
})();
