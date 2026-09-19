RetrievalBank.mount(document.getElementById('warmup'),{upTo:73,count:3});
LabelBudget.mount(document.getElementById('budget-trace'));
Predict.mount(document.getElementById('prediction'),{
 prompt:'Must every SSL gain curve contain one clear crossover?',
 options:[{label:'Every curve must cross once',value:'yes'},{label:'Some curves never cross zero',value:'no'}],correct:'no',
 reveal:'A measured curve may stay positive, stay negative, contain ties, or reverse several times. The notebook returns every strict adjacent mean-sign reversal and reports ties separately.'
});
Teachback.mount(document.getElementById('teachback'),{
 prompt:'Your mean gain changes sign between 20% and 40% labels. Explain what this establishes, what it does not, and how you would choose a deployment recipe.',
 points:['The same seed, split and nested label budget must be paired.','This is a bracket on the measured grid, not an exact population threshold.','Validation labels and extra pretraining compute count.','A test-derived choice needs fresh independent evaluation.'],
 model:'I first verify that treatment and control differ only in the intended pretraining intervention and share the relevant rows and supervised training. Opposite mean signs locate a descriptive bracket, but noisy repetitions do not establish a stable population crossing. I inspect practical baselines and count validation labels and pretraining cost. I select a recipe using a development protocol and confirm it on new held-out data.'
});
