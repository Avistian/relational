RetrievalBank.mount(document.getElementById('warmup'), {upTo:72,count:3});
ContrastiveViews.lossTrace(document.getElementById('loss-trace'));
ContrastiveViews.subsetTrace(document.getElementById('subset-trace'));
Predict.mount(document.getElementById('prediction'), {
 prompt:'Must a lower companion-recognition loss improve downstream classification?',
 options:[{label:'It guarantees useful classification',value:'yes'},{label:'It requires downstream measurement',value:'no'}],correct:'no',
 reveal:'The pretext task can preserve nuisance identity or erase class signal. Evaluate a frozen probe with matched splits and labeled budgets; inspect paired gains below.'
});
Teachback.mount(document.getElementById('teachback'), {
 prompt:'Trace one row from unlabeled input to a class prediction in SCARF and SubTab. Why do their objectives and transfer paths differ?',
 points:['SCARF matches a clean row to its corrupted companion in an N-way matrix.','SubTab encodes fixed feature subsets, reconstructs the whole row and can align pairs.','Projection heads train the pretext objective; the probe reads encoder latents.','The lab freezes encoders and counts validation labels; this differs from SCARF paper fine-tuning.'],
 model:'SCARF shares an encoder across a clean and a corrupted branch, and learns to identify the paired row. SubTab shares an encoder across subsets, asks each subset to reconstruct the whole, and can add pairwise recognition and distance. At transfer SCARF uses the clean-row latent; SubTab averages subset latents for each row. The lab freezes those representations and fits a linear classifier using designated labels. Neither pretext loss guarantees a useful class boundary.'
});
