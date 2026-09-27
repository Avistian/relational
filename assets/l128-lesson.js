(function(){
 RetrievalBank.mount(document.getElementById('warmup'),{upTo:128,count:3});
 TaskMetricsViz.mount(document.getElementById('l128-auc'),'auc');
 TaskMetricsViz.mount(document.getElementById('l128-map'),'map');
 Teachback.mount(document.getElementById('l128-teachback'),{prompt:'Explain why the DNF label flip changes the task contract even though reversing both labels and probabilities preserves AUROC.',points:['State what historical positive means','Separate target meaning from ranking quality','Explain the two inversions','Name the unavailable historical evidence'],model:'The old positive encodes no future non-finish status; the newer positive is its complement. Reversing both y and p preserves every positive-negative ordering credit, so the metric cannot establish label semantics. We verify query keys and labels against original SQL, but lack the old archive bytes and ordering.'});
})();
