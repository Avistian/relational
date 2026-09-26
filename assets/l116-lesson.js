(function(){
  RetrievalBank.mount(document.getElementById('l116-warmup'),{upTo:116,count:3});
  GNNDiagnostics.mountUpdate(document.querySelector('[data-gnn-update]'));
  GNNDiagnostics.mountLeak(document.querySelector('[data-gnn-leak]'));
  OversmoothingViz.mount(document.getElementById('l116-smoothing'));
  Teachback.mount(document.getElementById('l116-teachback'),{
    prompt:'A validation score changes even though every parameter stays fixed. Explain a possible mechanism, then give two probes that separate an absent optimizer step from label leakage.',
    points:['BN running buffers and dropout/evaluation state are distinct from learned parameters','Compare gradient norm with parameter-change norm','Perturb only unauthorized labels with fixed inputs and RNG','A passing local probe does not certify every pipeline boundary'],
    model:'Batch-normalization buffers can evolve without optimizer updates. Measure finite nonzero gradients and zero parameter movement to isolate the update path. Separately perturb held-out labels while holding inputs and RNG fixed; a changed training loss or update exposes label dependence. Audit selection and preprocessing separately.'
  });
})();
