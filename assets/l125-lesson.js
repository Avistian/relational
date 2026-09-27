(()=>{
 FrameContractViz.mount(document.getElementById('l125-frame'));
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:125,count:3});
 if(window.Teachback)Teachback.mount(document.getElementById('l125-teachback'),{prompt:'Why can perfect tensor shapes still produce the wrong graph prediction?',points:['Fitted preprocessing can include unavailable future rows.','Column order determines which learned role receives each token.','Node vectors must preserve entity identity.','Only graph message passing exchanges information between related rows.'],model:'I first freeze the preprocessing population, then preserve the fitted column roles and attach immutable entity IDs to every output vector. The row encoder mixes columns inside one row; the GNN uses foreign-key edges to move information across rows. Shape checks alone establish none of these contracts.'});
})();
