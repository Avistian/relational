(function(){
 RetrievalBank.mount(document.getElementById('warmup'),{upTo:129,count:3});
 FECutoffViz.mount(document.getElementById('l129-cutoff'));
 Teachback.mount(document.getElementById('l129-teachback'),{prompt:'Does rerunning the released SQL and matching its prediction quality reproduce the claim of 96% less human effort?',points:['Separate machine runtime from active human work','State that the original study used one expert','Identify reused infrastructure excluded from marginal effort','Name the historical evidence still missing'],model:'No. The replay starts with features already invented and implemented. The paper compared marginal human work by one expert with reusable infrastructure excluded. A new learner effort log measures another setting. Historical data, original search seeds/models and human work cannot be recovered from a close score.'});
})();
