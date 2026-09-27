(()=>{
 if(document.getElementById('l123-temporal'))TemporalNeighborhoodViz.mount(document.getElementById('l123-temporal'));
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:123,count:3});
 if(window.Teachback)Teachback.mount(document.getElementById('l123-teachback'),{prompt:'Why must every hop use the original query time, and why does that still not prove availability-time correctness?',points:['One root query defines one information cutoff.','An intermediate row date must not replace it.','Every sampled typed row and edge needs a visibility contract.','Ingestion and mutable-feature histories may be unavailable.'],model:'For a day8 query, a day7 memo about a day4 transfer is already known and may contribute. A day12 memo cannot. Each hop keeps day8. But a day5 transfer imported on day11 is still unavailable; an event-time-only release cannot verify that second clock. I must also defend feature histories and static-table assumptions.'});
})();
