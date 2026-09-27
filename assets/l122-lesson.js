(() => {
 if(document.getElementById('l122-reg'))RegConstructionViz.mount(document.getElementById('l122-reg'));
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:122,count:3});
 if(window.Teachback)Teachback.mount(document.getElementById('l122-teachback'),{
 prompt:'Why does changing person-table row order change edge_index while leaving the represented relationships unchanged?',
 points:['A raw primary key identifies an entity.','An edge endpoint is a table-local row coordinate.','Recompute the key-to-index mapping after permutation.','Keep features in the same new row order; preserve typed key pairs.'],
 model:'Person90 remains the same entity after a permutation, but its local index can change from 0 to 1. Rebuild the key lookup and edge endpoints and reorder row features consistently. The typed key pairs stay identical. A tensor equality check without accounting for the new coordinates would test the wrong invariant.'});
})();
