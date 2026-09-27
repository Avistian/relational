(function(){
if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:131,count:3});
StackTraceViz.mountTime(document.getElementById('l131-time'));
const data=JSON.parse(document.getElementById('l131-gradient-data').textContent);
StackTraceViz.mountGradient(document.getElementById('l131-gradient'),data);
})();
