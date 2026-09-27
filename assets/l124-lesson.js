(()=>{
 TaskTableViz.mount(document.getElementById('l124-task'));
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:124,count:3});
 if(window.Teachback)Teachback.mount(document.getElementById('l124-teachback'),{prompt:'Why is future data necessary for labels, but dangerous for model input and cohort selection?',points:['A query fixes an entity and prediction cutoff.','The label is measured in a later declared window.','Inputs must be available at the query cutoff.','A future-participation cohort differs from a past-known prediction population.'],model:'The task row is a question asked at t. Future events can answer that question after the horizon, but cannot be known features at t. If future participation determines which entities enter evaluation, the score describes that selected population. I must declare how I handle no event and incomplete observation.'});
})();
