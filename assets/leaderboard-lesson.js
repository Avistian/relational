(function(){
  RetrievalBank.mount(document.getElementById('warmup'),{upTo:56,count:3});
  LeaderboardAudit.mountWeight(document.getElementById('weighting-widget'));
  LeaderboardAudit.mountElo(document.getElementById('elo-widget'));
  Predict.mount(document.getElementById('predict-result'),{
    prompt:'Four archived methods, same datasets: must default, tuned and ensembled procedures have the same leader?',
    options:[{label:'The leader must stay fixed',value:'fixed'},{label:'The leader may change places',value:'change'}],correct:'change',
    reveal:'In our four-method mean-rank audit the leaders are CatBoost, TabM and RealMLP. Read the scope and uncertainty before interpreting that order.'});
  Teachback.mount(document.getElementById('teachback'),{
    prompt:'A teammate says: “TabM is first on an IID leaderboard, so it is the best model for predicting next quarter’s outcomes.” Write a corrected claim and the experiment you would require.',
    points:['Pin the snapshot, method variant and tuning/ensemble budget','Name the dataset population and temporal deployment mismatch','Distinguish equal-dataset ranking from pooled folds and accuracy','Request a leakage-safe temporal evaluation with matched baselines'],
    model:'That ranking describes a particular method pool and evaluation recipe on an IID dataset population. It does not establish next-quarter performance. I would freeze the snapshot and configurations, construct point-in-time features and label-availability cutoffs, select using earlier validation periods, then compare strong tabular baselines on untouched future periods. Cached-score analysis cannot certify the original training pipeline.'});
})();
