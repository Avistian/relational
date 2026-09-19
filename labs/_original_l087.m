% Native released MATLAB replay; run from relational/ with Statistics Toolbox.
% matlab -batch "run('labs/_original_l087.m')"
% This driver calls the archived functions without rewriting their RNG or AUC.
lab=fileparts(mfilename('fullpath'));
addpath(fullfile(lab,'sources','l087','MATLAB'));
addpath(fullfile(lab,'sources','l087','MATLAB','utils'));
names={'USAir','NS','PB','Yeast','Celegans','Power','Router','Ecoli'};
out=fullfile(lab,'results','l087','matlab');if ~exist(out,'dir'),mkdir(out);end
for d=1:length(names)
 data=load(fullfile(lab,'sources','l087','MATLAB','data',[names{d} '.mat']));
 scores=zeros(10,3);
 for seed=1:10
  rng(seed);
  [train,test]=DivideNet(data.net,.9,false);
  [train_pos,train_neg,test_pos,test_neg]=sample_neg(triu(train,1),triu(test,1),1,1,false);
  target=struct('pos',test_pos,'neg',test_neg);
  [scores(seed,1),~,cn]=CN(train,target);
  [scores(seed,2),~,aa]=AA(train,target);
  [scores(seed,3),~,ra]=RA(train,target);
  save(fullfile(out,sprintf('%s-%d.mat',names{d},seed)),'train_pos','train_neg','test_pos','test_neg','cn','aa','ra');
 end
 save(fullfile(out,[names{d} '-summary.mat']),'scores');
 disp(names{d});disp(mean(scores));disp(std(scores));
end
