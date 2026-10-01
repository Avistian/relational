# L169 selected context sweep

| Task / model | K=64 | 128 | 256 | 512 reused | 1024 |
|---|---:|---:|---:|---:|---:|
| rel-f1 / RDBPFN | 0.6932 | 0.7000 | 0.7171 | 0.7219 | 0.7188 |
| rel-f1 / RDBPFN_single | 0.6063 | 0.6242 | 0.6126 | 0.6640 | 0.7086 |
| rel-f1 / TabICLv1.1 | 0.7142 | 0.7087 | 0.7137 | 0.7176 | 0.7173 |
| rel-trial / RDBPFN | 0.5474 | 0.5386 | 0.5844 | 0.5986 | 0.6159 |
| rel-trial / RDBPFN_single | 0.5212 | 0.5393 | 0.5858 | 0.5961 | 0.5963 |
| rel-trial / TabICLv1.1 | 0.5303 | 0.5447 | 0.5669 | 0.5928 | 0.6088 |

On rel-f1, RDB-PFN changes from **0.69318** at 64 to **0.71876** at 1024 (+0.02558 AUROC). On rel-trial, RDB-PFN changes from **0.54736** at 64 to **0.61595** at 1024 (+0.06859 AUROC). **30/30 published-mean comparisons** fall within the predeclared 0.02 AUROC descriptive tolerance. Negative mean doubling contrasts: rel-f1/RDBPFN 512→1024 (-0.00318); rel-f1/RDBPFN_single 128→256 (-0.01161); rel-f1/TabICLv1.1 64→128 (-0.00548); rel-f1/TabICLv1.1 512→1024 (-0.00023); rel-trial/RDBPFN 64→128 (-0.00879). These are measured responses under changing support draws, not evidence for a universal monotonic law.

240fresh+60reused evaluations;229,050predictions. Source protocol preserves independently drawn contexts. Pretraining law NOT_ESTABLISHED; whole paper/fresh pretraining NOT_RUN.
