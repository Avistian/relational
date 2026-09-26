"""Rule out a constructor draw-order mismatch for the full SBM dimensions."""
import json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P/'relkit'),str(P/'sources/l107/original')]
import snapshot_l107 as port,egcn_h,egcn_o,models,utils
rows=[]
for variant in ['H','O']:
 hidden=50 if variant=='H' else 51;classifier=100 if variant=='H' else 565
 torch.manual_seed(1234);torch.rand(1000,3)
 source=(egcn_h if variant=='H' else egcn_o).EGCN(utils.Namespace({'feats_per_node':162,'layer_1_feats':hidden,'layer_2_feats':hidden}),torch.nn.RReLU())
 head=models.Classifier(utils.Namespace({'gcn_parameters':{'cls_feats':classifier}}),in_features=2*hidden,out_features=2)
 source_rng=torch.get_rng_state()
 torch.manual_seed(1234);torch.rand(1000,3)
 target=port.EvolveGCN(162,hidden,variant);target_head=port.PairClassifier(hidden,classifier)
 torch.testing.assert_close(source_rng,torch.get_rng_state(),rtol=0,atol=0)
 count=0
 for original,replica in zip(source.GRCU_layers,target.layers):
  pairs=[(original.GCN_init_weights,replica.initial),(original.evolve_weights.choose_topk.scorer,replica.summary.scorer)]
  for a,b in zip([original.evolve_weights.update,original.evolve_weights.reset,original.evolve_weights.htilda],replica.gates):
   pairs.extend((getattr(a,name),getattr(b,name)) for name in ['W','U','bias'])
  for a,b in pairs:torch.testing.assert_close(a,b,rtol=0,atol=0);count+=a.numel()
 for key,value in head.state_dict().items():torch.testing.assert_close(value,target_head.state_dict()[key],rtol=0,atol=0);count+=value.numel()
 rows.append({'variant':variant,'parameters_checked':count,'initial_values':'EXACT','RNG_after_construction':'EXACT'})
result={'status':'PASS','torch':torch.__version__,'scope':'Current local runtime, full SBM dimensions, released seed and discarded random features; no copied weights','variants':rows}
(P/'_initialization_l107_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
