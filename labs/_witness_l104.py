"""Capture concrete sampled violations without changing model or sampler behavior."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from relkit.tgat_l103 import load_wikipedia,TGAT
from relkit.leakage_l104 import AuditFinder,records_from_archive,score_fixed_questions
P=Path(__file__).resolve().parent
class WitnessFinder(AuditFinder):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.witnesses=[]
 def get_temporal_neighbor(self,nodes,cutoffs,num_neighbors=20):
  ids,edges,times=super().get_temporal_neighbor(nodes,cutoffs,num_neighbors)
  for i,j in np.argwhere((ids!=0)&(times>=np.asarray(cutoffs)[:,None])):
   if len(self.witnesses)<4:self.witnesses.append({'request_node':int(nodes[i]),'request_cutoff':float(cutoffs[i]),'sampled_neighbor':int(ids[i,j]),'sampled_edge_id':int(edges[i,j]),'sampled_event_time':float(times[i,j]),'violation':'SAME_TIME' if times[i,j]==cutoffs[i] else 'FUTURE'})
  return ids,edges,times
if __name__=='__main__':
 torch.set_num_threads(1);nodes,edges,data,_=load_wikipedia(P/'l103-cache')
 path=P/'evidence/l104/seed-0-selected.pt';expected=json.loads((P/'evidence/l104/notebook-inputs.json').read_text())[path.name]
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
 checkpoint=torch.load(path,map_location='cpu',weights_only=False);archive=np.load(P/'evidence/l104/seed-0-questions.npz')
 model=TGAT(AuditFinder(data['full'],len(nodes)),nodes,edges);model.load_state_dict(checkpoint['weights'],strict=False)
 questions={k:v[:120] for k,v in records_from_archive(archive,'all').items()};reports={}
 for mode in ['strict','inclusive','lookahead']:
  baseline=AuditFinder(data['full'],len(nodes),mode);model.ngh_finder=baseline;reference=score_fixed_questions(model,data['test'],questions,104)
  finder=WitnessFinder(data['full'],len(nodes),mode);model.ngh_finder=finder;traced=score_fixed_questions(model,data['test'],questions,104)
  for key in ['p','n']:np.testing.assert_array_equal(traced[key],reference[key])
  reports[mode]={'witnesses':finder.witnesses,'audit':finder.audit,'instrumentation_prediction_parity':'EXACT'}
 assert not reports['strict']['witnesses'] and reports['inclusive']['witnesses'] and reports['lookahead']['witnesses']
 result={'status':'PASS','scope':'120 all-test positive questions, seed-0 frozen checkpoint, CPU; captures actual sampled request records, not full parent chains','arms':reports}
 (P/'_witness_l104_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
