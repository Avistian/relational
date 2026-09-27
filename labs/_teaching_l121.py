"""Notebook's live mechanism computations. No external dataset or cloud launch."""
# NOTEBOOK_RUN
rules=exists_late([(11,7,0),(12,7,1),(13,8,1)],[7,8,9])
rows=[(11,7,2.,2,2),(12,7,8.,4,7),(13,8,3.,3,3),(14,7,4.,4,4)]
features_day5=aggregate_at(rows,[8,7,9],5)
features_day7=aggregate_at(rows,[8,7,9],7)
values=torch.tensor([[2.],[8.],[2.],[8.]],requires_grad=True)
path=path_signal(values,torch.tensor([0,1,2,2]),torch.tensor([0,0,1]),3,2)
path[0].sum().backward()
assert path.flatten().tolist()==[68.,100.]
assert np.array_equal(features_day5,[[1,3,3],[2,6,4],[0,0,0]])
report={'status':'PASS','evidence':'COURSE_ONLY','rule':rules,'day5_features':features_day5.tolist(),'day7_features':features_day7.tolist(),'path_outputs':path.flatten().tolist(),'root_A_input_gradients':values.grad.flatten().tolist(),'paper_result':'NOT_RUN','learner':'PENDING_WRITTEN_DEFENSE'}
Path('l121-task-report.json').write_text(json.dumps(report,indent=2));print(report)
