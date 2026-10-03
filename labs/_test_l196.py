"""Independent behavioral checks for learner contracts; initially RED."""
import copy,itertools,json
from pathlib import Path
from relkit.community_l196 import summarize_cases,route_question,feedback_state

def check(summarize=summarize_cases,route=route_question,feedback=feedback_state):
 cases=[dict(unseen=u,before=[0,1,2],after=[1,2,3] if u in ['a','0'] else [0,1,2],numeric_before=[1,2,3],numeric_after=[1,2,3],query_alone=[0],query_with_other=[0,1] if u in ['a','0'] else [3,0]) for u in ['a','z','0','e']]
 assert summarize(cases)==dict(cases=4,changed_cases=['a','0'],batch_sensitive_cases=['a','0'],numeric_controls_unchanged=True,model_effect='NOT_ESTABLISHED')
 for variant in [cases[:-1],cases+[cases[0]],[]]:
  try:summarize(variant)
  except ValueError:pass
  else:raise AssertionError('Incomplete/duplicate cases admitted')
 bad=copy.deepcopy(cases);bad[0]['after']=[1,2]
 try:summarize(bad)
 except ValueError:pass
 else:raise AssertionError('Mismatched vectors admitted')
 bad=copy.deepcopy(cases);bad[0]['after'][0]=float('nan')
 try:summarize(bad)
 except ValueError:pass
 else:raise AssertionError('Invalid numeric evidence admitted')
 bad=copy.deepcopy(cases);bad[0]['numeric_after'][0]=99
 assert summarize(bad)['numeric_controls_unchanged'] is False
 assert route('rdblearn')=='https://github.com/HKUSHXLab/rdblearn/issues'
 assert route('relbench')=='https://huggingface.co/relbench'
 assert route('pyg')=='https://github.com/pyg-team/pytorch_geometric/discussions'
 try:route('unknown')
 except ValueError:pass
 else:raise AssertionError('Unknown owner admitted')
 for posted,replied,verified in itertools.product([False,True],repeat=3):
  expected='DRAFT_ONLY' if not posted else 'AWAITING_RESPONSE' if not replied else 'RESPONSE_UNVERIFIED' if not verified else 'FEEDBACK_CHECKED'
  assert feedback('https://example.org/thread/1' if posted else '', 'An actual reply' if replied else '',verified)==expected
 for url in ['invented','file:///tmp/x','https://']:
  try:feedback(url,'',False)
  except ValueError:pass
  else:raise AssertionError('Invalid thread URL admitted')
 try:feedback('', '', 'yes')
 except ValueError:pass
 else:raise AssertionError('Nonboolean verification admitted')
 return {'status':'PASS','complete_cases':4,'feedback_states':8,'invalid_inputs':'REJECTED'}
if __name__=='__main__':print(json.dumps(check(),indent=2))
