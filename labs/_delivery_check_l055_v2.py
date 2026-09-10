"""Current L055 acceptance; no shared delivery registry mutation."""
import base64,hashlib,json,re
from pathlib import Path
from io import BytesIO
import nbformat
import numpy as np
from PIL import Image
from _build_l055 import ROOT,SLUG,build
from _check_quality_audit import check_lesson
from relkit.temporal_experiment_v2 import metric,summarize

r=json.loads((ROOT/'_verify_l055_v2_results.json').read_text())
# The measured source identities are checked, never relabeled after changes.
for artifact in ('_verify_l055_v2_results.json','_verify_l055_results.json'):
    d=json.loads((ROOT/artifact).read_text())
    for path,sha in d['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,(artifact,path)
student=nbformat.read(ROOT/(SLUG+'.ipynb'),as_version=4)
teacher=nbformat.read(ROOT/'solutions'/(SLUG+'.ipynb'),as_version=4)
for nb,solution in [(student,False),(teacher,True)]:
    canonical=build(solution)
    assert [(c.cell_type,c.source) for c in nb.cells]==[(c.cell_type,c.source) for c in canonical.cells]
    for c in nb.cells:
        if c.cell_type=='code':
            if solution: assert c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
            else: assert c.execution_count is None and not c.outputs
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==5
images=[]
for c in student.cells:
    if c.cell_type=='markdown':
        assert 'attachment:' not in c.source
        for encoded in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',c.source):
            raw=base64.b64decode(encoded);Image.open(BytesIO(raw)).verify();images.append(hashlib.sha256(raw).hexdigest())
assert len(images)==8
assert set(images)=={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'figures/l055').glob('*.png')}
current=json.loads((ROOT/'data/cache/l055/student-v2-results.json').read_text())
count=0
for name,task in r['tasks'].items():
    assert task['random'][0]['audit']['pool_hash']==task['temporal'][0]['audit']['pool_hash']
    assert task['random'][0]['audit']['sizes']==task['temporal'][0]['audit']['sizes']
    for strategy,records in task.items():
        for arm,runs in records[0]['arms'].items():
            for run,live in zip(runs,current['tasks'][name][strategy][0]['arms'][arm]):
                assert run['selected']==int(np.argmin(run['validation_errors']))==live['selected']
                value=metric(np.array(records[0]['test_targets']),np.array(run['predictions']),name=='sberbank-housing')
                assert abs(value-run['error'])<1e-6
                np.testing.assert_allclose(run['predictions'],live['predictions'],rtol=1e-6,atol=1e-7)
                count+=1
assert summarize(r)==r['summary']
assert current['live_identity']['functions'] and current['release_identity']['status']=='PASS'
ticket=json.loads((ROOT/'data/cache/l055/exit-v2.json').read_text())
assert ticket['interpretation_complete'] and ticket['live_identity']==current['live_identity']
assert len(ticket['paper_summary'])==64 and len(ticket['paper_missing'])==1
consistency=check_lesson(55,require_review=True)
checks=dict(status='PASS',prediction_selection_and_notebook_reconciliations=count,student_todos=5,portable_figures=8,
 solution_code_cells=sum(c.cell_type=='code' for c in teacher.cells),canonical_package=consistency,
 historical_source_identity='UNCHANGED',live_identity='PRESENT',extracted_data_identity='PASS',
 paper_reanalysis='2879 original reports; independent checker stored separately',paper_training='NOT_RUN',closer='NOT_RUN',live_colab='NOT_CHECKED',deployment='PARENT_VERIFICATION')
(ROOT/'_delivery_l055_v2_results.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k!='canonical_package'},indent=2))
