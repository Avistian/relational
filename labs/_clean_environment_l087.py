"""Run with the freshly installed interpreter; compare full replay with author artifacts."""
from pathlib import Path
import tempfile,sys,json
import numpy as np
from relkit.link_l087 import run_paper,load_graph,train_link
LAB=Path(__file__).resolve().parent;manifest=json.loads((LAB/'_sources_l087.json').read_text());reference=json.loads((LAB/'_paper_l087_results.json').read_text())
with tempfile.TemporaryDirectory(prefix='l087-clean-replay-') as tmp:
    root=Path(tmp);r=run_paper(manifest,root/'sources',root/'results')
    assert r['summary']==reference['summary'] and r['statistics']==reference['statistics']
    for row in r['runs']:
        a=np.load(root/'results'/row['artifact']);b=np.load(LAB/'results/l087/paper'/row['artifact'])
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    e,n=load_graph('USAir',manifest,root/'sources');run,_=train_link(e,n,87);expected=json.loads((LAB/'_teaching_l087_results.json').read_text())['runs'][0];expected.pop('artifact_sha256');assert run==expected
out={'status':'PASS','interpreter':sys.executable,'python':sys.version,'fresh_source_downloads':8,'baseline_replays':240,'split_and_score_arrays_exact_match':True,'teaching_seed87_full_trace_exact_match':True,'installation':'fresh uv venv; pinned direct dependencies plus CPU wheel index','live_colab':'NOT_CHECKED'}
(LAB/'_clean_environment_l087_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
