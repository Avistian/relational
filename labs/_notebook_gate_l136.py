# This source is embedded by the builder after visible model/trainer definitions.
RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    assert not Path('l136-full').exists(), 'Use a fresh output directory'
    assert torch.cuda.is_available(), 'Full experiment requires the documented GPU runtime'
    import pyg_lib, importlib.metadata
    for package,version in {'torch':'2.5.1','relbench':'1.1.0','pytorch-frame':'0.2.3','torch-geometric':'2.6.1','sentence-transformers':'3.3.1','numpy':'1.26.4','pandas':'2.2.3'}.items():
        assert importlib.metadata.version(package).split('+')[0]==version, 'Use the pinned runtime: '+package
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.modeling.utils import get_stype_proposal
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1);seed_everything(42)
    dataset=get_dataset('rel-f1',download=True);db=dataset.get_db()
    task=get_task('rel-f1','driver-position',download=True)
    for name,expected in [('db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'),('tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e')]:
        assert hashlib.sha256((Path(dataset.cache_dir)/name).read_bytes()).hexdigest()==expected
    types=get_stype_proposal(db)
    text_model=SentenceTransformer(TEXT_SPEC['model'],revision=TEXT_SPEC['revision'],device='cpu')
    def embed(strings):return torch.from_numpy(text_model.encode(strings,show_progress_bar=False))
    data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256))
    del text_model
    original={};exec(ORIGINAL_MODEL_SOURCE,original)
    root=Path('l136-full');root.mkdir();records=[]
    configuration=dict(id='lr005-full',lr=.005,fanout=[128,64])
    for seed in range(5):
        output=root/f'seed-{seed}'
        r=fit_tuning(data,stats,task,seed,output,10,'cuda',original['Model'],configuration=configuration,evaluate_test=True)
        a=np.load(output/'predictions.npz')
        for split in ['val','test']:
            score=regression_score(a[split+'_target'],a[split+'_pred'],1.)['mae']
            assert abs(score-r['scores'][split])<1e-12
        records.append(dict(seed=seed,scores=r['scores']))
    packet=dict(status='PASS',final_fits=len(records),records=records)
    (root/'packet.json').write_text(json.dumps(packet,indent=2));print(packet)
else:
    print('Full training NOT_RUN in this kernel. Author evidence is replayed separately.')
