"""Capture eight real token-state traces per variant for live learner pooling."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import BartModel,BartTokenizerFast
from relkit.rows_l163 import serialize_row,masked_mean
P=Path(__file__).resolve().parent;E=P/'evidence/l163';p=json.loads((E/'fixtures.json').read_text());start=time.perf_counter()
model_path=Path.home()/'.cache/relational/l163'/p['revision'];torch.set_num_threads(1)
tokenizer=BartTokenizerFast.from_pretrained(model_path,local_files_only=True);model=BartModel.from_pretrained(model_path,local_files_only=True,attn_implementation='eager').float().eval();model.requires_grad_(False)
states={};arrays=dict(np.load(E/'embeddings.npz'));differences=[]
for variant in p['variants']:
 texts=[serialize_row(row,variant) for row in p['rows'][:8]];tokens=tokenizer(texts,padding=True,return_tensors='pt',truncation=False)
 with torch.inference_mode():hidden=model.get_encoder()(**tokens).last_hidden_state.numpy()
 mask=tokens.attention_mask.numpy();pooled=masked_mean(hidden,mask).astype(np.float32)
 np.testing.assert_array_equal(pooled,arrays[variant][:8])
 states[variant+'_hidden']=hidden;states[variant+'_mask']=mask
 differences.append(dict(variant=variant,max_abs_difference=float(abs(pooled-arrays[variant][:8]).max())))
np.savez_compressed(E/'pool-states.npz',**states)
r=dict(status='PASS',row_indices=list(range(8)),revision=p['revision'],cache_parity='EXACT',differences=differences,seconds=time.perf_counter()-start,sha256=hashlib.sha256((E/'pool-states.npz').read_bytes()).hexdigest())
(E/'pool-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
