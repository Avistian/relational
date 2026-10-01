"""Fresh CPU BART encoding. Run via _budget_l163.py; never uses target values."""
def fetch_model(packet, destination):
    import hashlib,json,requests
    from pathlib import Path
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for name in ['config.json','vocab.json','merges.txt','tokenizer.json','model.safetensors']:
        path=destination/name
        url='https://huggingface.co/'+packet['model_id']+'/resolve/'+packet['revision']+'/'+name
        if not path.exists():
            response=requests.get(url,stream=True,timeout=(30,90));response.raise_for_status()
            tmp=path.with_suffix(path.suffix+'.partial')
            with tmp.open('wb') as out:
                for chunk in response.iter_content(1024*1024):out.write(chunk)
            tmp.replace(path)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        artifacts.append(dict(file=name,sha256=digest,bytes=path.stat().st_size,url=url))
    return artifacts


def encode_rows(packet, model_path, serialize, pool):
    import time, numpy as np, torch
    from transformers import BartTokenizerFast, BartModel
    torch.set_num_threads(1)
    tokenizer=BartTokenizerFast.from_pretrained(str(model_path),local_files_only=True)
    model=BartModel.from_pretrained(str(model_path),local_files_only=True,attn_implementation='eager').float().eval()
    model.requires_grad_(False);encoder=model.get_encoder()
    assert model.config.d_model==768 and model.config.encoder_layers==6 and model.config.max_position_embeddings==1024
    arrays={};traces={};timings={};parity=[]
    for variant in packet['variants']:
        texts=[serialize(row,variant) for row in packet['rows']];outputs=[];records=[];start=time.perf_counter()
        for pos in range(0,len(texts),packet['batch_size']):
            batch=texts[pos:pos+packet['batch_size']]
            tokens=tokenizer(batch,padding=True,truncation=False,return_tensors='pt')
            if tokens.input_ids.shape[1]>packet['max_length']:raise ValueError('Overlength row; no silent truncation allowed')
            with torch.inference_mode():hidden=encoder(**tokens).last_hidden_state
            pooled=pool(hidden.numpy(),tokens.attention_mask.numpy())
            # Independent torch reduction; catches NumPy pooling errors on real states.
            reference=(hidden*tokens.attention_mask[:,:,None]).sum(1)/tokens.attention_mask.sum(1)[:,None]
            np.testing.assert_allclose(pooled,reference.numpy(),atol=2e-6,rtol=2e-6)
            outputs.append(pooled.astype(np.float32))
            for j,text in enumerate(batch):
                ids=tokens.input_ids[j][tokens.attention_mask[j].bool()].tolist()
                records.append(dict(id=packet['rows'][pos+j]['id'],text=text,token_ids=ids,tokens=tokenizer.convert_ids_to_tokens(ids),length=len(ids)))
        arrays[variant]=np.concatenate(outputs);traces[variant]=records;timings[variant]=time.perf_counter()-start
        # Same first row alone: padding/batch composition should not materially change it.
        token=tokenizer(texts[0],return_tensors='pt')
        with torch.inference_mode():h=encoder(**token).last_hidden_state.numpy()
        alone=pool(h,token.attention_mask.numpy())[0]
        diff=float(np.max(np.abs(alone-arrays[variant][0])))
        np.testing.assert_allclose(alone,arrays[variant][0],atol=1e-5,rtol=1e-5);parity.append(dict(variant=variant,max_abs_difference=diff))
        print(variant,arrays[variant].shape,'seconds',round(timings[variant],2),flush=True)
    return arrays,traces,dict(seconds=timings,batch_padding_parity=parity,encoder_width=768,encoder_layers=6,model_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),device='cpu',dtype='float32')


if __name__=='__main__':
    import os
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    import hashlib,json,time,platform
    from importlib.metadata import version
    from pathlib import Path
    import numpy as np
    from relkit.rows_l163 import serialize_row,masked_mean
    p=Path(__file__).resolve().parent;e=p/'evidence/l163';raw=(e/'fixtures.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((e/'input-manifest.json').read_text())['fixtures_sha256']
    packet=json.loads(raw);model_path=Path.home()/'.cache/relational/l163'/packet['revision'];start=time.perf_counter()
    artifacts=fetch_model(packet,model_path)
    arrays,traces,receipt=encode_rows(packet,model_path,serialize_row,masked_mean)
    np.savez_compressed(e/'embeddings.npz',**arrays)
    (e/'token-traces.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n')
    receipt.update(status='PASS',rows_per_variant=len(packet['rows']),variants=packet['variants'],model_id=packet['model_id'],revision=packet['revision'],artifacts=artifacts,
      elapsed_with_fetch_seconds=time.perf_counter()-start,platform=platform.platform(),python=platform.python_version(),
      packages={n:version(n) for n in ['numpy','torch','transformers','tokenizers','safetensors']},
      source_sha256={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['relkit/rows_l163.py','_encode_l163.py']},
      fixtures_sha256=hashlib.sha256(raw).hexdigest(),embeddings_sha256=hashlib.sha256((e/'embeddings.npz').read_bytes()).hexdigest(),
      token_traces_sha256=hashlib.sha256((e/'token-traces.json').read_bytes()).hexdigest(),cloud_spend_usd=0)
    (e/'encoding-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Fresh frozen encoding complete:',round(receipt['elapsed_with_fetch_seconds'],2),'seconds')
