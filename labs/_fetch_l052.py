"""Extract only three numeric tasks using byte ranges of the authors' uncompressed tar.

Checks Content-Range on every request; never silently downloads the whole 3 GB file.
Once created, _data_l052.json supplies exact offsets and hashes for cheap reruns.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = 'https://huggingface.co/datasets/puhsu/tabular-benchmarks/resolve/main/data.tar'
TASKS = ('california', 'house', 'higgs-small')


def read_range(offset, size):
    request = urllib.request.Request(URL + '?l052range=' + str(offset), headers={'Range': f'bytes={offset}-{offset+size-1}'})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Server did not honor range request')
        data = response.read(size)
    if len(data) != size:
        raise RuntimeError('Truncated range')
    return data


def fetch():
    cache = HERE / 'data/cache/l052'; cache.mkdir(parents=True, exist_ok=True)
    manifest = HERE / '_data_l052.json'
    records = json.loads(manifest.read_text())['files'] if manifest.exists() else {}
    if not records:
        offset, block_start, block = 0, -1, b''
        while offset < 3094384640:
            if not (block_start <= offset and offset + 512 <= block_start + len(block)):
                block_start, block = offset, read_range(offset, min(65536,3094384640-offset))
            header = block[offset-block_start:offset-block_start+512]
            if not header.strip(b'\0'): break
            name = header[:100].split(b'\0')[0].decode()
            size = int(header[124:136].strip(b'\0 ') or b'0',8)
            parts = name.split('/')
            if len(parts) == 3 and parts[1] in TASKS and (parts[2].startswith(('X_num_', 'Y_')) or parts[2]=='info.json'):
                records[name] = {'offset':offset+512, 'size':size}
                print('Located',name,size,flush=True)
            offset += 512 + ((size+511)//512)*512
            if len(records) == 21: break
        if len(records) != 21: raise RuntimeError(f'Expected 21 files, found {len(records)}')
    for name, meta in records.items():
        target = cache / name.removeprefix('data/')
        if target.exists() and meta.get('sha256') == hashlib.sha256(target.read_bytes()).hexdigest(): continue
        data = read_range(meta['offset'], meta['size'])
        digest = hashlib.sha256(data).hexdigest()
        if meta.get('sha256') and meta['sha256'] != digest: raise RuntimeError('Dataset bytes changed')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);meta['sha256']=digest
        print('Saved',name,flush=True)
    manifest.write_text(json.dumps({'url':URL,'release':'authors linked archive; verified file hashes','files':records},indent=2))
    return cache


if __name__ == '__main__': fetch()
