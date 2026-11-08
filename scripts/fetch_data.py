"""Download a bounded Python subset. Dataset content is never executed."""
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
DATA.mkdir(exist_ok=True)
BASE='https://datasets-server.huggingface.co/filter?'
def fetch(job):
    split,human,offset=job
    path=DATA/f'{split}_{human}_{offset}.json'
    if path.exists():
        return
    where='"language" = \'Python\' AND "source" '+('= \'Human\'' if human else '<> \'Human\'')
    url=BASE+urllib.parse.urlencode(dict(dataset='HanxiGuo/CodeMirage',config='default',split=split,where=where,offset=offset,length=100))
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url,timeout=25) as r:
                obj=json.load(r)
            if 'rows' not in obj: raise ValueError(obj)
            path.write_text(json.dumps(obj))
            print(path.name,len(obj['rows']),flush=True)
            return
        except Exception:
            if attempt==1:
                # Official files put human rows before AI rows. Verify every fallback row.
                base = 0 if human else (7000 if split == 'train' else 3000)
                fallback = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(dict(dataset='HanxiGuo/CodeMirage',config='default',split=split,offset=base+offset,length=100))
                with urllib.request.urlopen(fallback,timeout=60) as r: obj=json.load(r)
                if not obj.get('rows') or any(e['row']['language']!='Python' or (e['row']['source']=='Human')!=human for e in obj['rows']):
                    raise ValueError('Fallback rows did not match requested class/language')
                path.write_text(json.dumps(obj))
                print(path.name,len(obj['rows']), 'fallback',flush=True)
                return
            time.sleep(2+attempt)
if __name__=='__main__':
    jobs=[(split,human,offset) for split in ('train','test') for human in (True,False) for offset in (0,100,200)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,jobs))
    manifest={'dataset':'HanxiGuo/CodeMirage','url':'https://huggingface.co/datasets/HanxiGuo/CodeMirage','license':'CC-BY-NC-ND-4.0','selection':'First 300 Python rows per class in each official split; convenience subset, not representative.','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.json') if p.name!='manifest.json'}}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))
