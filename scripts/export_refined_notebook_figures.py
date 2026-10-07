"""Render the three model maps for notebook-width display and update images only.

Never executes or rebuilds notebooks: code cells, outputs and metadata are preserved.
The lesson builders read the generated PNGs for future portable exports.
"""
import argparse,base64,json
from pathlib import Path
import cairosvg
from visual_details import DETAILS
from visual_detail_layouts import PRIMARY,render_board
ROOT=Path(__file__).resolve().parents[1]
NAMES={'b09':'b09-cost-frontier','b18a':'b18a-context-state','b19b':'b19b-forecasting-contracts'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--update',action='store_true');args=parser.parse_args()
    records=[]
    for key in sorted(PRIMARY):
        title,desc,*_=DETAILS[key]
        svg=render_board(key,title,desc,mobile=False,columns=2)
        target=ROOT/f'labs/figures/{key}/architecture-refined.png'
        previous=target.read_bytes() if target.exists() else b''
        new=cairosvg.svg2png(bytestring=svg.encode(),scale=1.5)
        if args.update:target.write_bytes(new)
        else:assert target.read_bytes()==new,f'Stale PNG: {target}'
        oldname='exaone-architecture.png' if key=='b09' else 'architecture.png'
        old=base64.b64encode((target.parent/oldname).read_bytes()).decode()
        encoded=base64.b64encode(new).decode()
        for folder,suffix in [('labs','.ipynb'),('labs/solutions','.ipynb'),('labs/html','.html')]:
            path=ROOT/f'{folder}/{NAMES[key]}{suffix}'
            if not path.exists():continue
            text=path.read_text()
            prior=base64.b64encode(previous).decode() if previous else old
            match=old if old in text else prior
            if encoded in text:
                pass
            elif match in text:
                if args.update:path.write_text(text.replace(match,encoded))
                else:raise AssertionError(f'Old notebook image: {path}')
            else:
                # Later refreshes replace only the first architecture data URI,
                # located by the unique new figure's role (never code payloads).
                assert encoded in text,f'No matching architecture payload: {path}'
            records.append(str(path.relative_to(ROOT)))
    print(json.dumps(dict(portable_figures=3,updated_or_verified=records)))
if __name__=='__main__':main()
