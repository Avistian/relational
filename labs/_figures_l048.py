"""Export actual widget states into Colab-compatible notebook PNGs."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from _figures_l047 import render
from relkit.benchmark_report import plot_scores,plot_ranks,plot_scaleup

HERE=Path(__file__).parent
DEST=HERE/'figures/l048'
LABELS={'mlp':'MLP','dense':'Dense cross + MLP','lowrank':'Factored cross + MLP','catboost':'CatBoost'}
STATES={'cross':'cross-640-1','degree':'degree-640-2','rank':'rank-640-1',
        'parallel':'architecture-640-0','stacked':'architecture-640-1','mix':'mix-640-0'}


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    render(HERE.parent/'assets/dcnv2-model-architecture.svg',DEST/'model-architecture.png')
    with tempfile.TemporaryDirectory(prefix='l048-svg-') as temp:
        subprocess.run(['node',str(HERE/'_viz_check_l048.js'),temp],check=True)
        for name,state in STATES.items():
            render(Path(temp)/(state+'.svg'),DEST/(name+'.png'))
    result=json.loads((HERE/'_verify_l048_results.json').read_text())
    for name,plot in [('scores',plot_scores),('ranks',plot_ranks)]:
        fig=plot(result,LABELS,title='Author reference · '+('local AUROC' if name=='scores' else 'rank uncertainty'))
        fig.savefig(DEST/(name+'.png'),dpi=150);plt.close(fig)
    scale=json.loads((HERE/'_scaleup_l048_results.json').read_text())
    fig=plot_scaleup(scale,title='Author reference · 739,012 MovieLens rows, three seeds')
    fig.savefig(DEST/'paper-results.png',dpi=150);plt.close(fig)
    manifest={'architecture_svg_sha256':hashlib.sha256((HERE.parent/'assets/dcnv2-model-architecture.svg').read_bytes()).hexdigest(),'states':STATES,'renderer':'librsvg, not a browser','results_sha256':hashlib.sha256((HERE/'_verify_l048_results.json').read_bytes()).hexdigest(),
              'widget_sha256':hashlib.sha256((HERE.parent/'assets/dcn-viz.js').read_bytes()).hexdigest(),
              'png_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.glob('*.png'))}}
    (DEST/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
