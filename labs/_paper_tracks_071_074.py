"""Bake complete, executable reproduction source into the four teaching notebooks."""
import ast
import json
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
REPRO=ROOT/'reproductions'

DESCRIPTIONS={
71:'''**Target:** VIME Table 2 MNIST, 6000 labeled / 54000 unlabeled / 10000 test rows. The original TensorFlow/Keras model and trainer below run in a separate Python 3.7 environment. It preserves the released initialization, optimizers, frozen transfer, fixed corruption bank, and 1000-step semi-supervised schedule. Ten declared seeds and all three released MLP arms are recorded.

**Unresolved paper detail:** the paper selected architectures by validation, but the release only supplies a fixed architecture. This executes the released experiment at the table's label budget; it cannot establish exact Table 2 parity. The code reports that limitation even if a score happens to match. The earlier PyTorch teaching experiment remains useful for interventions.''',
72:'''**Targets:** SCARF Tables 2 and 4, plus the released SubTab MNIST experiment compared with supplement Table A3. SCARF uses a 4-layer width-256 encoder, 2-layer heads, full encoder fine-tuning, Adam 0.001, batches of 128, static validation views, patience 3, and 30 splits. SubTab uses the released 343→784 encoder, full-row decoder, normalized 784-dimensional projector, four 75%-overlapping views, all six view pairs, and 15 epochs on all 60000 MNIST rows.

SCARF starts with three named OpenML datasets; its operator also runs all 69 published datasets. SubTab disables evaluation corruption as both official entry scripts require and reports the entire C sweep. The selected C behind Table A3 is not documented, so a coincident score is not proof of an exact match.''',
73:'''**Target:** SCARF Table 2 (100% labeled training) versus Table 4 (25% labeled training), using the same full implementation as Lesson 72. Validation and test proportions stay fixed at 10% and 20%; only the training-label fraction changes. Record 30 paired splits and compare both control and SCARF with the published per-dataset means.

The lesson's five-budget crossover experiment is an extension. It is distinct from these two published regimes. No new model or invented survey experiment is needed for this evaluation lesson.''',
74:'''**Target:** released CARTE single-table benchmark rows for wina_pl, wine_dot_com_prices and wine_vivino_price; seven training budgets, ten outer splits, fifteen validation-split models per fit. The default uses the June 2024 paper-era source; the newer release is an explicit alternative in the repository operator. The source below includes the entire architecture, graph conversion, checkpoint loader, prediction head, training loop and ensemble. Published per-split learning rates and scores are joined by dataset/budget/seed.

The model uses the release's checkpoint-loading behavior, including its handling of initial_x, and its full 300→150→75→1 prediction head. The 384-row teaching cache cannot supply these full-table runs. A full FastText model and released checkpoint are required. Fresh YAGO pretraining, all other benchmark datasets, and joint source-table transfer remain separate reproduction targets; downstream replay does not establish them.'''}



def source_chunks(source):
    """Keep imports together; expose individual functions and large-class methods."""
    tree=ast.parse(source);lines=source.splitlines(keepends=True);starts={0:'imports and module context'}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            line=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            starts[line]=node.name
            if isinstance(node,ast.ClassDef) and node.end_lineno-node.lineno>160:
                for method in node.body:
                    if isinstance(method,ast.FunctionDef):
                        line=min([method.lineno]+[d.lineno for d in method.decorator_list])-1
                        starts[line]=node.name+'.'+method.name
    indices=sorted(starts)
    return [(starts[a],''.join(lines[a:b])) for a,b in zip(indices,indices[1:]+[len(lines)]) if ''.join(lines[a:b]).strip()]



def paper_evidence(lesson):
    lines=['### Author-run reproduction evidence',
           'These are saved author measurements, not output from your current kernel. Protocol gaps above still apply.']
    files={71:['vime-release-results.json'],72:['scarf-paper-results.json','subtab-paper-results.json'],
           73:['scarf-paper-results.json'],74:['carte-paper-row-results.json']}[lesson]
    for file in files:
        lines.append('')
        path=REPRO/file
        if not path.exists():
            lines.append(f'`{file}`: full-run result not recorded yet.');continue
        result=json.loads(path.read_text())
        if file.startswith('vime'):
            lines+=['| Arm | Measured accuracy | Paper mean | Trials |','|---|---:|---:|---:|']
            for row in result['summary']:
                records=[r for r in result['records'] if r['arm']==row['arm']]
                lines.append(f"| {row['arm']} | {100*row['mean']:.2f}% | {100*records[0]['paper_mean']:.2f}% | {len(records)} |")
            lines.append('One historical full-data trial verifies the released pipeline. Ten-run table reproduction and the unpublished architecture selection remain incomplete.')
        elif file.startswith('scarf'):
            if 'summary' not in result:
                lines.append(f"SCARF: {len(result['records'])} completed fits; the declared 360-fit run is incomplete.");continue
            lines+=['| OpenML ID | Labels | Arm | Measured mean ± SD | Paper mean | Gap (pp) |','|---|---:|---|---:|---:|---:|']
            for r in result['summary']:
                lines.append(f"| {r['data_id']} | {100*r['fraction']:.0f}% | {r['arm']} | {100*r['mean']:.2f} ± {100*r['sd']:.2f}% | {100*r['paper_mean']:.2f}% | {100*r['gap']:+.2f} |")
            lines.append('Thirty splits per row, three published datasets. Seed SD measures split/training variability; it is not uncertainty over the full 69-dataset population.')
        elif file.startswith('subtab'):
            lines+=['| Probe C | Measured test accuracy | Table A3 mean reference |','|---:|---:|---:|']
            for r in result['records']:lines.append(f"| {r['C']:g} | {100*r['accuracy']:.2f}% | 97.86% |")
            lines.append('Full MNIST pretraining, seed 57. Keep the entire released C sweep; selecting its best test score would bias a held-out claim. The paper does not identify its Table A3 C.')
        else:
            lines+=['| Dataset / budget / split | Measured R² | Released reference | Gap |','|---|---:|---:|---:|']
            for r in result['records']:
                if 'r2' in r:lines.append(f"| {r['dataset']} / {r['budget']} / {r['split_seed']} | {r['r2']:.4f} | {r['paper_r2']:.4f} | {r['gap']:+.4f} |")
                else:lines.append(f"| {r['dataset']} / {r['budget']} / {r['split_seed']} | FAILED | {r['paper_r2']:.4f} | — |")
            lines.append('This is a selected downstream benchmark row, not a rerun of all table families or YAGO pretraining.')
    # Markdown needs a blank after each table before the explanatory paragraph.
    formatted=[]
    for line in lines[2:]:
        if formatted and formatted[-1].startswith('|') and not line.startswith('|'):
            formatted.append('')
        formatted.append(line)
    return '\n\n'.join(lines[:2])+'\n\n'+'\n'.join(formatted)


def paper_cells(lesson):
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text,metadata={'tags':['paper-reproduction']}))
    def code(text):cells.append(nbf.v4.new_code_cell(text,metadata={'tags':['paper-reproduction']}))
    md('<a id="paper-reproduction"></a>\n## Paper reproduction · implementation, protocol, and measured comparison\n\n'+DESCRIPTIONS[lesson]+'''\n\n**Read → predict → run.** Compare each architecture and stopping rule below with the teaching lane. Predict which protocol changes could reverse its ranking. The following PROVIDED cells write the visible source to an isolated `paper-run` folder. The runner imports those exact files, so edits in these cells are the implementation it trains. Run all source cells before enabling the final run. After editing a source chunk, rerun that file’s source cells from the first chunk so append cells do not duplicate definitions. Preserve the pinned copy for parity checks.

**Evidence rule:** a smoke run tests execution and is always INCOMPARABLE. A full run records published targets, measured gaps, source choices and missing details; a small score gap alone does not certify protocol identity. [Reproduction guide](../labs/reproductions/README.md).''')
    code("# PROVIDED: isolated files for the visible paper implementation\nfrom pathlib import Path\nimport sys, subprocess\nPAPER_ROOT=Path('paper-run').resolve()\nPAPER_ROOT.mkdir(exist_ok=True)")
    if lesson==71:
        files=[*sorted(p.relative_to(REPRO).as_posix() for p in (REPRO/'sources/vime').glob('*.py')),
               'requirements-vime.txt','vime_release.py','colab_vime.py','Dockerfile.vime']
    elif lesson in (72,73):
        files=['scarf.py','scarf_targets.json']+(['subtab.py'] if lesson==72 else [])
    else:
        files=[*sorted(p.relative_to(REPRO).as_posix() for p in (REPRO/'sources/carte_paper').rglob('*.py')),
               'sources/carte_ai/LICENSE.txt','requirements-carte-paper.txt','Dockerfile.carte','carte_bestparams.csv','carte_singletable.csv','carte_compat.py','carte_vectors.py','carte.py']
    for file in files:
        contents=(REPRO/file).read_text()
        if not contents:continue
        if file.endswith('.py'):
            md('### PROVIDED · '+file+'\n\n'+{
                'scarf.py':'Trace original-column corruption → f/g pretraining → validation checkpoint → new h → joint f/h fine-tuning. The full data split and comparison loop are visible here.',
                'subtab.py':'Trace subset windows → permutation noise → each of six pairs → joint loss → AdamW → mean latent representation → published linear-probe sweep.',
                'carte.py':'This orchestration calls the complete source modules shown above. It uses full tables and the released best-parameter ledger; it does not import the compact teaching model.',
                'vime_release.py':'This runner calls the original sources above. The table label budget overrides the release CLI default of 1000. The unresolved architecture search remains explicit.'
            }.get(file,'Read the operations used by the reproduction runner. This is executable source, not a link to an unseen model.'))
        dest='paper-run/'+file
        code(f"Path({str(Path(dest).parent)!r}).mkdir(parents=True,exist_ok=True)")
        if file.endswith('.py'):
            for i,(title,chunk) in enumerate(source_chunks(contents)):
                md('#### '+title)
                code('%%writefile '+('-a ' if i else '')+dest+'\n'+chunk)
        else:
            # Numeric target/config files are peripheral; model/trainer code above is plain visible Python.
            code(f'# PROVIDED: pinned configuration / reference data\nPath({dest!r}).write_text({contents!r})')
    md(paper_evidence(lesson))
    md('### RUN · declared reproduction target\n\nThe full run is gated because it can take hours. The default smoke mode verifies execution only. Save the JSON record and diagnose any discrepancy before claiming reproduction. Keep the paper score and your measured score in separate columns.')
    if lesson==71:
        code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    subprocess.run([sys.executable,str(PAPER_ROOT/'colab_vime.py')],check=True)\nelse:\n    print('Historical VIME release run NOT_RUN in this kernel; unpublished architecture selection remains unresolved.')")
    elif lesson in (72,73):
        code("RUN_PAPER_REPRO=False\nPAPER_SMOKE=True  # Set False for 30 splits, full rows and paper stopping limits.\nif RUN_PAPER_REPRO:\n    device='cuda' if torch.cuda.is_available() else 'cpu'\n    command=[sys.executable,str(PAPER_ROOT/'scarf.py'),'--device',device,'--output',str(PAPER_ROOT/'scarf-results.json')]\n    if PAPER_SMOKE:command.append('--smoke')\n    subprocess.run(command,check=True)\nelse:\n    print('SCARF Tables 2/4 run NOT_RUN in this kernel.')")
        if lesson==72:
            code("if RUN_PAPER_REPRO:\n    command=[sys.executable,str(PAPER_ROOT/'subtab.py'),'--device',device,'--output',str(PAPER_ROOT/'subtab-results.json')]\n    if PAPER_SMOKE:command.append('--smoke')\n    subprocess.run(command,check=True)")
    else:
        code("RUN_PAPER_REPRO=False\nPAPER_SMOKE=True\nFASTTEXT_PATH=Path('/content/cc.en.300.bin')  # full model, not the teaching vector cache\nCHECKPOINT_PATH=Path('data/l074/kg_pretrained.pt').resolve()\nif RUN_PAPER_REPRO:\n    command=[sys.executable,str(PAPER_ROOT/'carte.py'),'--fasttext',str(FASTTEXT_PATH),'--checkpoint',str(CHECKPOINT_PATH),'--output',str(PAPER_ROOT/'carte-results.json')]\n    if PAPER_SMOKE:command.append('--smoke')\n    subprocess.run(command,check=True)\nelse:\n    print('CARTE released downstream benchmark NOT_RUN in this kernel; fresh YAGO pretraining NOT_RUN.')")
    md('### Reproduction exit ticket\n\nSubmit the immutable configuration, data/source hashes, per-seed scores, reference-score gaps, and the execution log. Identify every remaining mismatch. Explain which published row was attempted, which was measured, and whether any missing detail prevents a parity claim. A file named `paper-results.json` is not by itself evidence of reproduction.')
    return cells
