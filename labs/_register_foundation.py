"""Idempotently register the authored sequence without claiming learner mastery."""
import json,re,html
from pathlib import Path
from _foundation_config import SLUGS,TITLES,PRIMARY,PREDICT
ROOT=Path(__file__).resolve().parents[1]

def replace_block(path,name,text):
    path=ROOT/path;s=path.read_text();begin=f'<!-- {name}:begin -->';end=f'<!-- {name}:end -->'
    block=begin+'\n'+text.strip()+'\n'+end
    if begin in s:s=re.sub(re.escape(begin)+r'.*?'+re.escape(end),lambda _:block,s,flags=re.S)
    else:s=s.rstrip()+'\n\n'+block+'\n'
    path.write_text(s)


def register():
    p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text());existing={r['id']:r for r in m['lessons']}
    changed=any(n not in existing for n in range(58,71))
    for n in range(58,71):
        slug=f'{n:04}-{SLUGS[n]}'
        existing[n]=dict(id=n,slug=slug,year=2,quarter=2 if n<=60 else 3,checkpoint=n in [60,70],
                         labPath=f'labs/{slug}.ipynb',title=TITLES[n],published=True)
        (ROOT/f'labs/_build_l{n:03}.py').write_text(f'"""Regenerate the complete L{n:03} lesson package."""\nfrom _build_foundation import build\nif __name__ == "__main__": build({n})\n')
        (ROOT/f'labs/_verify_l{n:03}.py').write_text(f'"""Regenerate declared L{n:03} evidence into a fresh, reviewable output."""\nfrom _run_foundation import run\nif __name__ == "__main__": print(run({n}))\n')
    (ROOT/'labs/_verify_l070.py').write_text('"""Rebuild both version panels and assemble the complete L070 checkpoint."""\nfrom _run_foundation import run, ROOT\nfrom _assemble_foundation_checkpoint import assemble\nif __name__ == "__main__":\n    historical=run(70,output=ROOT/\'data/cache/foundation/l070-historical-rerun.json\')\n    current=run(70,current=True,output=ROOT/\'data/cache/foundation/l070-current-rerun.json\')\n    out=ROOT/\'data/cache/foundation/l070-complete-rerun.json\'\n    assemble(historical,current,out)\n    print(out)\n')
    m['lessons']=sorted(existing.values(),key=lambda r:r['id']);m['version']+=int(changed);p.write_text(json.dumps(m,indent=2)+'\n')
    for name in ['index.html','notebooks.html']:
        p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda a:a[1]+str(m['version']),s);p.write_text(s)
    (ROOT/'learning-records/0123-lesson-057-completion-reported.md').write_text('# Lesson 057 completion reported\n\nOn 2026-09-09 the user reported lesson 057 complete and requested lessons 058–070, with a quality bar beyond preceding packages, followed by a push to main. Completion is self-reported and unscored: no EXIT artifact or retrieval answers were supplied. The new sequence is prepared learning material, not evidence of mastery; separate lettered extension units are not marked complete.\n')
    p=ROOT/'lessons/0057-cross-family-ensembling.html';s=p.read_text();s=s.replace('Next integer lesson: L058, surveys and meta-benchmarks.','Next integer lesson: <a href="0058-surveys-meta-benchmarks.html">L058, surveys and meta-benchmarks</a>.');p.write_text(s)
    replace_block('NOTES.md','FOUNDATION-058-070','''## Lesson 057 complete; lessons 058–070 prepared — 2026-09-09

- User reports L057 done; LR-0123 records self-reported, unscored completion. Mission and personal mastery glossary unchanged.
- User requested the full integer sequence and repeatedly raised the quality bar. All units ship source-grounded manuscripts, printable references, portable computation figures, live TODO/CHECK/EXIT notebooks, executed local solutions, prepared HTML, provenance, reproduction contracts and runnable follow-up operators.
- L058 reanalyzes 300 frozen TALENT tasks; L059 runs a 200-trial adaptive-selection negative control; L060 compares five families on eleven underlying datasets, separating random and temporal regimes.
- L061 trains three CountPFNs against an analytic posterior; L062 runs the historical v1 checkpoint; L063 samples/intervenes on SCMs; L064 audits historical v2; L065 measures honest query-role embedding heads; L066 measures TabICL context scaling; L067 actually fine-tunes copied v1 weights; L068 trains a matched synthetic prior ablation; L069 measures fixed-model corruptions; L070 compares explicit checkpoint versions.
- Reduced model/algorithm code is distinct from pretrained reference inference. Full original paper pretraining and benchmark reproduction remain NOT_ESTABLISHED. See per-lesson evidence/contract files for exact omissions and local measurements.
- Delivery status is machine-recorded in labs/_execution_foundation_results.json, labs/_source_check_foundation_results.json and labs/_delivery_foundation_results.json. Local rendering, live Colab and remote publication are separate checks.
- Next learning unit is L058, not L071: these thirteen authored lessons are available ahead of study, not completed by the learner.
''')
    rows=['## Lessons 058–070 · primary reading and exact evidence contracts']
    for n in range(58,71):rows.append(f'- **L{n:03}:** [{PRIMARY[n][0]}]({PRIMARY[n][1]}). [Local scope and regeneration](labs/l{n:03}-reproduction.md).')
    replace_block('RESOURCES.md','FOUNDATION-058-070','\n'.join(rows))
    replace_block('labs/README.md','FOUNDATION-058-070','''## Lessons 058–070 · benchmark evidence and tabular foundation models

Each integer unit has a standalone student notebook and prepared HTML. Start at L058 and
follow retrieval → input → live implementation → CHECK → evidence audit → EXIT. The tutor
reviews the submitted artifact and explanation; completed author runs do not establish learner mastery.

`_run_foundation.py --lesson N --preset smoke|lab|closer --output PATH` regenerates the
declared experiment. `--lesson 70 --current` runs explicitly pinned current-version arms.
Historical packages use isolated subprocess imports. Full original paper protocols are not
implemented; `--preset paper` fails honestly. See `lNNN-reproduction.md`, `_sources_foundation.json`
and `_delivery_foundation_results.json`. Student TODOs remain blank; local solutions follow the
workspace's ignored `labs/solutions/` convention.
''')
    ledger=[]
    for n in range(58,71):
        ledger.append(f'### L{n:03} — BAR: {TITLES[n]}\n\n{PREDICT[n][1]} The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/{n:04}-{SLUGS[n]}.html) · [measured evidence](labs/_verify_l{n:03}_results.json) · [scope](labs/l{n:03}-reproduction.md).')
    replace_block('thesis-dossier.md','FOUNDATION-058-070','\n\n'.join(ledger))
    links='\n'.join(f'| {n:03} | [Lesson](lessons/{n:04}-{SLUGS[n]}.html) | [Lab](labs/{n:04}-{SLUGS[n]}.ipynb) | [Evidence contract](labs/l{n:03}-reproduction.md) |' for n in range(58,71))
    replace_block('CURRICULUM.md','FOUNDATION-058-070','## Prepared lessons 058–070\n\nThese are available learning packages; only L057 completion was reported by the learner. Lettered extension units remain separate.\n\n| Unit | Lesson | Lab | Protocol |\n|---|---|---|---|\n'+links)
    p=ROOT/'reference/glossary.html';s=p.read_text();section='<section id="foundation-contracts"><h2>PFNs and evidence contracts (L058–070)</h2><table><tbody>'
    terms={'PFN':'A network trained across prior-sampled tasks to approximate context-conditioned prediction.',
    'Posterior predictive':'A query-label distribution obtained by averaging over the posterior uncertainty about a task.',
    'SCM':'Structural assignments from parent variables and exogenous noise, with a declared dependency graph.',
    'Query-role embedding':'A representation computed without that row\'s own target in its labeled context.',
    'Inducing vectors':'Learned queries that compress a set into a fixed number of attention summaries.',
    'Selection optimism':'A selected validation estimate that appears better than independent evaluation because decisions fit its noise.',
    'Second-order SCM':'A generator whose outputs parameterize changing mechanisms in another SCM.',
    'Open-environment contract':'The declared schema, class support, target semantics and availability assumptions a prediction must satisfy.'}
    section+=''.join(f'<tr><td class="term">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k,v in terms.items())+'</tbody></table></section>'
    if 'id="foundation-contracts"' in s:s=re.sub(r'<section id="foundation-contracts">.*?</section>',lambda _:section,s,flags=re.S)
    else:s=s.replace('</article>',section+'</article>')
    p.write_text(s)
    # Stable question IDs; equal two-word answer alternatives.
    facts={58:('What is the independent aggregation unit in the benchmark bootstrap?',['Dataset rows','Model repeats','Epoch checkpoints'],0),
    59:('What fits when repeated choices use validation labels?',['Selection procedure','Untouched evaluation','Fixed population'],0),
    60:('What should be averaged before cross-dataset model ranking?',['Training durations','Model seeds','Feature values'],1),
    61:('What must context and query share in the coin-task generator?',['Observed labels','Random draws','Latent probability'],2),
    62:('What may supply row-attention keys in the reduced inductive PFN?',['Context rows','Query targets','Future labels'],0),
    63:('What stays fixed in a paired edge-mechanism intervention?',['Descendant values','Exogenous noise','Edge weights'],1),
    64:('Which representation feeds the reduced v2 classification head?',['Context features','All features','Query target'],2),
    65:('Which role should a training row have during cross-fitted representation extraction?',['Unlabeled query','Labeled context','Scored prediction'],0),
    66:('Which stage still has quadratic context dependence in TabICL?',['Column compression','Dataset attention','Scalar encoding'],1),
    67:('What distinguishes fine-tuning from context retrieval?',['Changed neighbors','Changed distances','Updated weights'],2),
    68:('Besides event time, what controls labeled-context eligibility?',['Label availability','Future performance','Dataset size'],0),
    69:('What must remain visible when classes are unsupported?',['Selected accuracy','Population coverage','Training speed'],1),
    70:('What identifies a version-specific pretrained model?',['Package default','Display title','Checkpoint identity'],2)}
    pool=[]
    for n,(question,options,correct) in facts.items():pool.append(dict(id=f'l{n:03}-foundation-contract',lesson=n,quarter='Q2' if n<=60 else 'Q3',concept=f'foundation-{SLUGS[n]}',question=question,options=[dict(label=o,value=str(i)) for i,o in enumerate(options)],correct=str(correct),explain=PREDICT[n][1]))
    for filename,items in [('assets/retrieval-pool.js',pool),('assets/paper-deck.js',[dict(id=f'foundation-paper-{n}',paper=PRIMARY[n][0],year=2022 if n in [61,62] else 2025,lesson=n,front=PREDICT[n][0],back=PREDICT[n][1]) for n in [61,62,64,66]])]:
        p=ROOT/filename;s=p.read_text();marker='// FOUNDATION-SEQUENCE-058-070'
        if marker in s:s=s[:s.index(marker)].rstrip().rstrip(',')+'\n];\n})(window);\n'
        at=s.rfind('];');s=s[:at].rstrip()+',\n'+marker+'\n'+',\n'.join(json.dumps(v,indent=2) for v in items)+'\n'+s[at:];p.write_text(s)

if __name__=='__main__':register()
