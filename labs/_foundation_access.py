"""Static entry points for complete lesson packages, shared by the authoring builder."""
import html
import json
from pathlib import Path
from _foundation_config import SLUGS,TITLES,TASKS

ROOT=Path(__file__).resolve().parents[1]
PRACTICE={
58:'Parse means and SDs, rank the declared panel, bootstrap paired datasets, and select a tiny benchmark without seeing held-out method scores; compare 1,000 and 10,000 proposals.',
59:'Replay the noise control, implement analytic leave-one-out KRR and nested selection, and compare 30 versus 1,000 independent synthetic repetitions.',
60:'Fit a corrected five-family checkpoint, verify row identity and prediction metrics, select on validation, and audit the complete declared dataset and seed panels.',
61:'Train a count-based PFN on sampled coin tasks and compare its predictions with the exact Bayesian posterior.',
62:'Implement attention and its information mask; check a reduced row-token PFN on real rows. Historical pretrained v1 inference is a separate rerun.',
63:'Evaluate a synthetic causal graph, remove an edge with noise held fixed, and test which descendants change.',
64:'Implement the sample-axis operation of a reduced axial PFN and test query isolation. The saved Nature-v2 comparison uses separate pretrained weights.',
65:'Extract cross-fitted query embeddings and fit a linear head. The default exercise uses a random-weight encoder; actual v2 inference is a separate rerun.',
66:'Implement inducing attention and distribution-conditioned cell embeddings. The default exercise isolates a column mechanism; pretrained TabICL context scaling is separate.',
67:'Build disjoint retrieved training episodes and run a small optimization loop. The saved comparison separately measures adaptation of historical v1 weights.',
68:'Implement label-availability checks and a changing-mechanism task generator; train a short temporal PFN exercise.',
69:'Score unsupported classes and apply training-derived feature corruptions. The default exercise uses a fitted logistic model; the v2/XGBoost panel is separate.',
70:'Reconstruct the saved predictions and validation selections for all seven model arms, then calculate paired summaries and ranks. The full rerun is explicitly gated.'}
EVIDENCE={
58:'Frozen-result reanalysis: six methods on 300 tasks; a separate 276-task complete panel for tiny-benchmark selection. No new predictive-model fits.',
59:'Historical 200-repeat noise control plus new paper-equation KRR experiments: 30 and 1,000 repeated datasets, with matched fresh evaluation of nested and contaminated procedures.',
60:'Corrected five-family, eleven-dataset checkpoint with three seeds; 210 selected evaluations, with random and temporal regimes kept separate from each other and from historical evidence.',
61:'Three trained count-based PFNs, including checks outside their training context lengths.',
62:'Actual historical v1 checkpoint inference on three datasets and three seeds.',
63:'Paired SCM intervention with fixed exogenous noise.',
64:'The exact v2/XGBoost predictions reused from the lesson 70 historical panel.',
65:'Actual v2 cross-fitted representation heads versus native inference on three tasks and three seeds.',
66:'Actual TabICL v1.1 predictions at context sizes 60, 180, and 540, with three seeds.',
67:'Actual v1 global context, retrieval, and six-step fine-tuning on three tasks and three seeds.',
68:'Matched stationary-prior versus changing-edge-prior PFNs, trained for 400 steps across three seeds.',
69:'Actual v2/XGBoost corruption comparisons on three tasks and three seeds, plus a separate class-support diagnostic.',
70:'105 selected results: seven named model/checkpoint arms × five datasets × three seeds.'}


def link(href,label,primary=False,download=False):
    return f'<a href="{html.escape(href)}"'+(' class="lab-access-primary"' if primary else '')+(' download' if download else '')+f'>{html.escape(label)}</a>'


def launcher(n,prepared=False):
    slug=f'{n:04}-{SLUGS[n]}';colab=f'https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{slug}.ipynb'
    if prepared:
        links=[link(colab,'Run in Colab',True),link('#lab-exercises','Jump to exercises'),
               link(f'../{slug}.ipynb','Download notebook',download=True),link(f'../../lessons/{slug}.html','Lesson'),
               link('foundation-sequence.html', 'All 58–70 materials')]
        intro='This is the read-only lab preview. Run the notebook in Colab or download it for Jupyter to complete the exercises.'
    else:
        links=[link(f'../labs/html/{slug}.html','Open lab',True),link(colab,'Run in Colab'),
               link(f'../labs/{slug}.ipynb','Download notebook',download=True),link(f'../reference/{slug}.html','Reference'),
               link('../labs/html/foundation-sequence.html', 'All 58–70 materials')]
        intro='Read the lab here, or open the runnable notebook in Colab. The student tasks are intentionally blank.'
    return f'<aside class="lab-access" data-lab-launch aria-label="Lesson {n:03} lab and materials"><p><strong>Lab {n:03} · {len(TASKS[n])} code tasks + EXIT</strong></p><nav class="lab-access-links" aria-label="Lab actions">'+''.join(links)+f'</nav><p>{intro}</p></aside>'


def lab_plan(n):
    tasks=''.join(f'<li><code>{name}</code>: {html.escape(goal)}</li>' for name,_,_,goal in TASKS[n])
    exit_path = f'data/cache/l{n:03}-student/exit.json' if n in (58, 59, 60) else f'student-l{n:03}-exit.json'
    return f'<p><strong>What you will do:</strong> {PRACTICE[n]}</p><ol>{tasks}</ol><p><strong>Author-reference evidence:</strong> {EVIDENCE[n]}</p><p><strong>Submit:</strong> your completed notebook and <code>{exit_path}</code>, plus your written interpretation. CHECK cells give immediate code feedback; paste the EXIT output here for a reasoning review.</p>'


def build_directory():
    cards=[]
    for n in range(58,71):
        slug=f'{n:04}-{SLUGS[n]}'
        links=[link(f'../../lessons/{slug}.html','Lesson'),link(f'{slug}.html','Read lab',True),
               link(f'https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{slug}.ipynb','Run in Colab'),
               link(f'../{slug}.ipynb','Download notebook',download=True),link(f'../../reference/{slug}.html','Reference'),
               link(f'../l{n:03}-reproduction.md','Reproduction instructions'),link(f'../_verify_l{n:03}_results.json','Measured results')]
        if n == 60:
            links.append(link('../_verify_l060_v2_results.json','Corrected checkpoint results'))
        if n == 59:
            links.append(link('../_verify_l059_v2_results.json','Nested KRR results'))
            links.append(link('../_verify_l059_closer_results.json','1,000-repeat results'))
        if n == 58:
            links.append(link('../_verify_l058_v2_results.json','Updated audit and subset results'))
            links.append(link('../_verify_l058_closer_results.json','10,000-candidate results'))
        cards.append(f'<section class="lab-package" id="lesson-{n}"><h2>{n:03} · {TITLES[n]}</h2><nav class="lab-access-links" aria-label="Lesson {n:03} materials">'+''.join(links)+lab_plan(n)+'</section>')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lessons 58–70 · labs and study materials</title><link rel="stylesheet" href="../../assets/lesson.css"><link rel="stylesheet" href="../../assets/lab-access.css"></head><body><article>
<nav><a href="../../index.html">Course</a> · <a href="../../notebooks.html">All notebooks</a></nav>
<p class="mission-tag">Study guide · lessons 058–070</p><h1>Labs and study materials</h1>
<p>All thirteen labs, with their lessons, references, measured results, and reproduction instructions.</p>
<nav class="lab-access-links" aria-label="Start practicing"><a class="lab-access-primary" href="0058-surveys-meta-benchmarks.html">Open lab 58</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/0058-surveys-meta-benchmarks.ipynb">Run lab 58 in Colab</a></nav>
<details><summary>How to use the labs and interpret their results</summary>
<ol><li><strong>Retrieve and read:</strong> answer the lesson's opening prompt before checking its explanation.</li><li><strong>Practice:</strong> choose Run in Colab or download the notebook for Jupyter. Complete TODOs, run each CHECK, and inspect the live experiment.</li><li><strong>Explain:</strong> write the EXIT verdict and submit its artifact for review. Use the reference later for spaced recall.</li></ol>
<p><strong>What is runnable?</strong> The default notebooks contain the stated practice experiments. Pretrained inference and larger runs are explicitly gated and may download weights. Saved author results are separate from what runs in your notebook; each entry spells out that distinction. Full original paper benchmark reproduction remains unestablished.</p></details>
<nav class="lab-access-links" aria-label="Jump to lesson">'''+''.join(link(f'#lesson-{n}',f'{n:03}') for n in range(58,71))+'</nav>'+''.join(cards)+'''
<section><h2>Source code and verification</h2><p><a href="../_sources_foundation.json">Checkpoint and source identities</a> · <a href="../relkit/foundation_core.py">Model mechanisms</a> · <a href="../relkit/benchmark_core.py">Evaluation algorithms</a> · <a href="../_execution_foundation_results.json">Solution execution record</a> · <a href="../_delivery_foundation_results.json">Artifact checks</a></p><p>Teacher solution notebooks are kept locally under <code>labs/solutions/</code> under the course convention; downloadable student notebooks keep their TODOs blank. Ask for help with a specific CHECK or paste your EXIT output for review.</p></section></article></body></html>'''
    (ROOT/'labs/html/foundation-sequence.html').write_text(page)


def build_static_gallery():
    """Keep the existing gallery usable even when fetch or JavaScript is unavailable."""
    from bs4 import BeautifulSoup
    path=ROOT/'notebooks.html';soup=BeautifulSoup(path.read_text(),'html.parser')
    host=soup.find(id='nb-list');host.clear();ul=soup.new_tag('ul',attrs={'class':'nb-gallery'});host.append(ul)
    lessons=json.loads((ROOT/'lessons/manifest.json').read_text())['lessons']
    for item in sorted(lessons,key=lambda v:v['id'],reverse=True):
        if not item.get('published') or not item.get('labPath'):continue
        n=item['id'];lab=item['labPath'];slug=item['slug']
        markup=f'<li id="lab-{n}"><div class="nb-head"><span class="num">Lesson {n:04}</span><span class="title">{html.escape(item["title"])}</span></div><div class="nb-links">'+link('labs/html/'+Path(lab).stem+'.html','Read lab')+link('https://colab.research.google.com/github/Avistian/relational/blob/main/'+lab,'Run in Colab',True)+link(lab,'Download notebook',download=True)+link('lessons/'+slug+'.html','Lesson')+'</div></li>'
        ul.append(BeautifulSoup(markup,'html.parser').li)
    if not soup.find(id='foundation-materials'):
        banner=BeautifulSoup('<p id="foundation-materials" class="ask-teacher"><strong>Lessons 58–70:</strong> <a href="labs/html/foundation-sequence.html">Open the complete lab and materials directory</a> for exercises, references, results, and reproduction instructions.</p>','html.parser').p
        host.insert_before(banner)
    banner=soup.find(id='foundation-materials').extract()
    soup.find('h1').insert_after(banner)
    path.write_text(str(soup))
