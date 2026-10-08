"""Remove opening recall tests, preserving prerequisites and in-lesson exercises.

User preference, 2026-10-07. Applied after legacy lesson builders by the shared
publication pass; no browser-only hiding and no dependency outside the stdlib.
"""
from pathlib import Path
import hashlib,html,json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
HEADING=re.compile(r'^(?:warm[ -]up\b|retrieve (?:before|first|the)\b|retrieval before\b|start with retrieval\b|start cold\b|cold retrieval\b|recall (?:before|the)\b|first retrieve\b|close the notes\b|first recall\b|before reading: retrieve\b)',re.I)
PROMPT=re.compile(r'^(?:without\b|close (?:your |the |L\d|B\d)|before (?:reading|looking|opening|you read)\b|(?:first[, :]+)?(?:write|answer|retrieve|recall|explain|name)\b|three questions\b|a few questions\b|one extra cold question\b|from (?:L\d|B\d|lesson|earlier)|why\b|which\b|what\b|how\b|keep these answers\b|try (?:these|the)\b)',re.I)
WIDGET=r'<div\b[^>]*\bid=["\x27](?:[a-z0-9]+-)?warmup["\x27][^>]*>\s*</div>'

def plain(s):return html.unescape(re.sub('<[^>]*>','',s)).strip()
def review_heading(s):return bool(HEADING.match(re.sub(r'^[\d. ·:—–-]+','',plain(s))))
def reminders(s):
    replacements={
        'Work the cold retrieval first, then spend':'Spend',
        'Start with retrieval, then':'Start with',
        'retrieval, arithmetic and the mechanism':'arithmetic and the mechanism',
        'Cold retrieval → ':'',
        'protocol and cold recall':'protocol',
        'Start with retrieval; do not read the answers first.':'Start with the worked example.',
        'first complete the cold questions; then trace':'first trace',
        'retrieval and the worked trace first':'the worked trace first',
        'attempt retrieval first':'use them when needed',
        '10 minutes cold retrieval, ':'',
    }
    for before,after in replacements.items():s=s.replace(before,after)
    return s

INLINE=re.compile(r'^(?:retrieve (?:before|first|without|three prerequisites)|recall (?:before|grouped|without notes)|cold retrieval|close your notes|without (?:looking back|notes))',re.I)

def inline_review(s):
    clean=plain(s).lstrip('* ')
    if not INLINE.match(clean):return s
    # Some introductions attach real prerequisites to their questions.
    match=re.search(r'(?:<strong>|\*\*)?Prerequisites?\s*[:.]',s,re.I)
    if match:return s[match.start():]
    if '?' in clean or re.search(r'answer|in writing|write one|before reading onward',clean,re.I):return ''
    # Definition-only reminders remain useful: remove the testing instruction.
    return re.sub(r'^(?:<strong>|\*\*)?(?:Retrieve three prerequisites first|Recall before reading|Recall before tracing)[.:](?:</strong>|\*\*)?\s*','Prerequisites. ',s,flags=re.I)

RESIDUAL_OPENING_PROMPTS = ['Close the references and answer: a graph predictor sees customers and their orders; a tree sees only customer age. The graph predictor scores higher. Have we isolated the value of graph computation?', 'Before reading further, answer from memory: which records are legal at a prediction cutoff? Does an old event date guarantee that its label was known? Can a full aggregate answer every future question about its source rows?', 'Recall B06: what must stay fixed when comparing priors? Recall L074: what do row-graph edges represent? Recall B05: may a query label enter its retrieved context?']

def remove_residual_openings(text):
    # Exact legacy prompts: do not classify every prediction exercise as a review.
    if RESIDUAL_OPENING_PROMPTS[0] in text:
        text=text.replace('1 · Predict before reading','1 · Define the comparison unit')
        text=re.sub(r'<details><summary>Check your explanation</summary><p>No\. The graph predictor also received additional information\..*?</p></details>', '', text, flags=re.S)
    for prompt in RESIDUAL_OPENING_PROMPTS:
        text=text.replace('<p>'+prompt+'</p>','').replace(prompt,'')
    return re.sub(r'<noscript>\s*</noscript>', '', text)

def strip_html(text):
    text=remove_residual_openings(text)
    # These sections contain only the opening review, not model retrieval.
    text=re.sub(r'<section\b(?=[^>]*\bid=["\x27](?:retrieval|recall)["\x27])[^>]*>.*?</section>', '',text,flags=re.S|re.I)
    heads=list(re.finditer(r'<h([123])\b[^>]*>(.*?)</h\1>',text,re.S|re.I))
    edits=[]
    for i,h in enumerate(heads):
        if not review_heading(h[2]):continue
        end=heads[i+1].start() if i+1<len(heads) else len(text)
        block=text[h.end():end]
        block=re.sub(WIDGET,'',block,flags=re.I)
        # Strip only questions and their answers. Keep introductory explanations,
        # mission links, learning objectives, diagrams and worked prerequisites.
        def paragraph(m):
            return '' if PROMPT.match(plain(m[0])) else m[0]
        block=re.sub(r'<p\b[^>]*>.*?</p>\s*(?:<(?:ol|ul)\b[^>]*>.*?</(?:ol|ul)>)?',paragraph,block,flags=re.S|re.I)
        def detail(m):
            summary=re.search(r'<summary\b[^>]*>(.*?)</summary>',m[0],re.S|re.I)
            return '' if summary and re.search(r'(?i)recall|retriev|cold|answer|compare|check',plain(summary[1])) else m[0]
        block=re.sub(r'<details\b[^>]*>.*?</details>',detail,block,flags=re.S|re.I)
        start=h.start();line_start=text.rfind('\n',0,start)+1
        if not text[line_start:start].strip():start=line_start
        edits.append((start,end,block))
    for a,b,value in reversed(edits):text=text[:a]+value+text[b:]
    text=re.sub(WIDGET,'',text,flags=re.I)
    def opening_paragraph(m):
        prefix,body,suffix=m[1],m[2],m[3]
        revised=inline_review(body)
        return prefix+revised+suffix if revised else ''
    text=re.sub(r'(<p\b[^>]*>)(.*?)(</p>\s*(?:<(?:ol|ul)\b[^>]*>.*?</(?:ol|ul)>)?)',opening_paragraph,text,flags=re.S|re.I)
    return reminders(text)

def strip_markdown(text):
    text=remove_residual_openings(text)
    text=re.sub(r'\[\[WARMUP\]\]\s*','',text)
    heads=list(re.finditer(r'^#{1,2} (.+)$',text,re.M));edits=[]
    for i,h in enumerate(heads):
        if not review_heading(h[1]):continue
        end=heads[i+1].start() if i+1<len(heads) else len(text)
        parts=re.split(r'(\n\s*\n)',text[h.end():end]);drop_list=False
        for j in range(0,len(parts),2):
            part=parts[j];raw=plain(part).lstrip('* ')
            if PROMPT.match(raw):parts[j]='';drop_list=True
            elif drop_list and re.match(r'\s*(?:\d+[.)]|[-*]) ',part):parts[j]=''
            elif raw:drop_list=False
            if re.search(r'<summary[^>]*>[^<]*(?:recall|answers?|retriev|cold)',part,re.I):parts[j]=''
        edits.append((h.start(),end,''.join(parts)))
    for a,b,value in reversed(edits):text=text[:a]+value+text[b:]
    parts=re.split(r'(\n\s*\n)',text)
    drop=False
    for i in range(0,len(parts),2):
        old=parts[i];new=inline_review(old.strip())
        if new!=old.strip():parts[i]=new;drop=not new
        elif drop and re.match(r'\s*(?:\d+[.)]|[-*]) ',old):parts[i]=''
        elif old.strip():drop=False
    return reminders(strip_html(''.join(parts))).lstrip('\n')

def outputs():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0')
    hashes={}
    for name in tracked:
        p=ROOT/name
        if name.startswith('lessons/content/') and p.suffix=='.md':
            yield p,strip_markdown(p.read_text())
        elif name.startswith('labs/html/') and p.suffix=='.html':
            yield p,strip_html(p.read_text())
        elif name.startswith('labs/') and p.suffix=='.ipynb':
            original=p.read_text();nb=json.loads(original);changed=False
            for c in nb['cells']:
                if c['cell_type']!='markdown':continue
                source=c.get('source',[]);old=''.join(source) if isinstance(source,list) else source
                new=strip_markdown(old)
                if new!=old:c['source']=new.splitlines(keepends=True) if isinstance(source,list) else new;changed=True
            if changed:
                indent=1 if re.search(r'^ \"',original,re.M) else 2
                value=json.dumps(nb,ensure_ascii=False,indent=indent)+'\n'
                hashes[hashlib.sha256(original.encode()).hexdigest()]=hashlib.sha256(value.encode()).hexdigest()
                yield p,value
    for p in (ROOT/'labs').glob('_execution*results.json'):
        original=p.read_text();data=json.loads(original)
        if isinstance(data,dict) and data.get('notebook_sha256') in hashes:
            data['notebook_sha256']=hashes[data['notebook_sha256']]
            yield p,json.dumps(data,indent=2)+'\n'


if __name__=='__main__':
    changed=[]
    for p,value in outputs():
        if p.read_text()!=value:p.write_text(value);changed.append(str(p.relative_to(ROOT)))
    for p in (ROOT/'lessons').glob('*.html'):
        before=p.read_text();after=strip_html(before)
        if before!=after:p.write_text(after);changed.append(str(p.relative_to(ROOT)))
    print(json.dumps({'changed':len(changed),'files':changed},indent=2))
