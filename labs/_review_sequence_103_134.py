"""Apply reviewed reading bridges to canonical prose; rebuild with each lesson builder."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]
rows=json.loads((R/'reviews/lessons-103-135/sequence-content.json').read_text())
for key,(bridge,reminder,next_question) in rows.items():
 n=int(key);path=next((R/'lessons/content').glob(f'{n:04d}-*.md'));text=path.read_text()
 text=re.sub(r'<!-- sequence-review:start -->.*?<!-- sequence-review:end -->\s*','',text,flags=re.S)
 text=re.sub(r'\n<!-- sequence-next:start -->.*?<!-- sequence-next:end -->\s*','',text,flags=re.S)
 prev=next((R/'lessons/content').glob(f'{n-1:04d}-*.md'),None)
 if n==112:prev=Path('0111-ogb-benchmark-contract.md')
 previous=f'Lesson {n-1}'
 block=f'''<!-- sequence-review:start -->
<aside class="sequence-context" aria-label="Where this lesson fits">
<p class="sequence-eyebrow">From {previous} to this lesson</p>
<p>{bridge}</p>
<details><summary>Quick prerequisite reminder</summary><p>{reminder}</p></details>
</aside>
<!-- sequence-review:end -->

'''
 following=next((R/'lessons/content').glob(f'{n+1:04d}-*.md'),None)
 if n==110:following=Path('0111-ogb-benchmark-contract.md')
 link=f'[Continue to Lesson {n+1}]({following.stem}.html)' if following else f'Next: Lesson {n+1}'
 text=block+text.rstrip()+f'\n\n<!-- sequence-next:start -->\n**Carry this forward.** {next_question} {link}.\n<!-- sequence-next:end -->\n'
 path.write_text(text)
print('Updated',len(rows),'canonical lesson bridges; lesson 135 is owned by its authoring session.')
