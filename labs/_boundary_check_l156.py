"""Exercise the actual comparison AST: historical RED, current GREEN."""
import ast,math
from pathlib import Path
P=Path(__file__).resolve().parent
for file,fixed in [('sources/l156/replay_before_boundary.py',False),('_replay_l156.py',True)]:
    source=(P/file).read_text();tree=ast.parse(source)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Compare) and ast.get_source_segment(source,n).startswith("abs(lanes['paper']['metrics'][s]['mean']-target)"))
    code=compile(ast.Expression(node),'<production-comparison>','eval')
    def close(value,target):return eval(code,dict(math=math,lanes={'paper':{'metrics':{'test':{'mean':value}}}},s='test',target=target))
    for target in (3.193,4.022):
        for value in (target-.2,target+.2):assert bool(close(value,target))==fixed
        for value in (target-.2-1e-9,target+.2+1e-9):assert not close(value,target)
print('PASS: historical endpoints fail, current endpoints pass, values beyond remain outside')
