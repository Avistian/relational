"""Audit gate for EXAONE paper Figure 3 regression frontier; never substitutes diabetes."""
import json,argparse
from pathlib import Path
P=Path(__file__).resolve().parent
GAPS=['publication-era checkpoint/code mapping for every plotted variant','complete original outer-fold query/support identities and preprocessing','original measured hardware, precision, batching, warmup and caching per method','original TabArena baseline timing records and complete Elo comparison pool','regression-only timing aggregation matching Figure 3 rather than overall leaderboard cost']
def audit():
 r={'target':'EXAONE 2608.25774v1 Figure 3(b) regression accuracy-latency frontier','status':'INCOMPLETE_SOURCE_PROTOCOL','gaps':GAPS,'fresh_course':'B09-MATCHED-COST is a different experiment','full_pretraining':'NOT_RUN','historical_frontier_execution':'NOT_RUN','published_reference':'https://arxiv.org/html/2608.25774v1#S4'}
 (P/'evidence/b09/paper-gate.json').write_text(json.dumps(r,indent=2));return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args();print(json.dumps(audit(),indent=2))
 if args.execute:raise SystemExit('Refusing historical run: unresolved source protocol; no runnable full-figure reproduction is claimed.')
