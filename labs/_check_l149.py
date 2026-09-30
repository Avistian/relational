"""Failure probes for evidence ranking, including hidden protocol mismatches."""
import math
from relkit.weakness_l149 import oriented_gap, rank_catalog

def rejects(fn):
 try: fn()
 except ValueError: return
 raise AssertionError('Invalid evidence was accepted')

def check_gap(fn):
 assert math.isclose(fn(.72,.75,'AUROC'),-.03)
 assert math.isclose(fn(4.,3.,'MAE'),-1.)
 rejects(lambda:fn(float('nan'),3,'MAE'))
 rejects(lambda:fn(4,3,'accuracy'))
 rejects(lambda:fn(72,75,'AUROC'))
 rejects(lambda:fn(-1,3,'MAE'))

def check_rank(fn):
 rows=[dict(task='b',rdl=.75,fe=.8,metric='AUROC',units='fraction',variant='basic',split='test',source='published',status='EXACT'),dict(task='a',rdl=.7,fe=.8,metric='AUROC',units='fraction',variant='basic',split='test',source='published',status='EXACT')]
 assert [r['task'] for r in fn(rows)]==['a','b']
 rejects(lambda:fn(rows+[dict(rows[0],task='c',metric='MAE')]))
 rejects(lambda:fn(rows+[dict(rows[0],task='c',variant='boosted')]))
 rejects(lambda:fn(rows+[dict(rows[0],task='c',source='local')]))
 rejects(lambda:fn(rows+[dict(rows[0],task='c',split='val')]))
 rejects(lambda:fn(rows+[dict(rows[0],task='c',status='MISSING',rdl=None)]))
 rejects(lambda:fn(rows+[rows[0]]))
 rejects(lambda:fn(rows+[dict(rows[0],task='c',units='percentage points')]))
 if fn(rows)[0].get('gap') is None: raise AssertionError('Ranking must show its gap')

if __name__=='__main__':
 check_gap(oriented_gap);check_rank(rank_catalog);print('PASS: gap sign, units, provenance, variants, splits and missing evidence')
