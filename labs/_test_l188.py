"""Behavior contracts: fixture examples are synthetic, never literature evidence."""
import unittest
from relkit.literature_l188 import canonical_id, coverage, triage

class Contracts(unittest.TestCase):
    def test_identity(self):
        for s in ['2607.12345v2','https://arxiv.org/abs/2607.12345v12','http://arxiv.org/pdf/2607.12345v1.pdf']:
            self.assertEqual(canonical_id(s),'2607.12345')
        self.assertEqual(canonical_id('hep-th/9901001v2'),'hep-th/9901001')
        for s in ['https://evil.example/2607.12345','2600.12345','2607.12345garbage','foo']:
            with self.assertRaises(ValueError):canonical_id(s)
    def test_pagination(self):
        a={'start':0,'total':3,'ids':['2607.10001','2607.10002']}
        b={'start':2,'total':3,'ids':['2607.10003']}
        self.assertEqual(coverage([a,b]),'COMPLETE')
        self.assertEqual(coverage([b,a]),'COMPLETE')
        for pages in [[],[a],[a,dict(b,start=3)],[a,dict(b,total=4)],[a,dict(b,ids=['2607.10001'])],[a,dict(b,ids=[])],[dict(a,error='HTTP 502')]]:
            self.assertEqual(coverage(pages),'INCOMPLETE')
        self.assertEqual(coverage([{'start':0,'total':0,'ids':[]}]),'COMPLETE')
    def test_triage(self):
        r=dict(verified=True,in_window=True,reviewed=True,relevance='baseline',evidence='reported',reason='Required comparator')
        self.assertEqual(triage(r),'INCLUDE')
        self.assertEqual(triage(dict(r,relevance='incremental')),'EXCLUDE')
        for changes in [dict(verified=False),dict(reviewed=False),dict(reason=''),dict(evidence='unknown')]:
            self.assertEqual(triage(dict(r,**changes)),'DEFER')
        self.assertEqual(triage(dict(r,in_window=False)),'EXCLUDE')

if __name__=='__main__':unittest.main()
