"""Independent SQL oracle and adversarial contracts for the schema graph lab."""
import copy, hashlib, importlib.util, json, random, sys, unittest
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.find_spec('relkit.schema_l096')

class SchemaContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(spec, 'L096 schema implementation is missing')
        from relkit import schema_l096
        self.m=schema_l096
    def test_full_fixture(self):
        m=self.m; tables=m.fixture();g=m.build_graph(tables)
        self.assertEqual({t:g[t].num_nodes for t in g.node_types},dict(customer=3,product=3,orders=3,line_item=4))
        self.assertEqual(sum(g[e].edge_index.shape[1] for e in g.edge_types),24)
        self.assertEqual(m.graph_query(g),m.sql_query(tables))
        self.assertEqual(m.graph_query(g)['totals'],[[10,10,5],[10,50,1],[30,50,4]])
        self.assertEqual(len(m.graph_query(g)['paths']),4)
    def test_all_declared_fks_and_reverse(self):
        m=self.m;t=m.fixture();g=m.build_graph(t)
        for child,role,cols,parent,parentcols,nullable in m.FKS:
            edge=(child,role,parent); rev=(parent,'rev_'+child+'_'+role,child)
            self.assertTrue(m.torch.equal(g[edge].edge_index.flip(0),g[rev].edge_index))
            expected=[]
            for row in sorted(t[child],key=lambda r:tuple(r[c] for c in m.PKS[child])):
                key=tuple(row[c] for c in cols)
                if any(x is None for x in key):continue
                expected.append([list(tuple(row[c] for c in m.PKS[child])),list(key)])
            actual=[[list(g[child].primary_keys[a]),list(g[parent].primary_keys[b])] for a,b in g[edge].edge_index.t().tolist()]
            self.assertEqual(actual,expected)
    def test_invalid_keys(self):
        m=self.m
        for kind in ['duplicate','null','orphan','required_null']:
            t=m.fixture()
            if kind=='duplicate':t['line_item'].append(dict(t['line_item'][0]))
            if kind=='null':t['line_item'][0]['line_no']=None
            if kind=='orphan':t['orders'][0]['buyer_id']=999
            if kind=='required_null':t['orders'][0]['buyer_id']=None
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.build_graph(t)
    def test_nullable_composite_and_orphan_distinction(self):
        m=self.m
        parent={(7,2):0}; rows=[{'a':None,'b':9},{'a':7,'b':2}]
        self.assertEqual(m.fk_edges(rows,('a','b'),parent,True).tolist(),[[1],[0]])
        with self.assertRaises(ValueError):m.fk_edges([{'a':7,'b':3}],('a','b'),parent,True)
    def test_empty_relations_and_row_order(self):
        m=self.m;t=m.fixture();baseline=m.graph_query(m.build_graph(t))
        rng=random.Random(96)
        for _ in range(20):
            for rows in t.values():rng.shuffle(rows)
            self.assertEqual(m.graph_query(m.build_graph(t)),baseline)
        t['line_item']=[];g=m.build_graph(t)
        self.assertEqual(g['line_item','product','product'].edge_index.shape,(2,0))
        self.assertEqual(m.graph_query(g),m.sql_query(t))
    def test_full_generated_suite_sql_oracle(self):
        m=self.m
        for seed in range(32):
            t=m.generated_fixture(seed)
            with self.subTest(seed=seed):self.assertEqual(m.graph_query(m.build_graph(t)),m.sql_query(t))
    def test_intervention_preserves_events(self):
        m=self.m;t=m.fixture();g=m.build_graph(t)
        a=m.graph_query(g)
        t['line_item'][1]['product_id']=50
        b=m.graph_query(m.build_graph(t))
        self.assertEqual(b,m.sql_query(t))
        self.assertEqual(a['totals'],[[10,10,5],[10,50,1],[30,50,4]])
        self.assertEqual(b['totals'],[[10,10,2],[10,50,4],[30,50,4]])
        self.assertEqual(len(a['paths']),len(b['paths']))
    def test_sql_rejects_corruption(self):
        m=self.m;t=m.fixture();t['orders'][0]['buyer_id']=999
        with self.assertRaises(m.sqlite3.IntegrityError):m.sql_query(t)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SchemaContract))
    if result.wasSuccessful():
        report={'status':'PASS','tests':result.testsRun,'generated_databases':32,'row_permutations':20,'oracle':'Independent SQLite joins, path rows, aggregate totals and LEFT JOIN counts','source_sha256':hashlib.sha256((P/'relkit/schema_l096.py').read_bytes()).hexdigest()}
        (P/'_verify_l096_results.json').write_text(json.dumps(report,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
