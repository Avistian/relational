"""L096: explicit four-table integer-key schema, graph construction and SQL oracle.

Course fixture, not a published benchmark. No database credentials or repo imports.
Supported keys are non-null Python ints (possibly composite); nullable FKs use
MATCH SIMPLE semantics. This is an explicit specification, not a general SQL parser.
"""
from collections import Counter
import hashlib
import json
import random
import sqlite3
import torch
from torch_geometric.data import HeteroData

PKS = {'customer':('customer_id',), 'product':('product_id',),
       'orders':('order_id',), 'line_item':('order_id','line_no')}
# Child, role, child columns, parent, referenced columns, nullable.
FKS = [('orders','buyer',('buyer_id',),'customer',('customer_id',),False),
       ('orders','referrer',('referrer_id',),'customer',('customer_id',),True),
       ('line_item','order',('order_id',),'orders',('order_id',),False),
       ('line_item','product',('product_id',),'product',('product_id',),False)]
DDL = '''
CREATE TABLE customer(customer_id INTEGER PRIMARY KEY NOT NULL);
CREATE TABLE product(product_id INTEGER PRIMARY KEY NOT NULL);
CREATE TABLE orders(order_id INTEGER PRIMARY KEY NOT NULL,
 buyer_id INTEGER NOT NULL REFERENCES customer(customer_id),
 referrer_id INTEGER REFERENCES customer(customer_id));
CREATE TABLE line_item(order_id INTEGER NOT NULL REFERENCES orders(order_id),
 line_no INTEGER NOT NULL, product_id INTEGER NOT NULL REFERENCES product(product_id),
 quantity INTEGER NOT NULL CHECK(quantity>0), PRIMARY KEY(order_id,line_no));
'''
COLUMNS = {'customer':('customer_id',), 'product':('product_id',),
 'orders':('order_id','buyer_id','referrer_id'),
 'line_item':('order_id','line_no','product_id','quantity')}

def fixture():
    """All 13 rows of the complete hand-traceable course database."""
    return {'customer':[{'customer_id':v} for v in [10,30,90]],
     'product':[{'product_id':v} for v in [10,50,80]],
     'orders':[dict(order_id=100,buyer_id=10,referrer_id=30),
               dict(order_id=200,buyer_id=30,referrer_id=None),
               dict(order_id=300,buyer_id=10,referrer_id=None)],
     'line_item':[dict(order_id=o,line_no=n,product_id=p,quantity=q)
                   for o,n,p,q in [(100,1,10,2),(100,2,10,3),(100,3,50,1),(200,1,50,4)]]}

def key_index(rows, columns):
    """Reject missing/noninteger or duplicate PKs; stable lexicographic local IDs."""
    keys=[tuple(row[c] for c in columns) for row in rows]
    if any(type(v) is not int for key in keys for v in key):
        raise ValueError('Keys must contain non-null Python integers')
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate primary key')
    return {key:i for i,key in enumerate(sorted(keys))}

def fk_edges(rows, columns, parent_index, nullable):
    """Rows are in child-local order. NULL omits an edge; orphan keys fail."""
    pairs=[]
    for child_id,row in enumerate(rows):
        key=tuple(row[c] for c in columns)
        if any(x is None for x in key):
            if not nullable:raise ValueError('NULL in a required foreign key')
            continue
        if any(type(x) is not int for x in key) or key not in parent_index:
            raise ValueError(f'Orphan or invalid foreign key: {key}')
        pairs.append((child_id,parent_index[key]))
    return torch.tensor(pairs,dtype=torch.long).reshape(-1,2).t().contiguous()

def build_graph(tables):
    """One node per row. One role-specific forward and reverse store per FK."""
    if set(tables)!=set(PKS):raise ValueError('Expected exactly the four declared tables')
    indices={t:key_index(tables[t],pk) for t,pk in PKS.items()}
    ordered={t:sorted(tables[t],key=lambda r:tuple(r[c] for c in PKS[t])) for t in PKS}
    g=HeteroData()
    for table in PKS:
        g[table].num_nodes=len(ordered[table])
        g[table].primary_keys=list(indices[table]) # Audit identity, not model features.
    quantity=[r['quantity'] for r in ordered['line_item']]
    if any(type(q) is not int or q<=0 for q in quantity):raise ValueError('Positive integer quantities required')
    g['line_item'].quantity=torch.tensor(quantity,dtype=torch.long)
    for child,role,cols,parent,parentcols,nullable in FKS:
        if parentcols!=PKS[parent]:raise ValueError('This explicit builder references PKs only')
        e=fk_edges(ordered[child],cols,indices[parent],nullable)
        g[child,role,parent].edge_index=e
        g[parent,'rev_'+child+'_'+role,child].edge_index=e.flip(0).clone()
    g.validate(raise_on_error=True)
    return g

def graph_query(g):
    """Join by typed IDs, keep each line, then SUM quantities and count orders."""
    buyer=dict(g['orders','buyer','customer'].edge_index.t().tolist())
    order=dict(g['line_item','order','orders'].edge_index.t().tolist())
    product=dict(g['line_item','product','product'].edge_index.t().tolist())
    paths=[];totals=Counter();order_counts=Counter(buyer.values())
    for line in range(g['line_item'].num_nodes):
        o=order[line];c=buyer[o];p=product[line]
        cid=g['customer'].primary_keys[c][0];pid=g['product'].primary_keys[p][0]
        oid,line_no=g['line_item'].primary_keys[line];q=int(g['line_item'].quantity[line])
        paths.append([cid,oid,line_no,pid,q]);totals[cid,pid]+=q
    counts=[[key[0],order_counts[i]] for i,key in enumerate(g['customer'].primary_keys)]
    return {'paths':sorted(paths),'totals':[[c,p,q] for (c,p),q in sorted(totals.items())],
            'customer_order_counts':counts}

def sql_query(tables):
    """Independent oracle: SQLite performs constraints, joins, SUM and LEFT JOIN."""
    db=sqlite3.connect(':memory:')
    try:
        db.execute('PRAGMA foreign_keys=ON')
        assert db.execute('PRAGMA foreign_keys').fetchone()==(1,)
        db.executescript(DDL)
        for table,cols in COLUMNS.items():
            marks=','.join('?' for _ in cols)
            db.executemany(f'INSERT INTO {table} VALUES ({marks})',
                           [tuple(r[c] for c in cols) for r in tables[table]])
        join='FROM customer c JOIN orders o ON c.customer_id=o.buyer_id JOIN line_item l ON o.order_id=l.order_id JOIN product p ON l.product_id=p.product_id'
        paths=db.execute('SELECT c.customer_id,o.order_id,l.line_no,p.product_id,l.quantity '+join+' ORDER BY 1,2,3,4,5').fetchall()
        totals=db.execute('SELECT c.customer_id,p.product_id,SUM(l.quantity) '+join+' GROUP BY c.customer_id,p.product_id ORDER BY 1,2').fetchall()
        counts=db.execute('SELECT c.customer_id,COUNT(o.order_id) FROM customer c LEFT JOIN orders o ON c.customer_id=o.buyer_id GROUP BY c.customer_id ORDER BY 1').fetchall()
        return {'paths':[list(x) for x in paths],'totals':[list(x) for x in totals],
                'customer_order_counts':[list(x) for x in counts]}
    finally:db.close()

def generated_fixture(seed):
    """Fixed coverage extension: 32 seeds, 7 customers, 6 products, 12 orders."""
    rng=random.Random(seed);cs=[10+11*i for i in range(7)];ps=[10+13*i for i in range(6)]
    tables={'customer':[{'customer_id':v} for v in cs],
            'product':[{'product_id':v} for v in ps], 'orders':[], 'line_item':[]}
    for i in range(12):
        oid=100+7*i
        tables['orders'].append(dict(order_id=oid,buyer_id=rng.choice(cs[:-1]),referrer_id=rng.choice([None]+cs)))
        for j in range(rng.randrange(6)):
            tables['line_item'].append(dict(order_id=oid,line_no=j+1,product_id=rng.choice(ps[:-1]),quantity=rng.randint(1,9)))
    return tables

def run_suite():
    """Execute the complete declared fixture suite; never sample a larger target."""
    records=[]
    for name,t in [('worked',fixture())]+[(f'seed-{s}',generated_fixture(s)) for s in range(32)]:
        g=build_graph(t);actual=graph_query(g);expected=sql_query(t)
        assert actual==expected, name
        records.append({'name':name,'data_sha256':hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest(),
          'nodes':{k:g[k].num_nodes for k in g.node_types},
          'edges':{'|'.join(k):g[k].edge_index.shape[1] for k in g.edge_types},'query':actual})
    return {'status':'PASS','scope':'Complete course fixture suite; no model-paper experiment',
            'databases':len(records),'records':records,'full_paper_parity':'NOT_APPLICABLE_NO_MODEL_PAPER'}
