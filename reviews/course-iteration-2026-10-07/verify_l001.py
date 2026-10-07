"""Independent executable fixture for the lesson's displayed SQL."""
import sqlite3,json,re,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=(ROOT/'lessons/0001-single-table-assumption.html').read_text()
query=html.unescape(re.search(r'<pre>(WITH order_features.*?)</pre>',s,re.S)[1])
con=sqlite3.connect(':memory:')
con.executescript('''CREATE TABLE customers(customer_id INTEGER,signup_date TEXT);
CREATE TABLE orders(order_id INTEGER,customer_id INTEGER,order_date TEXT,amount REAL);
CREATE TABLE sessions(session_id INTEGER,customer_id INTEGER,start_time TEXT,pages_viewed INTEGER);
CREATE TABLE support_tickets(ticket_id INTEGER,customer_id INTEGER,opened_at TEXT);
INSERT INTO customers VALUES(1,'2025-01-01'),(2,'2025-02-01'),(3,'2025-05-01');
INSERT INTO orders VALUES(1,1,'2025-03-30',20),(2,1,'2025-03-31',30),(3,1,'2025-04-02',999);
INSERT INTO sessions VALUES(1,1,'2025-03-30',4),(2,1,'2025-03-31',8);
INSERT INTO support_tickets VALUES(1,1,'2025-03-28');''')
rows=con.execute(query,{'cutoff':'2025-04-01'}).fetchall()
assert rows==[(1,90,2,6.0,1),(2,59,0,None,0)],rows
naive=con.execute("SELECT COUNT(o.order_id),COUNT(t.ticket_id) FROM customers c JOIN orders o USING(customer_id) JOIN sessions s USING(customer_id) JOIN support_tickets t USING(customer_id) WHERE o.order_date<='2025-04-01'").fetchone()
assert naive==(4,4),naive
# Change only a future event: the visible feature rows must remain byte-for-byte equal.
con.execute('UPDATE orders SET amount=-999 WHERE order_id=3')
assert con.execute(query,{'cutoff':'2025-04-01'}).fetchall()==rows
result={'status':'PASS','feature_rows':rows,'raw_join_counts':naive,'checks':['displayed SQLite query executed','one row per eligible customer','future signup excluded','future order excluded','empty histories retained','NULL average not zero','fanout counterexample']}
Path(__file__).with_name('l001-check.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
