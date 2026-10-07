"""Keep historical GPU identity separate from the corrected reporting boundary."""
import ast,base64,gzip,hashlib,json,zlib
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent

def check():
    archived=(P/'sources/l153/recommendation_before_boundary.py').read_text()
    current=(P/'relkit/recommendation_l153.py').read_text()
    assert current==archived.replace('<=.02 else','<=.02+8*math.ulp(.107) else')
    budget=json.loads((P/'_budget_l153.json').read_text())
    assert hashlib.sha256(archived.encode()).hexdigest()==budget['source_hashes']['labs/relkit/recommendation_l153.py']
    old=gzip.decompress((P/'sources/l153/notebook_before_boundary.py.gz').read_bytes()).decode()
    historical=json.loads((P/'_notebook_l153_results.json').read_text())
    assert hashlib.sha256(old.encode()).hexdigest()==historical['code_sha256']
    book=nbformat.read(P/'solutions/0153-recommendation-portfolio.ipynb',4)
    new='\n\n'.join(c.source for c in book.cells if c.cell_type=='code')
    def remaining(source):
        tree=ast.parse(source);packet=None;kept=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('portfolio_entry','check_portfolio'):continue
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in node.targets):
                encoded=next(n.value for n in ast.walk(node.value) if isinstance(n,ast.Constant) and isinstance(n.value,str))
                packet=json.loads(zlib.decompress(base64.b64decode(encoded)));continue
            kept.append(node)
        return ast.dump(ast.Module(body=kept,type_ignores=[]),include_attributes=False),packet
    old_ast,old_packet=remaining(old);new_ast,new_packet=remaining(new)
    assert old_ast==new_ast,'Notebook change outside reporting boundary/check and source packet'
    assert old_packet.keys()==new_packet.keys()
    for name in old_packet:
        if name=='relkit/recommendation_l153.py':assert old_packet[name]==archived and new_packet[name]==current
        else:assert old_packet[name]==new_packet[name],name
    return dict(status='PASS',historical_code_sha256=historical['code_sha256'],current_code_sha256=hashlib.sha256(new.encode()).hexdigest(),scope='Only paper-score boundary, its live CHECK and embedded copy changed; model/trainer and evidence identical',historical_full_gpu_gate='NOT_RUN; archived pinned default notebook only')

if __name__=='__main__':print(check())
