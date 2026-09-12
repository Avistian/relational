"""Count actual final-paper appendix roster rows, separately from reported study sizes."""
from pathlib import Path
import hashlib,json,re,urllib.request
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 root=ROOT/'data/cache/l067-source';root.mkdir(parents=True,exist_ok=True)
 paper=json.loads((ROOT/'_sources_l067_v2.json').read_text())['paper'];pdf=root/'paper-neurips2024.pdf'
 if not pdf.exists():urllib.request.urlretrieve(paper['url'],pdf)
 assert hashlib.sha256(pdf.read_bytes()).hexdigest()==paper['sha256'],'Pinned final paper changed'
 try:
  import pypdf
 except ImportError:
  raise SystemExit('This optional paper audit requires pypdf: .venv/bin/python -m pip install pypdf==6.18.1')
 text='\n'.join(page.extract_text() for page in pypdf.PdfReader(pdf).pages)
 tables={}
 for name,start,end,reported in [('small','Table 2: 47 Small Datasets','Table 3: 48 Medium/Large Datasets',47),('medium_large','Table 3: 48 Medium/Large Datasets','Table 4: 71 Datasets Selected',48),('deep_subset','Table 4: 71 Datasets Selected','A.2 Experiment Details',71)]:
  a=text.index(start);b=text.index(end,a+len(start));rows=[]
  for line in text[a:b].splitlines():
   m=re.fullmatch(r'(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)',line)
   if m:rows.append(dict(name=m[1],openml_task_id=int(m[2]),instances=int(m[3]),features=int(m[4]),classes=int(m[5]),categorical=int(m[6]),imbalance=float(m[7])))
  tables[name]=dict(caption_count=reported,listed_count=len(rows),rows=rows)
 assert [tables[k]['listed_count'] for k in ['small','medium_large','deep_subset']]==[37,42,60]
 result=dict(status='DISCREPANCY_CONFIRMED',paper_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),extracted_text_sha256=hashlib.sha256(text.encode()).hexdigest(),extractor='pypdf '+pypdf.__version__,tables=tables,listed_main_total=tables['small']['listed_count']+tables['medium_large']['listed_count'],reported_main_total=95,scope='Direct extraction from hash-verified final PDF, with rendered pp15-17 inspected separately; not a reconstruction of experiment logs. This count does not establish which tasks were run or invalidate reported scores.')
 (ROOT/'_paper_audit_l067_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v['listed_count'] for k,v in tables.items()})
