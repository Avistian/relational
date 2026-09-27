"""Offline archive identity preflight. This never launches training or spends money."""
import argparse,hashlib,json
from pathlib import Path

def inspect_archives(folder):
 expected={'db.zip':'deb00ccdf825e569b34935834444429cd1c0074b50226b12d616aab22d36242d','engage.zip':'9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18'}
 checks={}
 for name,digest in expected.items():
  p=Path(folder)/name
  checks[name]='MISSING' if not p.exists() else ('MATCH' if hashlib.sha256(p.read_bytes()).hexdigest()==digest else 'HASH_MISMATCH')
 return {'status':'ARCHIVES_MATCH' if all(x=='MATCH' for x in checks.values()) else 'BLOCKED_DATA','files':checks,'paper_result':'NOT_RUN','protocol_identity':'NOT_ESTABLISHED','next_step':'Recover author-attested Table 2 configuration, environment and text revision before interpreting a candidate run as the published experiment.'}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--archive-dir',type=Path,required=True);parser.add_argument('--report',type=Path)
 args=parser.parse_args();r=inspect_archives(args.archive_dir);s=json.dumps(r,indent=2)+'\n';print(s)
 if args.report:args.report.write_text(s)
 raise SystemExit(0 if r['status']=='ARCHIVES_MATCH' else 2)
