"""Recover and reconstruct every beta task row from checksum-matching archives.

Missing archives return exit 2; mismatched bytes return exit 2. No downloads,
training, cloud launch, or benchmark-score claim occurs here. Full real-data
reconstruction has not run until both historical archives are recovered.
"""
import argparse,contextlib,hashlib,io,json,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'sources/l126/beta'))
import pandas as pd
from relbench.data import Database,Dataset
from relbench.tasks.stackex import EngageTask
from relkit.beta_l126 import unpack_verified,beta_events,beta_cutoffs,engagement_table,align_predictions,average_precision
EXPECTED={'db.zip':'dfb84faa4918c6c4ecac791a69a30a477a7bee097d7295d48c78ceb8f59c997c','engage.zip':'9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18'}

def reconstruct(db, archived):
    dataset=Dataset(db,pd.Timestamp('2019-01-01'),pd.Timestamp('2021-01-01'),[EngageTask])
    task=EngageTask(dataset,process=True)
    cutoffs=beta_cutoffs(dataset.db.min_timestamp)
    report={}
    for split in ['train','val','test']:
        source_db=db if split=='test' else dataset.db
        own=engagement_table(source_db.table_dict['users'].df,beta_events(source_db.table_dict),cutoffs[split])
        with contextlib.redirect_stderr(io.StringIO()):original=task.make_table(source_db,pd.Series(cutoffs[split])).df
        keys=['OwnerUserId','timestamp']
        for name,expected in [('original_source',original),('archived_labels',archived['full_test' if split=='test' else split].df)]:
            predicted=own.rename(columns={'contribution':'score'})
            aligned=align_predictions(expected,predicted,keys)
            if not (aligned==expected.contribution.to_numpy()).all():raise ValueError(f'{split}: {name} label mismatch')
        report[split]={'rows':len(own),'cutoffs':len(cutoffs[split]),'positive':int(own.contribution.sum()),'source_and_archive':'EXACT'}
    masked=archived['test'].df
    if 'contribution' in masked:raise ValueError('Archived public test table exposes labels')
    full=archived['full_test'].df
    align_predictions(masked,full.rename(columns={'contribution':'score'}),['OwnerUserId','timestamp'])
    return {'status':'PASS','historical_full_contract':'COMPLETE_FOR_PINNED_ARCHIVES','splits':report,'historical_paper_identity':'NOT_ESTABLISHED','numerical_paper_target':'NOT_APPLICABLE'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive-dir',required=True,type=Path);ap.add_argument('--report',type=Path);args=ap.parse_args()
    found={name:('MISSING' if not (args.archive_dir/name).exists() else 'MATCH' if hashlib.sha256((args.archive_dir/name).read_bytes()).hexdigest()==digest else 'HASH_MISMATCH') for name,digest in EXPECTED.items()}
    r={'status':'BLOCKED_DATA','files':found,'historical_full_contract':'NOT_RUN'};code=2
    if all(v=='MATCH' for v in found.values()):
        with tempfile.TemporaryDirectory(prefix='l126-recover-') as tmp:
            roots={}
            for name,digest in EXPECTED.items():
                dest=Path(tmp)/Path(name).stem;unpack_verified((args.archive_dir/name).read_bytes(),digest,dest)
                paths=list(dest.rglob('*.parquet'))
                if not paths:raise ValueError(f'{name}: no tables')
                roots[name]=paths[0].parent
            r=reconstruct(Database.load(roots['db.zip']),Database.load(roots['engage.zip']).table_dict);r['files']=found;code=0
    text=json.dumps(r,indent=2)+'\n'
    if args.report:args.report.write_text(text)
    print(text);return code
if __name__=='__main__':raise SystemExit(main())
