"""Regression for test-cap foreign keys: source correction must precede graph build."""
import json
from pathlib import Path
import pandas as pd
from relbench.base import Database,Table
from relbench.datasets import get_dataset
users=Table(pd.DataFrame({'Id':pd.Series([0,1],dtype='Int64'),'time':pd.to_datetime(['2020-01-01','2022-01-01'])}),fkey_col_to_pkey_table={},pkey_col='Id',time_col='time')
posts=Table(pd.DataFrame({'Id':pd.Series([0,1],dtype='Int64'),'UserId':pd.Series([0,1],dtype='Int64'),'time':pd.to_datetime(['2020-02-01','2020-02-02'])}),fkey_col_to_pkey_table={'UserId':'users'},pkey_col='Id',time_col='time')
db=Database({'users':users,'posts':posts}).upto(pd.Timestamp('2021-01-01'))
assert db.table_dict['posts'].df.UserId.notna().sum()==2
get_dataset('rel-stack').validate_and_correct_db(db)
assert db.table_dict['posts'].df.UserId.notna().sum()==1
assert pd.isna(db.table_dict['posts'].df.UserId.iloc[1])
r=dict(status='PASS',future_parent_removed=True,reference_nulled=True,remaining_edge=[0,0])
Path(__file__).with_name('_correction_check_l134_results.json').write_text(json.dumps(r,indent=2));print(r)
