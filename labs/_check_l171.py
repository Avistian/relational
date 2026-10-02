"""Behavioral contracts; injected functions ensure notebook TODOs are live."""
def check171(validate_corpus, split_corpus, audit_tables):
    import copy
    import pandas as pd
    def record(name, families, digest):
        return dict(database=name, source_families=families, archive_sha256=digest*64)
    records=[record('a',['alpha'],'a'),record('b',['beta'],'b'),record('c',['gamma'],'c')]
    assert [x['database'] for x in validate_corpus(list(reversed(records)))]==['a','b','c']
    def rejects(fn, *args):
        try: fn(*args)
        except ValueError: return
        raise AssertionError('invalid input accepted')
    rejects(validate_corpus,records+[records[0]])
    for field,value in [('database',''),('source_families',[]),('source_families',['']),('archive_sha256','wrong')]:
        bad=copy.deepcopy(records);bad[0][field]=value;rejects(validate_corpus,bad)
    assert split_corpus(records,'a')==dict(train=['b','c'],heldout=['a'],quarantine=[])
    # Alias and transitive bridge: name-only or direct-neighbor checks must fail.
    linked=copy.deepcopy(records);linked[1]['source_families']=['alpha','bridge'];linked[2]['source_families']=['bridge']
    assert split_corpus(linked,'a')==dict(train=[],heldout=['a'],quarantine=['b','c'])
    duplicate=copy.deepcopy(records);duplicate[2]['archive_sha256']=duplicate[0]['archive_sha256']
    assert split_corpus(duplicate,'a')['quarantine']==['c']
    rejects(split_corpus,records,'missing')
    tables={
      'parents':dict(df=pd.DataFrame({'id':[1,2]}),pkey_col='id',fkey_col_to_pkey_table={},time_col=None),
      'children':dict(df=pd.DataFrame({'id':[10,11,12,13], 'parent':[1,1,None,9], 'date':pd.to_datetime(['2000-01-01','2005-01-01','2010-01-01',None])}),pkey_col='id',fkey_col_to_pkey_table={'parent':'parents'},time_col='date')}
    report=audit_tables(tables,'2005-01-01','2010-01-01')
    assert report['rows']==6 and report['foreign_key_columns']==1
    assert report['foreign_keys'][0]==dict(table='children',column='parent',target='parents',rows=4,nulls=1,dangling=1,nonnull=3)
    assert report['tables']['children']['time_windows']==dict(before_val=1,val_to_test=1,at_or_after_test=1,nulls=1)
    assert report['tables']['parents']['time_status']=='NO_TIME_COLUMN'
    assert report['integrity']=='FAIL' and report['availability']=='NOT_ESTABLISHED'
    clean=copy.deepcopy(tables);clean['children']['df'].loc[3,'parent']=2
    assert audit_tables(clean,'2005-01-01','2010-01-01')['integrity']=='PASS'
    for values in [[1,1],[1,None]]:
        bad=copy.deepcopy(clean);bad['parents']['df']['id']=values
        assert audit_tables(bad,'2005-01-01','2010-01-01')['integrity']=='FAIL'
    bad=copy.deepcopy(clean);bad['children']['fkey_col_to_pkey_table']['parent']='absent'
    rejects(audit_tables,bad,'2005-01-01','2010-01-01')
    rejects(audit_tables,clean,'2010-01-01','2005-01-01')
    return 'PASS'

if __name__=='__main__':
    from relkit.corpus_l171 import validate_corpus,split_corpus,audit_tables
    print(check171(validate_corpus,split_corpus,audit_tables))
