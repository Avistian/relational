"""Small adversarial contracts, callable against the learner's live functions."""
def check172(fit,encode,tokenize):
 import pandas as pd
 import numpy as np
 def reject(fn):
  try:fn()
  except ValueError:return
  raise AssertionError('Expected ValueError')
 x=pd.Series([10.,20.,30.,999.]);admit=pd.Series([True,True,True,False])
 fitted=fit(x,'number',admit)
 assert fitted['mean']==20 and np.isclose(fitted['scale'],np.sqrt(200/3))
 changed=x.copy();changed.iloc[3]=-9999
 assert fit(changed,'number',admit)==fitted,'Heldout values changed fit'
 out=encode(pd.Series([20.,None,999.]),fitted,[False,False,True])
 assert out['state']==['VALUE','MISSING','MASKED'] and out['payload']==[0.,0.,0.]
 assert encode(pd.Series([20.,None,-42.]),fitted,[False,False,True])==out,'Masked value leaks'
 cat=fit(pd.Series(['b','a','future',None]),'category',[True,True,False,True])
 assert cat['vocabulary']==['a','b']
 assert encode(pd.Series(['b','future',None,'a']),cat,[False,False,False,True])==dict(state=['VALUE','UNKNOWN','MISSING','MASKED'],payload=[2,0,0,0])
 key=fit(pd.Series([9007199254740993,7],dtype='Int64'),'key',[False,False])
 assert encode(pd.Series([9007199254740993,None],dtype='Int64'),key)['payload']==['9007199254740993','']
 empty=fit(x,'number',[False]*4);assert empty['mean']==0 and empty['scale']==1 and empty['fit_nonnull']==0
 const=fit(pd.Series([4.,4.]),'number',[True,True]);assert const['scale']==1
 frame=pd.DataFrame({'n':[10.,20.],'c':['a','b']});schema={'n':{'kind':'number','role':'feature'},'c':{'kind':'category','role':'feature'}}
 a=tokenize(frame,schema,[True,False]);b=tokenize(frame[['c','n']],schema,[True,False]);assert a==b
 assert a['rows']==2 and list(a['columns'])==['c','n']
 reject(lambda:fit(x,'mystery',admit));reject(lambda:fit(x,'number',[True]))
 reject(lambda:fit(x,'number',[1,1,1,0]));reject(lambda:encode(x,fitted,[False]))
 reject(lambda:tokenize(frame,{'n':schema['n']},[True,True]))
 reject(lambda:tokenize(frame,schema,[True,True],{'absent':[False,False]}))
 reject(lambda:fit(pd.Series([float('inf')]),'number',[True]))
 dt=fit(pd.Series(['1970-01-02',None]),'timestamp',[False,False])
 assert encode(pd.Series(['1970-01-02',None]),dt)['payload']==[1.,0.]
 text=fit(pd.Series(['é',None]),'text',[False,False]);assert encode(pd.Series(['é',None]),text)['payload']==['é','']
 return 'PASS'

if __name__=='__main__':
 from relkit.tokenization_l172 import fit_column,encode_column,tokenize_table
 print(check172(fit_column,encode_column,tokenize_table))
