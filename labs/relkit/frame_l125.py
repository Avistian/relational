"""Visible L125 mechanisms; course experiment, not Hu Table 2 reproduction.

Frame 0.3.0 reference implementation is pinned in sources/l125/frame.
The four learner functions are also used by the real-data pipeline.
"""
import hashlib
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_frame import stype, NAStrategy
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_frame.nn import (LinearEncoder, EmbeddingEncoder, TimestampEncoder,
    LinearEmbeddingEncoder, MultiCategoricalEmbeddingEncoder, StypeWiseFeatureEncoder)


def numeric_tokens(raw, mean, scale, weight, bias):
    """[B,C] -> [B,C,d]; scale already includes the upstream epsilon."""
    filled=torch.where(torch.isnan(raw),mean,raw)
    return ((filled-mean)/scale).unsqueeze(-1)*weight + bias


def categorical_indices(raw, cardinalities):
    """Disjoint column vocabularies; all negative IDs share padding index 0."""
    sizes=torch.as_tensor(cardinalities,device=raw.device,dtype=torch.long)
    if raw.ndim!=2 or raw.shape[1]!=len(sizes) or (sizes<0).any():
        raise ValueError('Column cardinality mismatch')
    if (raw>=sizes).any():raise ValueError('ID outside fitted vocabulary')
    offsets=torch.cumsum(sizes,0)-sizes
    return torch.where(raw<0,0,raw+offsets+1)


def align_rows(row_ids, encoded, requested_ids):
    """Gather by immutable entity ID; repeated query IDs are legitimate."""
    if len(row_ids)!=len(encoded):raise ValueError('Row count mismatch')
    lookup={key:i for i,key in enumerate(row_ids)}
    if len(lookup)!=len(row_ids):raise ValueError('Duplicate source row ID')
    if any(key not in lookup for key in requested_ids):raise ValueError('Unknown row ID')
    positions=torch.tensor([lookup[key] for key in requested_ids],device=encoded.device,dtype=torch.long)
    return encoded[positions]


def fixed_text_vectors(texts):
    """Deterministic 16-bin token counts: a visible adapter, NOT pretrained text.

    This deliberately avoids network/model dependencies in the course path.
    Collisions and lack of semantics are limitations, not a paper approximation.
    """
    out=torch.zeros(len(texts),16)
    for i,text in enumerate(texts):
        for word in re.findall(r'\w+',str(text).lower()):
            bucket=int.from_bytes(hashlib.sha256(word.encode()).digest()[:4],'little')%16
            out[i,bucket]+=1
    return out


def fit_and_convert(train, query, types):
    """Fit dictionaries/statistics only on the supplied fitting rows."""
    ds=Dataset(train,col_to_stype=types,
               col_to_text_embedder_cfg=TextEmbedderConfig(fixed_text_vectors))
    ds.materialize()
    return ds,ds.convert_to_tensor_frame(query)


def encoder_recipe():
    """Fresh instances per table; text_embedded is grouped under embedding."""
    return {stype.numerical:LinearEncoder(na_strategy=NAStrategy.MEAN),
            stype.categorical:EmbeddingEncoder(),
            stype.multicategorical:MultiCategoricalEmbeddingEncoder(),
            stype.timestamp:TimestampEncoder(),
            stype.embedding:LinearEmbeddingEncoder()}


class VisibleResidualBlock(nn.Module):
    """Frame FCResidualBlock equations, with LayerNorm and configurable dropout."""
    def __init__(self,input_width,width,dropout):
        super().__init__()
        self.lin1=nn.Linear(input_width,width);self.lin2=nn.Linear(width,width)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.dropout=nn.Dropout(dropout)
        self.shortcut=nn.Linear(input_width,width) if input_width!=width else None
    def forward(self,x):
        out=self.dropout(torch.relu(self.norm1(self.lin1(x))))
        out=self.dropout(torch.relu(self.norm2(self.lin2(out))))
        return out+(self.shortcut(x) if self.shortcut is not None else x)


class VisibleRowResNet(nn.Module):
    """Typed tokens -> flatten -> residual blocks -> normalized row readout.

    Same state keys and equations as pinned Frame ResNet (LayerNorm setting).
    Numeric and categorical kernels call the learner's functions. Other typed
    kernels remain the pinned library primitives, shown in the notebook appendix.
    """
    def __init__(self,ds,width=8,out_width=8,layers=2,dropout=0.):
        super().__init__()
        self.encoder=StypeWiseFeatureEncoder(out_channels=width,col_stats=ds.col_stats,
            col_names_dict=ds.tensor_frame.col_names_dict,stype_encoder_dict=encoder_recipe())
        self.backbone=nn.Sequential(*[VisibleResidualBlock(ds.tensor_frame.num_cols*width if i==0 else width,width,dropout) for i in range(layers)])
        self.decoder=nn.Sequential(nn.LayerNorm(width),nn.ReLU(),nn.Linear(width,out_width))
    def tokens(self,tf):
        pieces=[];names=[]
        for kind in tf.stypes:
            columns=self.encoder.col_names_dict[kind]
            enc=self.encoder.encoder_dict[kind.value];raw=tf.feat_dict[kind]
            if kind==stype.numerical:
                out=numeric_tokens(raw,enc.mean,enc.std,enc.weight,enc.bias)
            elif kind==stype.categorical:
                # Offsets are stored in the source encoder; lengths come from stats.
                from torch_frame.data.stats import StatType
                counts=[len(s[StatType.COUNT][0]) for s in enc.stats_list]
                out=enc.emb(categorical_indices(raw,counts))
            else:out=enc(raw,columns)
            pieces.append(out);names.extend(columns)
        return torch.cat(pieces,dim=1),names
    def forward(self,tf):
        tokens,_=self.tokens(tf)
        return self.decoder(self.backbone(tokens.flatten(1)))


def typed_fixture():
    train=pd.DataFrame({'amount':[10.,20.,30.,np.nan], 'region':['north','south','north','south'],
        'note':['red apple','green apple','red pear','green pear'],
        'created':pd.to_datetime(['2024-01-01','2024-01-02','2024-02-01','2024-02-02']),
        'vector':[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.]],
        'tags':[['a','b'],['b'],['c'],['a']]})
    query=train.iloc[[2,0]].copy();query.loc[query.index[0],'region']='unseen';query.loc[query.index[1],'amount']=np.nan
    types={'amount':stype.numerical,'region':stype.categorical,'note':stype.text_embedded,
        'created':stype.timestamp,'vector':stype.embedding,'tags':stype.multicategorical}
    return train,query,types


def mean_messages(source,edges,num_targets):
    """Visible one-relation mean aggregation; edges [2,E] source -> target."""
    out=source.new_zeros((num_targets,source.shape[1]));degree=source.new_zeros(num_targets)
    out.index_add_(0,edges[1],source[edges[0]])
    degree.index_add_(0,edges[1],torch.ones(edges.shape[1],device=source.device))
    return out/degree.clamp_min(1).unsqueeze(-1)


class CourseGraphHead(nn.Module):
    """One typed results -> drivers GraphSAGE relation; no temporal sampler.

    All edges passed here must already be filtered to one cutoff. This is a
    gradient/interface exercise, not the historical two-layer heterogeneous GNN.
    """
    def __init__(self,width):
        super().__init__();self.root=nn.Linear(width,width);self.neighbor=nn.Linear(width,width,bias=False);self.head=nn.Linear(width,1)
    def forward(self,drivers,results,edges,seeds):
        messages=mean_messages(results,edges,len(drivers))
        hidden=torch.relu(self.root(drivers)+self.neighbor(messages))
        return self.head(hidden[seeds]).flatten()

# Explicit teaching schema: identifiers become graph structure, never scalar features.
F1_SCHEMA={
 'constructor_results':{'points':'numerical'},
 'constructor_standings':{'points':'numerical','position':'numerical','wins':'numerical'},
 'circuits':{'circuitRef':'text_embedded','name':'text_embedded','location':'text_embedded','country':'categorical','lat':'numerical','lng':'numerical','alt':'numerical'},
 'qualifying':{'number':'numerical','position':'numerical'},
 'drivers':{'driverRef':'text_embedded','code':'categorical','forename':'text_embedded','surname':'text_embedded','dob':'timestamp','nationality':'categorical'},
 'results':{c:'numerical' for c in ['number','grid','position','positionOrder','points','laps','milliseconds','fastestLap','rank','statusId']},
 'races':{'year':'numerical','round':'numerical','name':'text_embedded','time':'categorical'},
 'standings':{'points':'numerical','position':'numerical','wins':'numerical'},
 'constructors':{'constructorRef':'text_embedded','name':'text_embedded','nationality':'categorical'}}
F1_KEYS={'constructor_results':'constructorResultsId','constructor_standings':'constructorStandingsId','circuits':'circuitId','qualifying':'qualifyId','drivers':'driverId','results':'resultId','races':'raceId','standings':'driverStandingsId','constructors':'constructorId'}


def load_f1_archive(raw):
    """Checksum-verified released data; filter dated tables to Jan 1, 2010."""
    import io,zipfile
    digest='ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Unexpected F1 archive')
    tables={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in F1_SCHEMA:
            path=next(p for p in z.namelist() if p.endswith('/'+name+'.parquet') or p==name+'.parquet')
            df=pd.read_parquet(io.BytesIO(z.read(path)))
            if 'date' in df:df=df.loc[df.date<=pd.Timestamp('2010-01-01')].copy()
            tables[name]=df.reset_index(drop=True)
    return tables


def encode_f1_tables(tables,fit_cutoff='2004-09-03'):
    """All nine tables, width 8, two residual blocks, random initialization.

    Dated-table statistics use rows <= fit_cutoff. Static tables have no creation
    history and are fit in full; therefore point-in-time validity is unestablished.
    Returns exportable vectors and live models/TensorFrames for the gradient lab.
    """
    torch.manual_seed(125);torch.set_num_threads(2)
    models={};frames={};exports={};report={}
    for name,table in tables.items():
        types={c:getattr(stype,t) for c,t in F1_SCHEMA[name].items()}
        features=table[list(types)].copy()
        for c,t in types.items():
            if t==stype.numerical:features[c]=pd.to_numeric(features[c],errors='coerce').astype(float)
            elif t in (stype.categorical,stype.text_embedded):features[c]=features[c].fillna('').astype(str)
        fit_mask=table.date<=pd.Timestamp(fit_cutoff) if 'date' in table else pd.Series(True,index=table.index)
        if not fit_mask.any():raise ValueError('No fitting rows for '+name)
        ds,tf=fit_and_convert(features.loc[fit_mask].copy(),features,types)
        model=VisibleRowResNet(ds).eval()
        with torch.no_grad():
            vectors=torch.cat([model(tf[i:i+1024]) for i in range(0,len(table),1024)])
        if not torch.isfinite(vectors).all():raise ValueError('Nonfinite embeddings: '+name)
        ids=table[F1_KEYS[name]].astype(int).tolist()
        # Prove alignment survives a shuffled export, rather than merely count rows.
        permutation=torch.arange(len(ids)-1,-1,-1)
        restored=align_rows([ids[i] for i in permutation],vectors[permutation],ids)
        torch.testing.assert_close(restored,vectors)
        models[name]=model;frames[name]=tf;exports[name]={'ids':ids,'vectors':vectors}
        report[name]={'rows':len(table),'fit_rows':int(fit_mask.sum()),'columns':{c:t.value for c,t in types.items()},'token_shape':[len(table),len(types),8],'row_shape':list(vectors.shape),'identity_check':'PASS','finite':'PASS','fit_boundary':fit_cutoff if 'date' in table else 'STATIC_HISTORY_UNAVAILABLE'}
    return models,frames,exports,report


def gradient_step(tables,models,frames,task_bytes,cutoff='2004-09-03'):
    """One update on 23 released TRAIN queries, for gradient evidence only."""
    import io,zipfile
    expected='775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e'
    if hashlib.sha256(task_bytes).hexdigest()!=expected:raise ValueError('Unexpected task archive')
    with zipfile.ZipFile(io.BytesIO(task_bytes)) as z:
        task=pd.read_parquet(io.BytesIO(z.read('driver-position/train.parquet')))
    task=task.loc[task.date==pd.Timestamp(cutoff)].copy().sort_values('driverId')
    if len(task)==0:raise ValueError('No training queries')
    driver_ids=tables['drivers'].driverId.astype(int).tolist();lookup={key:i for i,key in enumerate(driver_ids)}
    eligible=(tables['results'].date<=pd.Timestamp(cutoff)) & tables['results'].driverId.notna()
    positions=np.flatnonzero(eligible.to_numpy());r=tables['results'].iloc[positions]
    edges=torch.tensor([list(range(len(r))),[lookup[int(k)] for k in r.driverId]],dtype=torch.long)
    seeds=torch.tensor([lookup[int(k)] for k in task.driverId])
    target=torch.tensor(task.position.to_numpy(),dtype=torch.float32)
    head=CourseGraphHead(8);params=list(models['drivers'].parameters())+list(models['results'].parameters())+list(head.parameters())
    optimizer=torch.optim.Adam(params,lr=.001);optimizer.zero_grad()
    drivers=models['drivers'](frames['drivers']);results=models['results'](frames['results'][torch.as_tensor(positions)])
    prediction=head(drivers,results,edges,seeds);loss=torch.nn.functional.l1_loss(prediction,target);loss.backward()
    norms={name:float(sum(p.grad.abs().sum() for p in models[name].parameters() if p.grad is not None)) for name in ['drivers','results']}
    if not all(v>0 and np.isfinite(v) for v in norms.values()):raise ValueError('Broken encoder gradients')
    before=[p.detach().clone() for p in params];optimizer.step()
    changed=sum(not torch.equal(a,b) for a,b in zip(before,params))
    if not changed:raise ValueError('No parameter update')
    return {'status':'PASS','training_queries':len(task),'cutoff':cutoff,'visible_results':len(r),'max_input_time':str(r.date.max()),'loss_before_one_step':float(loss.detach()),'encoder_gradient_l1':norms,'changed_parameter_tensors':changed,'metric_status':'NOT_A_BENCHMARK','static_history':'UNAVAILABLE'}
